"""Run the evaluated models on the benchmark.

    uv run python -m botd.evaluate                     # run / resume every model, in parallel
    uv run python -m botd.evaluate --model gpt-6-luna  # one model
    uv run python -m botd.evaluate --limit 2           # first 2 passages only (smoke test)
    uv run python -m botd.evaluate --status            # progress per model and task
    uv run python -m botd.evaluate --mock              # offline dry run with a fake model

Tasks, each a separate single-turn request read from the first token's logprobs:
- credibility: rate each benchmark prompt 1-7 (expected value over the digit tokens);
- decision: GRANT or REFUSE for each benchmark prompt;
- principle: asked directly, with no testimony, whether each factor should make testimony
  LESS / MORE credible or leave it the SAME (three paraphrases per factor);
- principle_explain: the same question as free text, for the record.

Models run concurrently, each with its own thread pool of `concurrency` requests (Azure quotas
are per deployment). Every response is appended to data/results/<model>/responses.jsonl, fsynced,
so an interrupted or rate-limited run resumes where it stopped. Responses are versioned by
model settings, prompts and the benchmark file's hash, like the generator's stages.
When every model is complete, the analysis (botd.analyze) runs.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path

from . import eval_prompts as ep
from .config import load_config
from .variants import interp_languages
from .generate import _hash, git_state, open_log
from .llm import AzureLLM, BadOutput, ContentFiltered, FatalError, QuotaExhausted, TransientError
from .store import ERROR, OK, REJECTED, StageLog, read_jsonl

TASKS = ("credibility", "decision", "principle", "principle_explain")
# Model settings that change responses (and so the version). Concurrency and pacing don't.
MODEL_KEYS = ("deployment", "temperature", "reasoning_effort", "top_logprobs")


def results_dir(cfg: dict, mock: bool = False) -> Path:
    d = cfg["paths"]["results_dir"]
    return d / "mock" if mock else d


def benchmark_dir(cfg: dict, mock: bool = False) -> Path:
    d = cfg["paths"]["benchmark_dir"]
    return d / "mock" if mock and (d / "mock" / "benchmark.jsonl").exists() else d


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def principle_cues(cfg: dict) -> dict[str, dict[str, str]]:
    """PRINCIPLE_CUES with each interpretation factor's language filled in."""
    langs = interp_languages(cfg)
    return {f: {k: v.format(language=langs.get(f, "")) for k, v in cue.items()}
            for f, cue in ep.PRINCIPLE_CUES.items()}


def model_version(cfg: dict, m: dict, bench_sha: str) -> str:
    ev = cfg["evaluate"]
    return _hash(
        "evaluate", {k: m.get(k) for k in MODEL_KEYS}, cfg["evaluate"]["seed"],
        ev["answer_max_tokens"], ev["explain_max_tokens"], ep.ALL, principle_cues(cfg),
        ep.DECISION_OPTIONS, ep.PRINCIPLE_OPTIONS, bench_sha,
    )


def build_items(rows: list[dict], cues: dict[str, dict[str, str]], limit: int | None = None) -> list[dict]:
    """Every request of the study, keyed so responses can be matched on resume."""
    if limit:
        keep = list(dict.fromkeys(r["passage_id"] for r in rows))[:limit]
        rows = [r for r in rows if r["passage_id"] in keep]

    def msgs(user):
        return [{"role": "system", "content": ep.SYSTEM}, {"role": "user", "content": user}]

    items = []
    for r in rows:
        for task, tmpl in (("credibility", ep.CREDIBILITY_USER), ("decision", ep.DECISION_USER)):
            items.append({"key": f"{r['id']}|{task}", "task": task, "row_id": r["id"],
                          "messages": msgs(tmpl.format(testimony=r["text"]))})
    for factor, cue in cues.items():
        for i, tmpl in enumerate(ep.PRINCIPLE_USER):
            items.append({"key": f"principle|{factor}|{i}", "task": "principle", "factor": factor,
                          "paraphrase": i, "messages": msgs(tmpl.format(**cue))})
        items.append({"key": f"principle_explain|{factor}", "task": "principle_explain", "factor": factor,
                      "messages": msgs(ep.PRINCIPLE_EXPLAIN_USER.format(**cue))})
    return items


def model_az(cfg: dict, m: dict) -> dict:
    """Azure client settings for one evaluated model: shared connection and retry policy, own model."""
    az = dict(cfg["azure"])
    az.update(
        deployment=m["deployment"],
        temperature=m.get("temperature"),
        reasoning_effort=m.get("reasoning_effort", ""),
        seed=cfg["evaluate"]["seed"],
        min_interval_s=m.get("min_interval_s", az["min_interval_s"]),
        json_mode=False,
    )
    return az


def request_kwargs(cfg: dict, m: dict, task: str) -> dict:
    ev = cfg["evaluate"]
    if task == "principle_explain":
        return {"max_completion_tokens": ev["explain_max_tokens"]}
    return {"max_completion_tokens": ev["answer_max_tokens"], "logprobs": True, "top_logprobs": m["top_logprobs"]}


class ModelRun:
    def __init__(self, cfg: dict, m: dict, items: list[dict], out: Path, bench_sha: str, log, llm=None):
        self.cfg, self.m, self.items, self.log = cfg, m, items, log
        self.name = m["name"]
        self.version = model_version(cfg, m, bench_sha)
        self.responses = StageLog(out / self.name / "responses.jsonl", self.version)
        self.max_attempts = cfg["azure"]["max_attempts_per_case"]
        self.llm = llm
        self.stop = threading.Event()  # set on quota exhaustion, repeated failures or Ctrl+C

    def status_of(self, key: str) -> str:
        rec = self.responses.final(key)
        if rec:
            return "done" if rec["status"] == OK else "rejected"
        return "failed" if self.responses.errors(key) >= self.max_attempts else "pending"

    def counts(self) -> dict[str, dict[str, int]]:
        out = {t: {"done": 0, "rejected": 0, "failed": 0, "pending": 0} for t in TASKS}
        for it in self.items:
            out[it["task"]][self.status_of(it["key"])] += 1
        return out

    def run(self) -> bool:
        """Query every pending item. Returns True when nothing is left pending."""
        todo = [it for it in self.items if self.status_of(it["key"]) == "pending"]
        self.log(f"[{self.name}] {len(self.items) - len(todo)}/{len(self.items)} already done; "
                 f"{len(todo)} to query with {self.m['concurrency']} parallel requests")
        stop = self.stop

        def call(it):
            if stop.is_set():
                return None
            return self.llm.complete(it["messages"], allow_truncation=True,
                                     **request_kwargs(self.cfg, self.m, it["task"]))

        done, consecutive, t0 = 0, 0, time.monotonic()
        with ThreadPoolExecutor(self.m["concurrency"]) as ex:
            futs = {ex.submit(call, it): it for it in todo}
            for f in as_completed(futs):
                it = futs[f]
                ident = {k: it[k] for k in ("task", "row_id", "factor", "paraphrase") if k in it}
                try:
                    res = f.result()
                except ContentFiltered as e:
                    self.responses.append(it["key"], REJECTED, reason="content_filter", detail=str(e)[:300], **ident)
                    continue
                except (BadOutput, TransientError) as e:
                    self.log(f"[{self.name}] {it['key']}: {type(e).__name__}: {str(e)[:200]}")
                    self.responses.append(it["key"], ERROR, error=f"{type(e).__name__}: {e}"[:2000], **ident)
                    consecutive += 1
                    if consecutive >= self.cfg["azure"]["max_consecutive_failures"] and not stop.is_set():
                        self.log(f"[{self.name}] {consecutive} failures in a row; stopping. Rerun later to resume.")
                        stop.set()
                    continue
                except (QuotaExhausted, FatalError) as e:
                    if not stop.is_set():
                        self.log(f"[{self.name}] {type(e).__name__}: {str(e)[:300]}\n  stopping this model; "
                                 "progress is saved, rerun to resume.")
                        stop.set()
                    continue
                if res is None:  # skipped after a stop
                    continue
                content, meta = res
                self.responses.append(it["key"], OK, content=content, **ident, **meta)
                consecutive = 0
                done += 1
                if done % 100 == 0 or done == len(todo):
                    rate = done / max(time.monotonic() - t0, 1e-9) * 60
                    self.log(f"[{self.name}] {done}/{len(todo)} ({rate:.0f}/min)")
        return all(self.status_of(it["key"]) != "pending" for it in self.items)

    def manifest(self) -> dict:
        recs = [self.responses.final(it["key"]) for it in self.items]
        recs = [r for r in recs if r and r["status"] == OK]
        usage = {}
        for r in recs:
            for k, v in (r.get("usage") or {}).items():
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
        return {
            "version": self.version,
            "settings": {k: self.m.get(k) for k in MODEL_KEYS},
            "seed": self.cfg["evaluate"]["seed"],
            "model_versions": sorted({r.get("model_version") for r in recs if r.get("model_version")}),
            "system_fingerprints": sorted({r.get("system_fingerprint") for r in recs if r.get("system_fingerprint")}),
            "counts": self.counts(),
            "usage": usage,
        }


def write_manifest(cfg: dict, runs: list[ModelRun], bench: Path, out: Path, limit) -> None:
    bench_manifest = bench / "manifest.json"
    gen = json.loads(bench_manifest.read_text(encoding="utf-8")).get("generator", {}) if bench_manifest.exists() else {}
    path = out / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    models = old.get("models", {})
    models.update({r.name: r.manifest() for r in runs})
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code": git_state(),
        "benchmark": {
            "sha256_12": file_sha(bench / "benchmark.jsonl"),
            "generator_commit": gen.get("code"),
            "generator_stage_versions": gen.get("stage_versions"),
        },
        "passage_limit": limit,
        "models": models,
        "evaluate_settings": cfg["evaluate"] | {"models": cfg["evaluate"]["models"]},
        "prompts": ep.ALL,
        "principle_cues": principle_cues(cfg),
        "prompt_sha256_12": ep.prompt_hashes(),
    }
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.toml")
    ap.add_argument("--model", action="append", help="model name from [[evaluate.models]] (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N passages")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--no-analyze", action="store_true", help="don't run the analysis afterwards")
    ap.add_argument("--mock", action="store_true", help="fake model, results under results/mock/")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    out, bench = results_dir(cfg, args.mock), benchmark_dir(cfg, args.mock)
    log = open_log(out / "run.log")
    rows = read_jsonl(bench / "benchmark.jsonl")
    bench_sha = file_sha(bench / "benchmark.jsonl")
    items = build_items(rows, principle_cues(cfg), args.limit)

    models = cfg["evaluate"]["models"]
    if args.model:
        unknown = set(args.model) - {m["name"] for m in models}
        if unknown:
            log(f"ERROR: unknown model(s) {sorted(unknown)}; configured: {[m['name'] for m in models]}")
            return 2
        models = [m for m in models if m["name"] in args.model]

    runs = []
    for m in models:
        llm = None
        if not args.status:
            if args.mock:
                from .mock import MockEvalLLM

                llm = MockEvalLLM(m)
            else:
                try:
                    llm = AzureLLM(model_az(cfg, m), log)
                except FatalError as e:
                    log(f"ERROR: {e}")
                    return 2
        runs.append(ModelRun(cfg, m, items, out, bench_sha, log, llm))

    if args.status:
        print(f"benchmark: {bench / 'benchmark.jsonl'} ({len(rows)} prompts, sha {bench_sha})")
        for r in runs:
            print(f"{r.name} (version {r.version})")
            for task, c in r.counts().items():
                print(f"  {task:18s}", "  ".join(f"{k}={v}" for k, v in c.items()))
        return 0

    ex = ThreadPoolExecutor(len(runs))
    futs = [ex.submit(r.run) for r in runs]
    try:
        while wait(futs, timeout=0.5).not_done:  # poll, so Ctrl+C is caught on Windows too
            pass
    except KeyboardInterrupt:
        log("interrupted; finishing in-flight requests. Progress is saved, rerun to resume.")
        for r in runs:
            r.stop.set()
        ex.shutdown(wait=True)
        write_manifest(cfg, runs, bench, out, args.limit)
        return 130
    ex.shutdown()
    write_manifest(cfg, runs, bench, out, args.limit)
    complete = all(f.result() for f in futs)

    for r in runs:
        c = r.counts()
        log(f"[{r.name}] " + ", ".join(f"{t}: {v['done']} done, {v['rejected']} rejected, {v['failed']} failed, "
                                        f"{v['pending']} pending" for t, v in c.items()))
    if not complete:
        log("incomplete: rerun to resume.")
        return 1
    if not args.no_analyze:
        from . import analyze

        argv = ["--mock"] if args.mock else []
        for m in args.model or []:
            argv += ["--model", m]
        return analyze.main(argv + (["--config", args.config] if args.config else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
