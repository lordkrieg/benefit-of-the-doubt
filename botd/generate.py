"""Generate the matched-variant credibility benchmark from AsyLex.

    uv run python -m botd.generate              # run / resume everything
    uv run python -m botd.generate --status     # progress per stage
    uv run python -m botd.generate --stage assemble
    uv run python -m botd.generate --mock       # offline dry run with a fake LLM

Stages: prepare -> extract (screen + rewrite) -> annotate (hedging / religious edits)
-> leakcheck -> assemble. Each LLM stage appends one fsynced JSONL record per case
under work_dir, so an interrupted or rate-limited run resumes where it stopped.
"""

import argparse
import csv
import hashlib
import inspect
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from . import asylex, prompts, variants
from .config import ROOT, load_config, public_config
from .llm import AzureLLM, BadOutput, ContentFiltered, FatalError, QuotaExhausted, TransientError
from .store import ERROR, OK, REJECTED, StageLog, read_jsonl, write_jsonl

STAGES = ("extract", "annotate", "leakcheck")


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def clean(text: str) -> str:
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _hash(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _src(*objs) -> list[str]:
    return [inspect.getsource(o) for o in objs]


def pool_version(cfg: dict) -> str:
    """Everything that determines the candidate pool."""
    return _hash("pool", cfg["dataset"], cfg["sample"], inspect.getsource(asylex))


def stage_versions(cfg: dict) -> dict[str, str]:
    """A version per LLM stage: its prompts, validators, settings, model, and upstream stage.

    Records from any other version are ignored, so a change reruns exactly the stages it
    affects. Operational settings (pacing, retries, timeouts, output-token cap) are excluded.
    """
    az = {k: cfg["azure"].get(k) for k in ("deployment", "temperature", "reasoning_effort")}
    extract = _hash(
        "extract", az, cfg["passage"], cfg["dataset"]["max_chars"],
        [prompts.EXTRACT_SYSTEM, prompts.EXTRACT_USER, prompts.RELIGION_RULE, prompts.COUNTRY_RULE],
        [variants.BASELINE_HEDGES.pattern, variants.LEAK_TERMS.pattern, variants.BASELINE_RELIGIOUS],
        _src(variants.validate_extract, Pipeline.extract, clean),
    )
    annotate = _hash(
        "annotate", extract, az, cfg["variants"], cfg["names"],
        [prompts.ANNOTATE_SYSTEM, prompts.ANNOTATE_USER],
        [variants.DETAIL.pattern, variants.HEDGED_ACTOR.pattern, variants.ADDED_ACTS.pattern],
        _src(variants.validate_annotate, variants._check_hedge, variants._check_religious, variants._insertions,
             variants._locate, variants.build_versions, variants.diff_check, Pipeline.annotate),
    )
    leakcheck = _hash(
        "leakcheck", extract, az, cfg["names"],
        [prompts.LEAKCHECK_SYSTEM, prompts.LEAKCHECK_USER],
        _src(Pipeline.leakcheck),
    )
    return {"extract": extract, "annotate": annotate, "leakcheck": leakcheck}


def ensure_candidates(cfg: dict, work: Path, log, rebuild: bool = False) -> list[dict]:
    path, meta_path = work / "candidates.jsonl", work / "candidates.meta.json"
    version = pool_version(cfg)
    if path.exists() and meta_path.exists() and not rebuild:
        if json.loads(meta_path.read_text(encoding="utf-8")).get("version") == version:
            return read_jsonl(path)
        log("dataset/sample settings or sampling code changed; rebuilding the candidate pool")
    ds = cfg["dataset"]
    paths = asylex.download(ds["hf_repo"], ds["hf_revision"], cfg["paths"]["raw_dir"], cfg["hf_token"], log)
    log("scanning AsyLex decisions (takes a minute)...")
    cands = asylex.build_candidates(paths, cfg, log)
    write_jsonl(path, cands)
    meta_path.write_text(json.dumps({"version": version, "n": len(cands)}), encoding="utf-8")
    log(f"wrote {len(cands)} candidates to {path}")
    return cands


def git_state() -> dict:
    """Commit of the generating code, and whether the code or config had uncommitted changes."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    tracked = ("botd", "config.toml", "pyproject.toml", "uv.lock")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", *tracked))}


class Pipeline:
    def __init__(self, cfg: dict, llm, work: Path, log):
        self.cfg, self.llm, self.log = cfg, llm, log
        self.versions = stage_versions(cfg)
        self.logs = {s: StageLog(work / f"{s}.jsonl", self.versions[s]) for s in STAGES}
        self.max_attempts = cfg["azure"]["max_attempts_per_case"]

    def _run(self, stage: str, cid: str, fn) -> dict | str:
        """Run one stage for one case. Returns its terminal record, or "error"/"failed"."""
        log = self.logs[stage]
        if rec := log.final(cid):
            return rec
        if log.errors(cid) >= self.max_attempts:
            return "failed"
        try:
            status, fields = fn()
        except ContentFiltered as e:
            return log.append(cid, REJECTED, reason="content_filter", detail=str(e)[:300])
        except (BadOutput, TransientError) as e:
            self.log(f"  {stage} {cid}: {type(e).__name__}: {str(e)[:300]}")
            log.append(cid, ERROR, error=f"{type(e).__name__}: {e}"[:2000])
            return "error"
        return log.append(cid, status, **fields)

    def extract(self, case: dict):
        cfg = self.cfg
        pc = cfg["passage"]
        user = prompts.EXTRACT_USER.format(
            case_id=case["case_id"],
            min_words=pc["min_words"],
            max_words=pc["max_words"],
            religion_rule=prompts.RELIGION_RULE if pc["require_religion_compatible"] else "",
            country_rule=prompts.COUNTRY_RULE if pc["generalize_country"] else "",
            text=clean(case["text"])[: cfg["dataset"]["max_chars"]],
        )
        msgs = [{"role": "system", "content": prompts.EXTRACT_SYSTEM}, {"role": "user", "content": user}]
        obj, meta = self.llm.chat_json(msgs, lambda o: variants.validate_extract(o, cfg))
        fields = {
            "usable": obj["usable"],
            "reject_reason": obj.get("reject_reason", ""),
            "claimant_gender": obj.get("claimant_gender", "unknown"),
            "country_of_origin": obj.get("country_of_origin", "unknown"),
            "persecution_ground": obj.get("persecution_ground", ""),
            "claimant_religion": obj.get("claimant_religion", "unknown"),
            "outcome_in_decision": obj.get("outcome_in_decision", "unclear"),
            "template": obj.get("testimony", "") if obj["usable"] else "",
            "meta": meta,
        }
        if not obj["usable"]:
            fields["reason"] = "screened_out"
            return REJECTED, fields
        # Guard against AsyLex label errors: the decision text must agree with the label.
        label = "granted" if case["outcome"] == 1 else "refused"
        if fields["outcome_in_decision"] != label:
            fields["reason"] = "label_mismatch"
            return REJECTED, fields
        return OK, fields

    def annotate(self, case: dict, ext: dict):
        cfg, vc = self.cfg, self.cfg["variants"]
        template = ext["template"]
        pair = variants.pick_names(case["case_id"], ext["claimant_gender"], cfg)
        user = prompts.ANNOTATE_USER.format(
            h_min=vc["hedging"]["min_edits"],
            h_max=vc["hedging"]["max_edits"],
            h_markers=", ".join(f'"{m}"' for m in vc["hedging"]["markers"]),
            h_added=vc["hedging"]["max_added_words"],
            r_added=vc["religious"]["max_added_words"],
            r_min=vc["religious"]["min_edits"],
            r_max=vc["religious"]["max_edits"],
            r_lexicon=", ".join(vc["religious"]["lexicon"]),
            testimony=template,
        )
        msgs = [{"role": "system", "content": prompts.ANNOTATE_SYSTEM}, {"role": "user", "content": user}]
        obj, meta = self.llm.chat_json(msgs, lambda o: variants.validate_annotate(o, template, cfg, pair))
        return OK, {"hedging": obj["hedging"], "religious": obj["religious"], "meta": meta}

    def leakcheck(self, case: dict, ext: dict):
        pair = variants.pick_names(case["case_id"], ext["claimant_gender"], self.cfg)
        b = pair["baseline"]
        text = ext["template"].replace("{FULL_NAME}", f"{b['given']} {b['family']}").replace("{GIVEN_NAME}", b["given"])
        msgs = [
            {"role": "system", "content": prompts.LEAKCHECK_SYSTEM},
            {"role": "user", "content": prompts.LEAKCHECK_USER.format(testimony=text)},
        ]

        def validate(o):
            return [] if isinstance(o.get("leak"), bool) else ['"leak" must be true or false']

        obj, meta = self.llm.chat_json(msgs, validate)
        fields = {"leak": obj["leak"], "issues": obj.get("issues", []), "meta": meta}
        if obj["leak"]:
            fields["reason"] = "leak"
            return REJECTED, fields
        return OK, fields

    def gate(self, ext: dict) -> str | None:
        """Deterministic checks on an accepted extraction. Returns a rejection reason or None."""
        if self.cfg["passage"]["generalize_country"] and variants.names_origin(ext["template"], ext["country_of_origin"]):
            return "country_named"
        return None

    def process(self, case: dict) -> str:
        """Returns "accepted", "rejected", "error" (retry later) or "failed" (attempts exhausted)."""
        cid = case["case_id"]
        ext = self._run("extract", cid, lambda: self.extract(case))
        if isinstance(ext, str):
            return ext
        if ext["status"] == REJECTED or self.gate(ext):
            return "rejected"
        for stage, fn in (("annotate", self.annotate), ("leakcheck", self.leakcheck)):
            rec = self._run(stage, cid, lambda fn=fn: fn(case, ext))
            if isinstance(rec, str):
                return rec
            if rec["status"] == REJECTED:
                return "rejected"
        return "accepted"

    def status_of(self, cid: str) -> str:
        """Status from the logs alone, without calling the LLM."""
        for stage in STAGES:
            rec = self.logs[stage].final(cid)
            if rec is None:
                return "failed" if self.logs[stage].errors(cid) >= self.max_attempts else "pending"
            if rec["status"] == REJECTED or (stage == "extract" and self.gate(rec)):
                return "rejected"
        return "accepted"


def targets(cfg: dict) -> dict[int, int]:
    return {1: cfg["sample"]["n_granted"], 0: cfg["sample"]["n_refused"]}


def run_generation(pipe: Pipeline, cands: list[dict], cfg: dict, log) -> bool:
    """Process candidates until each label hits its target. Returns True when complete."""
    target = targets(cfg)
    accepted = {0: 0, 1: 0}
    for c in cands:
        if pipe.status_of(c["case_id"]) == "accepted":
            accepted[c["outcome"]] += 1
    log(f"resuming: {accepted[1]}/{target[1]} granted, {accepted[0]}/{target[0]} refused accepted")

    consecutive = 0
    for case in cands:
        o = case["outcome"]
        if accepted[o] >= target[o]:
            continue
        if pipe.status_of(case["case_id"]) in ("accepted", "rejected", "failed"):
            continue
        log(f"case {case['case_id']} ({'granted' if o else 'refused'}, {case['citation']})")
        status = pipe.process(case)
        log(f"  -> {status}")
        if status == "accepted":
            accepted[o] += 1
            log(f"  progress: {accepted[1]}/{target[1]} granted, {accepted[0]}/{target[0]} refused")
        if status == "error":
            consecutive += 1
            if consecutive >= cfg["azure"]["max_consecutive_failures"]:
                log(f"{consecutive} failed cases in a row; stopping. Rerun later to resume.")
                return False
        else:
            consecutive = 0
        if accepted[0] >= target[0] and accepted[1] >= target[1]:
            break

    done = accepted[0] >= target[0] and accepted[1] >= target[1]
    if not done:
        pending = sum(pipe.status_of(c["case_id"]) == "pending" for c in cands)
        if pending:
            log(f"incomplete: {pending} candidates still pending (errors); rerun to retry them.")
        else:
            log("candidate pool exhausted before reaching the target; raise sample.candidate_pool_multiplier "
                "and rerun with --rebuild-candidates.")
    return done


def assemble(pipe: Pipeline, cands: list[dict], cfg: dict, out: Path, log) -> None:
    target = targets(cfg)
    chosen = {0: [], 1: []}
    for c in cands:
        if len(chosen[c["outcome"]]) < target[c["outcome"]] and pipe.status_of(c["case_id"]) == "accepted":
            chosen[c["outcome"]].append(c)
    cases = sorted(chosen[1] + chosen[0], key=lambda c: (c["rank"], -c["outcome"]))
    log(f"assembling {len(chosen[1])} granted + {len(chosen[0])} refused passages")

    passages, rows, model_versions, fingerprints = [], [], set(), set()
    for case in cases:
        cid = case["case_id"]
        ext, ann, leak = (pipe.logs[s].final(cid) for s in STAGES)
        pair = variants.pick_names(cid, ext["claimant_gender"], cfg)
        edits = {"hedging": ann["hedging"], "religious": ann["religious"]}
        versions = variants.build_versions(ext["template"], edits, pair, cfg)
        checks = variants.diff_check(versions, ext["template"], edits, pair, cfg)
        if any(v != "ok" for v in checks.values()):
            log(f"  WARNING: {cid} failed diff check {checks}; skipped")
            continue
        for rec in (ext, ann, leak):
            model_versions.add(rec["meta"].get("model_version"))
            fingerprints.add(rec["meta"].get("system_fingerprint"))
        passages.append(
            {
                "passage_id": f"asylex-{cid}",
                "case_id": cid,
                "citation": case["citation"],
                "year": case["year"],
                "division": case["division"],
                "outcome": "granted" if case["outcome"] == 1 else "refused",
                "label_source": case["label_source"],
                "asylex_label": {1: "granted", 0: "refused", None: None}[case.get("asylex_label")],
                "claimant_gender": ext["claimant_gender"],
                "country_of_origin": ext["country_of_origin"],
                "persecution_ground": ext["persecution_ground"],
                "claimant_religion": ext["claimant_religion"],
                "outcome_in_decision": ext["outcome_in_decision"],
                "names": pair,
                "template": ext["template"],
                "edits": edits,
                "word_count": len(versions["baseline"].split()),
            }
        )
        for variant, factor, contrast in variants.VERSIONS:
            rows.append(
                {
                    "id": f"asylex-{cid}-{variant}",
                    "passage_id": f"asylex-{cid}",
                    "case_id": cid,
                    "variant": variant,
                    "factor": factor,
                    "contrast_with": f"asylex-{cid}-{contrast}" if contrast else None,
                    "changed_words": variants.changed_words(versions[contrast], versions[variant]) if contrast else 0,
                    "outcome": "granted" if case["outcome"] == 1 else "refused",
                    "text": versions[variant],
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "passages.jsonl", passages)
    write_jsonl(out / "benchmark.jsonl", rows)

    rng = random.Random(cfg["dataset"]["seed"])
    k = max(1, round(len(passages) * cfg["sample"]["manual_review_fraction"])) if passages else 0
    review_ids = {p["passage_id"] for p in rng.sample(passages, k)}
    tmp = out / "manual_review.csv.tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "passage_id", "variant", "factor", "contrast_with", "outcome", "text",
                    "only_target_changed (y/n)", "no_leak (y/n)", "faithful_to_decision (y/n)", "notes"])
        for r in rows:
            if r["passage_id"] in review_ids:
                w.writerow([r["id"], r["passage_id"], r["variant"], r["factor"], r["contrast_with"], r["outcome"],
                            r["text"], "", "", "", ""])
    tmp.replace(out / "manual_review.csv")

    rejections = {}
    for c in cands:
        for s in STAGES:
            rec = pipe.logs[s].final(c["case_id"])
            reason = None
            if rec and rec["status"] == REJECTED:
                reason = rec.get("reason", "other")
            elif rec and s == "extract":
                reason = pipe.gate(rec)
            if reason:
                rejections[reason] = rejections.get(reason, 0) + 1
                break
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {
            "dataset": f"https://huggingface.co/datasets/{cfg['dataset']['hf_repo']}",
            "citation": "Barale, Rovatsos & Bhuta (2023). Automated Refugee Case Analysis: An NLP Pipeline "
            "for Supporting Legal Practitioners. Findings of ACL 2023.",
            "license": "CC BY-NC-SA 4.0 (research use only); this derived dataset uses the same licence.",
        },
        "generator": {
            "code": git_state(),
            "asylex_revision": cfg["dataset"]["hf_revision"],
            "candidate_pool_version": pool_version(cfg),
            "stage_versions": pipe.versions,
            "deployment": cfg["azure"]["deployment"],
            "model_versions": sorted(v for v in model_versions if v),
            "system_fingerprints": sorted(f for f in fingerprints if f),
            "prompts": prompts.ALL,
            "prompt_sha256_12": prompts.prompt_hashes(),
        },
        "counts": {
            "passages": len(passages),
            "granted": sum(p["outcome"] == "granted" for p in passages),
            "refused": sum(p["outcome"] == "refused" for p in passages),
            "prompts": len(rows),
            "manual_review_passages": len(review_ids),
            "candidates_rejected_by_reason": rejections,
        },
        "variants": [{"variant": v, "factor": f, "contrast_with": c} for v, f, c in variants.VERSIONS],
        "config": public_config(cfg),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {len(passages)} passages / {len(rows)} prompts to {out}")


def print_status(pipe: Pipeline, cands: list[dict], cfg: dict) -> None:
    counts: dict[tuple, int] = {}
    for c in cands:
        key = ("granted" if c["outcome"] else "refused", pipe.status_of(c["case_id"]))
        counts[key] = counts.get(key, 0) + 1
    t = targets(cfg)
    print(f"candidates: {len(cands)}  targets: {t[1]} granted / {t[0]} refused")
    for label in ("granted", "refused"):
        print(f"  {label:8s}", "  ".join(f"{s}={counts.get((label, s), 0)}"
                                         for s in ("accepted", "rejected", "failed", "pending")))
    for s in STAGES:
        recs = pipe.logs[s].records
        n = {k: sum(r["status"] == k for rs in recs.values() for r in rs) for k in (OK, REJECTED, ERROR)}
        print(f"  {s:10s} ok={n[OK]} rejected={n[REJECTED]} error-records={n[ERROR]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.toml")
    ap.add_argument("--stage", choices=["all", "prepare", "generate", "assemble"], default="all")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--rebuild-candidates", action="store_true", help="re-scan AsyLex and redraw the pool")
    ap.add_argument("--mock", action="store_true", help="use a fake LLM and separate mock/ dirs (pipeline test)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    work, out = cfg["paths"]["work_dir"], cfg["paths"]["output_dir"]
    if args.mock:
        work, out = work / "mock", out / "mock"
    log = Logger(work / "run.log")

    cands = ensure_candidates(cfg, cfg["paths"]["work_dir"], log, args.rebuild_candidates)
    if args.stage == "prepare":
        return 0

    llm = None
    if args.stage in ("all", "generate") and not args.status:
        if args.mock:
            from .mock import MockLLM

            llm = MockLLM(cfg["azure"], {c["case_id"]: c["outcome"] for c in cands})
        else:
            try:
                llm = AzureLLM(cfg["azure"], log)
            except FatalError as e:
                log(f"ERROR: {e}")
                return 2
    pipe = Pipeline(cfg, llm, work, log)

    if args.status:
        print_status(pipe, cands, cfg)
        return 0

    if args.stage in ("all", "generate"):
        try:
            done = run_generation(pipe, cands, cfg, log)
        except QuotaExhausted as e:
            log(f"quota exhausted: {e}\nProgress is saved; rerun later to resume.")
            done = False
        except FatalError as e:
            log(f"FATAL: {e}")
            return 2
        except KeyboardInterrupt:
            log("interrupted; progress is saved. Rerun to resume.")
            return 130
        if not done and args.stage == "all":
            log("not assembling yet (targets not met). Use --stage assemble to write a partial dataset.")
            return 1

    assemble(pipe, cands, cfg, out, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
