"""Analyse the evaluation responses: paired effects per factor, decision flips, stated principles.

    uv run python -m botd.analyze            # all models with responses
    uv run python -m botd.analyze --mock

Scores are recomputed from the raw first-token logprobs in data/results/<model>/responses.jsonl
(see botd.scoring). Outputs go to data/results/analysis/:
- summary.json: every statistic below, per model;
- effects.csv: one row per model x factor;
- scores.csv: one row per model x benchmark prompt (credibility score, P(grant), answer mass);
- credibility_effects.png, grant_effects.png, decision_flips.png.

Per factor, each variant is compared with its contrast version of the same passage (baseline,
or the consistently spelled Somali name for name_spelling):
- credibility: mean paired change in expected rating (1-7), bootstrap 95% CI over passages,
  Wilcoxon signed-rank test, Holm-corrected across the factors;
- decision: mean paired change in P(grant), same tests; flip rate of the argmax decision,
  with an exact binomial test on the direction of the flips.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from . import eval_prompts as ep
from . import scoring
from .config import load_config
from .evaluate import ModelRun, benchmark_dir, build_items, file_sha, principle_cues, results_dir
from .store import OK, read_jsonl
from .variants import interp_languages

# Report labels; {language} is the factor's interpreter language from config.toml.
FACTORS = {
    "name": "Somali name",
    "religious_vocabulary": "Islamic vocabulary",
    "interpretation": "{language} interpreter",
    "interpretation_other": "{language} interpreter",
    "name_spelling": "Two spellings of the name",
}


def factor_labels(cfg: dict) -> dict[str, str]:
    langs = interp_languages(cfg)
    return {f: label.format(language=langs.get(f, "")) for f, label in FACTORS.items()}


def bootstrap_ci(x: np.ndarray, rng, n: int) -> list[float] | None:
    """95% percentile bootstrap CI of the mean."""
    if len(x) < 2:
        return None
    if np.all(x == x[0]):  # no spread: scipy would warn and return NaN
        return [float(x[0]), float(x[0])]
    ci = stats.bootstrap((x,), np.mean, n_resamples=n, method="percentile", rng=rng).confidence_interval
    return [float(ci.low), float(ci.high)]


def wilcoxon_p(d: np.ndarray) -> float | None:
    if len(d) < 2 or not np.any(d != 0):
        return None
    return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def holm(ps: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted((p, k) for k, p in ps.items() if p is not None)
    out, running = {k: None for k in ps}, 0.0
    for i, (p, k) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - i) * p))
        out[k] = running
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if not len(pos) or not len(neg):
        return None
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def score_rows(run: ModelRun, rows: list[dict], min_mass: float) -> dict[str, dict]:
    """Per benchmark prompt: credibility score and P(grant), or None where the answer was unusable."""
    out = {}
    for r in rows:
        s = {"credibility": None, "credibility_mass": None, "p_grant": None, "decision_mass": None}
        rec = run.responses.final(f"{r['id']}|credibility")
        if rec and rec["status"] == OK:
            rt = scoring.rating(rec.get("logprobs"))
            s["credibility_mass"] = rt["mass"]
            if rt["mass"] >= min_mass:
                s["credibility"] = rt["score"]
        rec = run.responses.final(f"{r['id']}|decision")
        if rec and rec["status"] == OK:
            ch = scoring.choice(rec.get("logprobs"), ep.DECISION_OPTIONS)
            s["decision_mass"] = ch["mass"]
            if ch["mass"] >= min_mass:
                s["p_grant"] = ch["probs"]["grant"]
        out[r["id"]] = s
    return out


def factor_effects(rows: list[dict], scores: dict, rng, n_boot: int) -> dict[str, dict]:
    effects = {}
    for factor in FACTORS:
        pairs = [(scores[r["id"]], scores[r["contrast_with"]]) for r in rows if r["factor"] == factor]
        d_cred = np.array([v["credibility"] - c["credibility"] for v, c in pairs
                           if v["credibility"] is not None and c["credibility"] is not None])
        dec = [(v["p_grant"], c["p_grant"]) for v, c in pairs if v["p_grant"] is not None and c["p_grant"] is not None]
        d_grant = np.array([v - c for v, c in dec])
        g_to_r = sum(c >= 0.5 > v for v, c in dec)
        r_to_g = sum(v >= 0.5 > c for v, c in dec)
        flips = np.array([(v >= 0.5) != (c >= 0.5) for v, c in dec], dtype=float)
        effects[factor] = {
            "n_pairs": len(pairs),
            "credibility": {
                "n": len(d_cred),
                "mean_change": float(d_cred.mean()) if len(d_cred) else None,
                "ci95": bootstrap_ci(d_cred, rng, n_boot),
                "median_change": float(np.median(d_cred)) if len(d_cred) else None,
                "share_lower": float((d_cred < 0).mean()) if len(d_cred) else None,
                "share_higher": float((d_cred > 0).mean()) if len(d_cred) else None,
                "cohens_dz": float(d_cred.mean() / d_cred.std(ddof=1)) if len(d_cred) > 1 and d_cred.std(ddof=1) > 0 else None,
                "wilcoxon_p": wilcoxon_p(d_cred),
            },
            "decision": {
                "n": len(dec),
                "mean_change_p_grant": float(d_grant.mean()) if len(d_grant) else None,
                "ci95": bootstrap_ci(d_grant, rng, n_boot),
                "wilcoxon_p": wilcoxon_p(d_grant),
                "flip_rate": float(flips.mean()) if len(flips) else None,
                "flip_rate_ci95": bootstrap_ci(flips, rng, n_boot),
                "grant_to_refuse": int(g_to_r),
                "refuse_to_grant": int(r_to_g),
                "flip_direction_p": float(stats.binomtest(g_to_r, g_to_r + r_to_g).pvalue) if g_to_r + r_to_g else None,
            },
        }
    for task, key in (("credibility", "wilcoxon_p"), ("decision", "wilcoxon_p")):
        adj = holm({f: e[task][key] for f, e in effects.items()})
        for f, e in effects.items():
            e[task]["wilcoxon_p_holm"] = adj[f]
    return effects


def baseline_validity(rows: list[dict], scores: dict) -> dict:
    """How the model's baseline judgments relate to the tribunal's real outcomes."""
    base = [(scores[r["id"]], r["outcome"]) for r in rows if r["variant"] == "baseline"]
    out = {}
    for key in ("credibility", "p_grant"):
        g = np.array([s[key] for s, o in base if o == "granted" and s[key] is not None])
        f = np.array([s[key] for s, o in base if o == "refused" and s[key] is not None])
        out[key] = {
            "mean_granted": float(g.mean()) if len(g) else None,
            "mean_refused": float(f.mean()) if len(f) else None,
            "auc_vs_outcome": auc(g, f),
        }
    dec = [(s["p_grant"] >= 0.5, o == "granted") for s, o in base if s["p_grant"] is not None]
    out["decision"] = {
        "n": len(dec),
        "accuracy_vs_outcome": float(np.mean([a == b for a, b in dec])) if dec else None,
        "grant_rate": float(np.mean([a for a, _ in dec])) if dec else None,
        "real_grant_rate": float(np.mean([b for _, b in dec])) if dec else None,
    }
    return out


def stated_principles(run: ModelRun, min_mass: float) -> dict[str, dict]:
    out = {}
    for factor in FACTORS:
        per = []
        for i in range(len(ep.PRINCIPLE_USER)):
            rec = run.responses.final(f"principle|{factor}|{i}")
            if rec and rec["status"] == OK:
                ch = scoring.choice(rec.get("logprobs"), ep.PRINCIPLE_OPTIONS)
                if ch["mass"] >= min_mass:
                    per.append(ch["probs"])
        mean = {k: float(np.mean([p[k] for p in per])) for k in ep.PRINCIPLE_OPTIONS} if per else None
        rec = run.responses.final(f"principle_explain|{factor}")
        out[factor] = {
            "n_paraphrases": len(per),
            "probs": mean,
            "answer": max(mean, key=mean.get) if mean else None,
            "per_paraphrase": [max(p, key=p.get) for p in per],
            "explanation": rec.get("content") if rec and rec["status"] == OK else None,
        }
    return out


def say_do(stated: dict, effects: dict, alpha: float = 0.05) -> dict[str, dict]:
    """Compare what the model says about each factor with what it does."""
    out = {}
    for f in FACTORS:
        c = effects[f]["credibility"]
        p = c["wilcoxon_p_holm"]
        if p is None or c["mean_change"] is None or p >= alpha:
            does = "same"
        else:
            does = "less" if c["mean_change"] < 0 else "more"
        says = stated[f]["answer"]
        out[f] = {"says": says, "does": does, "consistent": None if says is None else says == does}
    return out


def coverage(run: ModelRun, rows: list[dict], scores: dict) -> dict:
    out = {}
    for task, key, mass in (("credibility", "credibility", "credibility_mass"), ("decision", "p_grant", "decision_mass")):
        masses = [s[mass] for s in scores.values() if s[mass] is not None]
        recs = [run.responses.final(f"{r['id']}|{task}") for r in rows]
        out[task] = {
            "prompts": len(rows),
            "answered": len(masses),
            "scored": sum(s[key] is not None for s in scores.values()),
            "content_filtered": sum(1 for rec in recs if rec and rec.get("reason") == "content_filter"),
            "mean_answer_mass": float(np.mean(masses)) if masses else None,
            "min_answer_mass": float(np.min(masses)) if masses else None,
        }
    return out


# --- figures -----------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_effects(summary: dict, path: Path, task: str, key: str, xlabel: str) -> None:
    """Mean paired change per factor with its 95% bootstrap CI, one series per model."""
    plt = _pyplot()
    models, factors = list(summary["models"]), list(FACTORS)
    y = np.arange(len(factors))
    step = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for j, m in enumerate(models):
        means, lo, hi = [], [], []
        for f in factors:
            e = summary["models"][m]["effects"][f][task]
            mean, ci = e[key], e["ci95"]
            means.append(np.nan if mean is None else mean)
            # max(0, ...): rounding can put the mean a hair outside a near-zero-width CI
            lo.append(max(0.0, mean - ci[0]) if mean is not None and ci else 0)
            hi.append(max(0.0, ci[1] - mean) if mean is not None and ci else 0)
        ax.errorbar(means, y + (j - (len(models) - 1) / 2) * step, xerr=[lo, hi], fmt="o", capsize=3, label=m)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels=[summary["factor_labels"][f] for f in factors])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    fig.legend(loc="outside lower center", ncols=len(models))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_flips(summary: dict, path: Path) -> None:
    """Share of passages whose grant/refuse decision flips, per factor and model."""
    plt = _pyplot()
    models, factors = list(summary["models"]), list(FACTORS)
    y = np.arange(len(factors))
    step = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for j, m in enumerate(models):
        rates = [(summary["models"][m]["effects"][f]["decision"]["flip_rate"] or 0) * 100 for f in factors]
        ax.barh(y + (j - (len(models) - 1) / 2) * step, rates, height=step, label=m)
    ax.set_yticks(y, labels=[summary["factor_labels"][f] for f in factors])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Passages whose decision flips (%)")
    fig.legend(loc="outside lower center", ncols=len(models))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.toml")
    ap.add_argument("--model", action="append", help="model name from [[evaluate.models]] (repeatable)")
    ap.add_argument("--mock", action="store_true", help="analyse results/mock/")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    out, bench = results_dir(cfg, args.mock), benchmark_dir(cfg, args.mock)
    an = out / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(bench / "benchmark.jsonl")
    bench_sha = file_sha(bench / "benchmark.jsonl")
    items = build_items(rows, principle_cues(cfg))
    min_mass = cfg["evaluate"]["min_answer_mass"]
    n_boot, seed = cfg["analyze"]["bootstrap_samples"], cfg["analyze"]["seed"]

    models = [m for m in cfg["evaluate"]["models"] if not args.model or m["name"] in args.model]
    summary = {"benchmark_sha256_12": bench_sha, "n_prompts": len(rows),
               "n_passages": len({r["passage_id"] for r in rows}), "min_answer_mass": min_mass,
               "bootstrap_samples": n_boot, "factor_labels": factor_labels(cfg), "models": {}}
    all_scores = {}
    for m in models:
        run = ModelRun(cfg, m, items, out, bench_sha, print)
        if not any(run.responses.final(it["key"]) for it in items):
            print(f"{m['name']}: no responses for this benchmark and model version; skipped")
            continue
        rng = np.random.default_rng(seed)
        scores = score_rows(run, rows, min_mass)
        effects = factor_effects(rows, scores, rng, n_boot)
        stated = stated_principles(run, min_mass)
        summary["models"][m["name"]] = {
            "version": run.version,
            "coverage": coverage(run, rows, scores),
            "effects": effects,
            "baseline_validity": baseline_validity(rows, scores),
            "stated": stated,
            "say_do": say_do(stated, effects),
        }
        all_scores[m["name"]] = scores
    if not summary["models"]:
        print("nothing to analyse; run botd.evaluate first")
        return 1

    names = list(all_scores)
    summary["agreement"] = {}
    base_ids = [r["id"] for r in rows if r["variant"] == "baseline"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res = {}
            for key in ("credibility", "p_grant"):
                xs = [(all_scores[a][k][key], all_scores[b][k][key]) for k in base_ids
                      if all_scores[a][k][key] is not None and all_scores[b][k][key] is not None]
                res[f"{key}_spearman"] = float(stats.spearmanr(*zip(*xs)).statistic) if len(xs) > 2 else None
                res["n"] = len(xs)
            summary["agreement"][f"{a} vs {b}"] = res

    (an / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(an / "effects.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "factor", "n", "credibility_mean_change", "credibility_ci_low", "credibility_ci_high",
                    "credibility_share_lower", "credibility_p_holm", "p_grant_mean_change", "p_grant_ci_low",
                    "p_grant_ci_high", "p_grant_p_holm", "flip_rate", "grant_to_refuse", "refuse_to_grant",
                    "stated_p_less", "stated_answer"])
        for m, s in summary["models"].items():
            for fac in FACTORS:
                c, d, st = s["effects"][fac]["credibility"], s["effects"][fac]["decision"], s["stated"][fac]
                w.writerow([m, fac, c["n"], c["mean_change"], *(c["ci95"] or [None, None]), c["share_lower"],
                            c["wilcoxon_p_holm"], d["mean_change_p_grant"], *(d["ci95"] or [None, None]),
                            d["wilcoxon_p_holm"], d["flip_rate"], d["grant_to_refuse"], d["refuse_to_grant"],
                            (st["probs"] or {}).get("less"), st["answer"]])
    with open(an / "scores.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "id", "passage_id", "variant", "outcome", "credibility", "credibility_mass",
                    "p_grant", "decision_mass"])
        for m, scores in all_scores.items():
            for r in rows:
                s = scores[r["id"]]
                w.writerow([m, r["id"], r["passage_id"], r["variant"], r["outcome"], s["credibility"],
                            s["credibility_mass"], s["p_grant"], s["decision_mass"]])

    plot_effects(summary, an / "credibility_effects.png", "credibility", "mean_change",
                 "Change in credibility rating (1–7), variant minus contrast")
    plot_effects(summary, an / "grant_effects.png", "decision", "mean_change_p_grant",
                 "Change in P(grant), variant minus contrast")
    plot_flips(summary, an / "decision_flips.png")
    print(f"wrote analysis for {', '.join(summary['models'])} to {an}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
