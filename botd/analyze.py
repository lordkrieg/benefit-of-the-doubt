"""Analyse the evaluation responses: paired effects per factor, decision flips, stated principles.

    uv run python -m botd.analyze            # all models with responses
    uv run python -m botd.analyze --mock

Scores are recomputed from the raw first-token logprobs in data/results/<model>/responses.jsonl
(see botd.scoring). Outputs go to data/results/analysis/:
- summary.json: every statistic below, per model;
- effects.csv: one row per model x factor;
- scores.csv: one row per model x benchmark prompt (credibility score, P(grant), answer mass);
- report.md: tables and the models' own explanations;
- credibility_effects.png, grant_effects.png, decision_flips.png.

Per factor, each variant is compared with its contrast version of the same passage (baseline,
or the consistently spelled Somali name for name_spelling):
- credibility: mean paired change in expected rating (1-7), bootstrap 95% CI over passages,
  Wilcoxon signed-rank test, Holm-corrected across the five factors;
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
from .evaluate import ModelRun, benchmark_dir, build_items, file_sha, results_dir
from .store import OK, read_jsonl

FACTORS = {
    "name": "Somali name",
    "religious_vocabulary": "Islamic vocabulary",
    "interpretation": "Interpreter mentioned",
    "hedging": "Hedged dates and numbers",
    "name_spelling": "Two spellings of the name",
}
# Categorical slots 1-2 of the reference palette (light mode), in model order.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def bootstrap_ci(x: np.ndarray, rng, n: int, stat=np.mean) -> list[float] | None:
    if len(x) < 2:
        return None
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = stat(x[idx], axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


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


def _style(ax, plt):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_effects(summary: dict, path: Path, key: str, title: str, xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summary["models"])
    factors = list(FACTORS)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(factors) * max(1, len(models)) + 1.4), facecolor=SURFACE)
    _style(ax, plt)
    ax.axvline(0, color=INK_2, linewidth=1)
    step = 0.32
    for j, m in enumerate(models):
        eff = summary["models"][m]["effects"]
        for i, f in enumerate(factors):
            y = len(factors) - 1 - i + (len(models) - 1) / 2 * step - j * step
            e = eff[f][key.split(".")[0]]
            mean, ci = e[key.split(".")[1]], e["ci95"]
            if mean is None:
                continue
            if ci:
                ax.plot(ci, [y, y], color=MODEL_COLORS[j], linewidth=2, solid_capstyle="round")
            ax.plot([mean], [y], "o", markersize=8, color=MODEL_COLORS[j], markeredgecolor=SURFACE,
                    markeredgewidth=2, label=m if i == 0 else None, zorder=3)
            p = e["wilcoxon_p_holm"]
            if p is not None and p < 0.05:
                ax.annotate("*", (ci[1] if ci else mean, y), xytext=(4, -3), textcoords="offset points",
                            color=INK, fontsize=11)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([FACTORS[f] for f in reversed(factors)], color=INK, fontsize=10)
    lim = max(0.05, *(abs(v) for m in models for f in factors
                      for v in (summary["models"][m]["effects"][f][key.split(".")[0]]["ci95"] or [0])))
    ax.set_xlim(-lim * 1.25, lim * 1.25)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=22)
    ax.text(0, 1.02, "Mean paired change vs. contrast version, 95% bootstrap CI; * Holm-corrected Wilcoxon p < .05",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(models), frameon=False, fontsize=9,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_flips(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summary["models"])
    factors = list(FACTORS)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(factors) * max(1, len(models)) + 1.4), facecolor=SURFACE)
    _style(ax, plt)
    h = 0.8 / max(1, len(models))
    for j, m in enumerate(models):
        eff = summary["models"][m]["effects"]
        for i, f in enumerate(factors):
            d = eff[f]["decision"]
            if d["flip_rate"] is None:
                continue
            y = len(factors) - 1 - i + 0.4 - h * (j + 0.5)
            ax.barh(y, d["flip_rate"] * 100, height=h - 0.06, color=MODEL_COLORS[j], label=m if i == 0 else None)
            ax.annotate(f"{d['flip_rate'] * 100:.0f}%  ({d['grant_to_refuse']} to refuse, {d['refuse_to_grant']} to grant)",
                        (d["flip_rate"] * 100, y), xytext=(4, 0), textcoords="offset points", va="center",
                        color=INK_2, fontsize=8)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([FACTORS[f] for f in reversed(factors)], color=INK, fontsize=10)
    top = max([eff["decision"]["flip_rate"] or 0 for m in models for eff in summary["models"][m]["effects"].values()] + [0.05])
    ax.set_xlim(0, top * 100 * 1.8)
    ax.set_xlabel("Passages whose grant/refuse decision flips (%)", color=INK_2, fontsize=9)
    ax.set_title("Decision flips by factor", loc="left", color=INK, fontsize=12, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(models), frameon=False, fontsize=9,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- report ------------------------------------------------------------------------------


def _f(x, fmt="{:+.2f}"):
    return "–" if x is None else fmt.format(x)


def _ci(ci, fmt="{:+.2f}"):
    return "–" if not ci else f"[{fmt.format(ci[0])}, {fmt.format(ci[1])}]"


def _p(p):
    return "–" if p is None else ("<.001" if p < 0.001 else f"{p:.3f}")


def write_report(summary: dict, path: Path) -> None:
    L = ["# Evaluation results", ""]
    L += [f"Benchmark sha `{summary['benchmark_sha256_12']}`, {summary['n_prompts']} prompts from "
          f"{summary['n_passages']} passages. Scores are recomputed from first-token logprobs. "
          "Changes are variant minus contrast version of the same passage (the contrast for *two spellings* "
          "is the consistently spelled Somali name). CIs are 95% bootstrap over passages; p-values are "
          "Wilcoxon signed-rank, Holm-corrected across the five factors.", ""]
    for m, s in summary["models"].items():
        L += [f"## {m}", ""]
        cov = s["coverage"]
        L += [f"Scored {cov['credibility']['scored']}/{cov['credibility']['prompts']} credibility and "
              f"{cov['decision']['scored']}/{cov['decision']['prompts']} decision prompts "
              f"(mean probability on valid answer tokens {_f(cov['credibility']['mean_answer_mass'], '{:.3f}')} / "
              f"{_f(cov['decision']['mean_answer_mass'], '{:.3f}')}; content-filtered "
              f"{cov['credibility']['content_filtered']} / {cov['decision']['content_filtered']}).", ""]
        L += ["### Credibility (1–7)", "",
              "| Factor | n | Mean change | 95% CI | Rated lower | Cohen's dz | p (Holm) |",
              "|---|---:|---:|---|---:|---:|---:|"]
        for f, label in FACTORS.items():
            c = s["effects"][f]["credibility"]
            L.append(f"| {label} | {c['n']} | {_f(c['mean_change'])} | {_ci(c['ci95'])} | "
                     f"{_f(c['share_lower'], '{:.0%}')} | {_f(c['cohens_dz'])} | {_p(c['wilcoxon_p_holm'])} |")
        L += ["", "### Grant / refuse", "",
              "| Factor | n | Change in P(grant) | 95% CI | p (Holm) | Flip rate | To refuse | To grant | Direction p |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for f, label in FACTORS.items():
            d = s["effects"][f]["decision"]
            L.append(f"| {label} | {d['n']} | {_f(d['mean_change_p_grant'], '{:+.3f}')} | "
                     f"{_ci(d['ci95'], '{:+.3f}')} | {_p(d['wilcoxon_p_holm'])} | {_f(d['flip_rate'], '{:.0%}')} | "
                     f"{d['grant_to_refuse']} | {d['refuse_to_grant']} | {_p(d['flip_direction_p'])} |")
        L += ["", "### What it says vs. what it does", "",
              "Stated: the model is asked directly (three paraphrases, no testimony) whether the factor should "
              "make testimony LESS or MORE credible or leave it the SAME. Does: the direction of a significant "
              "credibility effect above, else SAME.", "",
              "| Factor | P(less) | P(same) | P(more) | Says | Does | Consistent |", "|---|---:|---:|---:|---|---|---|"]
        for f, label in FACTORS.items():
            st, sd = s["stated"][f], s["say_do"][f]
            pr = st["probs"] or {}
            L.append(f"| {label} | {_f(pr.get('less'), '{:.2f}')} | {_f(pr.get('same'), '{:.2f}')} | "
                     f"{_f(pr.get('more'), '{:.2f}')} | {sd['says'] or '–'} | {sd['does']} | "
                     f"{'–' if sd['consistent'] is None else ('yes' if sd['consistent'] else '**no**')} |")
        v = s["baseline_validity"]
        L += ["", "### Baseline vs. the tribunal's real outcome", "",
              f"- Credibility: mean {_f(v['credibility']['mean_granted'], '{:.2f}')} for granted vs "
              f"{_f(v['credibility']['mean_refused'], '{:.2f}')} for refused claims; AUC "
              f"{_f(v['credibility']['auc_vs_outcome'], '{:.2f}')}.",
              f"- P(grant): mean {_f(v['p_grant']['mean_granted'], '{:.2f}')} vs "
              f"{_f(v['p_grant']['mean_refused'], '{:.2f}')}; AUC {_f(v['p_grant']['auc_vs_outcome'], '{:.2f}')}.",
              f"- Decision: grants {_f(v['decision']['grant_rate'], '{:.0%}')} of baseline passages (tribunal: "
              f"{_f(v['decision']['real_grant_rate'], '{:.0%}')}); agrees with the tribunal on "
              f"{_f(v['decision']['accuracy_vs_outcome'], '{:.0%}')}.", ""]
        L += ["### In its own words", ""]
        for f, label in FACTORS.items():
            text = (s["stated"][f]["explanation"] or "–").strip().replace("\n", " ")
            L.append(f"- **{label}:** {text}")
        L.append("")
    if summary.get("agreement"):
        L += ["## Agreement between models", ""]
        for pair, a in summary["agreement"].items():
            L.append(f"- {pair}: Spearman ρ of baseline credibility {_f(a['credibility_spearman'], '{:.2f}')}, "
                     f"of baseline P(grant) {_f(a['p_grant_spearman'], '{:.2f}')} (n = {a['n']}).")
        L.append("")
    L += ["![Credibility effects](credibility_effects.png)", "", "![Grant probability effects](grant_effects.png)",
          "", "![Decision flips](decision_flips.png)", ""]
    path.write_text("\n".join(L), encoding="utf-8")


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
    items = build_items(rows)
    min_mass = cfg["evaluate"]["min_answer_mass"]
    n_boot, seed = cfg["analysis"]["bootstrap_samples"], cfg["analysis"]["seed"]

    models = [m for m in cfg["evaluate"]["models"] if not args.model or m["name"] in args.model]
    summary = {"benchmark_sha256_12": bench_sha, "n_prompts": len(rows),
               "n_passages": len({r["passage_id"] for r in rows}), "min_answer_mass": min_mass,
               "bootstrap_samples": n_boot, "models": {}}
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

    plot_effects(summary, an / "credibility_effects.png", "credibility.mean_change",
                 "Change in credibility rating by factor", "Change in expected credibility rating (1–7 scale)")
    plot_effects(summary, an / "grant_effects.png", "decision.mean_change_p_grant",
                 "Change in grant probability by factor", "Change in P(grant)")
    plot_flips(summary, an / "decision_flips.png")
    write_report(summary, an / "report.md")
    print(f"wrote analysis for {', '.join(summary['models'])} to {an}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
