# Benefit of the Doubt

Matched-variant benchmark for bias in LLM credibility judgments on asylum testimony. See [RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md).

The [Makefile](Makefile) wraps the commands below: `make generate`, `make evaluate`, `make analyze`, `make status`, `make smoke`, `make mock`; `make` alone lists them.

## Generating the benchmark

Secrets go in `.env` (`AZURE_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `HUGGINGFACE`); research settings in [config.toml](config.toml). Generation settings are under `[generate.*]`, evaluation settings under `[evaluate]`. Set `generate.model.deployment` to your Azure AI Foundry chat deployment (or export `AZURE_DEPLOYMENT`).

```sh
uv run python -m botd.generate              # run, or resume after an interruption / rate limit
uv run python -m botd.generate --status     # progress per stage
uv run python -m botd.generate --stage assemble   # write outputs from whatever is finished
uv run python -m botd.generate --mock       # offline dry run with a fake LLM
```

Pipeline: **prepare** (download AsyLex at a pinned revision, label each first-instance RPD/CRDD decision from its own determination wording, draw a seeded, label-balanced pool) → **extract** (screen and rewrite as a 200–400 word first-person testimony with name placeholders; reject if the LLM reads a different outcome than the label, or if the testimony names the claimant's country) → **annotate** (LLM proposes exact-substring religious-vocabulary edits, checked by rule: they may not add actions) → **leakcheck** (second LLM pass for leaked board findings) → **assemble**.

Every LLM stage appends one fsynced JSONL record per case to `data/work/`, so reruns skip finished work. Rate limits are retried with `Retry-After` / backoff; a long quota lockout or repeated failures stop the run cleanly for a later resume.

### Reproducibility

- **Versioned stages.** Each record carries its stage's version: a hash of the stage's prompts, validator code, relevant `config.toml` settings, model settings and the upstream stage's version. Records from other versions are ignored (but kept), so changing a prompt, validator or setting reruns exactly the affected stages, and the dataset never mixes versions. Operational settings (pacing, retries) don't affect versions.
- **Deterministic selection.** The candidate pool depends only on the pinned AsyLex revision, `[generate.asylex]`/`[generate.sample]` settings and the seed; it is rebuilt automatically when those change. Passages are accepted in pool order, so `--stage assemble` over the same work logs always produces identical outputs.
- **LLM outputs.** Requests use `temperature = 0` and a fixed `seed`, and each record stores the model version and `system_fingerprint`, but Azure OpenAI does not guarantee identical outputs. The work logs in `data/work/` (`extract/annotate/leakcheck.jsonl`) are therefore the reproducible record of generation: publish them alongside the dataset. `passages.jsonl` alone (templates + edits + names) is enough to rebuild every prompt.
- **Provenance.** `manifest.json` records the git commit (and whether code/config had uncommitted changes), the AsyLex revision, stage versions, model versions and fingerprints, and the full prompts. Commit the code before a final run so the recorded commit is clean.

All six versions of a passage (baseline, Somali name, Islamic vocabulary, Somali interpreter, Spanish interpreter, two spellings of the name) are built by rule-based substitution from one template, and a diff check undoes each variant's single change to prove it equals its contrast version. Outputs in `data/benchmark/`:

- `benchmark.jsonl`: one row per prompt (`variant`, `factor`, `contrast_with`, `text`, outcome)
- `passages.jsonl`: per-decision template, edits, names, and metadata (citation, country of origin, persecution ground)
- `manual_review.csv`: a seeded 20% sample for hand checking
- `manifest.json`: provenance (commit, AsyLex revision, stage versions, model versions), full prompts, rejection counts, config

## Running the evaluation

The evaluated models are set in `[[evaluate.models]]` in [config.toml](config.toml): **DeepSeek-V4-Pro** (primary) and **gpt-6-luna** (comparison), both on the same Azure endpoint as the generator.

```sh
uv run python -m botd.evaluate                     # run / resume every model, then analyse
uv run python -m botd.evaluate --model gpt-6-luna  # one model
uv run python -m botd.evaluate --limit 2           # first 2 passages only (smoke test)
uv run python -m botd.evaluate --status            # progress per model and task
uv run python -m botd.evaluate --mock              # offline dry run with a fake model
uv run python -m botd.analyze                      # re-run the analysis only
```

**Tasks.** Every request is a separate single-turn conversation with a one-token answer, scored from the first token's logprobs rather than sampled text:

- **credibility**: each of the 600 benchmark prompts rated 1–7; the score is the expected value over the digit tokens.
- **decision**: GRANT or REFUSE for each prompt; the score is P(grant), renormalised over the two answers.
- **principle**: asked directly, with no testimony, whether each factor should make testimony LESS or MORE credible or leave it the SAME (three paraphrases per factor), plus a free-text explanation.

**Model settings.** DeepSeek-V4-Pro runs at temperature 0 with the top 20 logprobs (all seven digits are always present). gpt-6-luna is a reasoning model: Azure only returns logprobs for it with `reasoning_effort = "none"`, fixes its temperature at 1 and caps `top_logprobs` at 5. So it is evaluated with reasoning off, and digits outside its top 5 count as zero. The share of probability on valid answer tokens is recorded for every response; answers below `min_answer_mass` (0.5) are excluded.

**Parallel runs.** Azure rate limits are per deployment (currently 500 requests and 500k tokens per minute each), so the models run at the same time, each with its own pool of `concurrency` in-flight requests paced by `min_interval_s`. A full run (about 1,220 requests per model) takes a few minutes. Retries, backoff and the clean stop on quota exhaustion are the same as the generator's. The Azure Batch API was not used: it doesn't cover DeepSeek and returns results asynchronously, within up to 24 hours.

**Reproducibility.** Every response, with its raw top logprobs, model version and token usage, is appended and fsynced to `data/results/<model>/responses.jsonl`, so reruns skip finished requests. Responses are versioned by model settings, prompts and the benchmark file's hash: changing any of them reruns only what it affects. Scores are recomputed from the raw logprobs at analysis time, so a change to scoring needs no new requests. `data/results/manifest.json` records the code commit, the benchmark hash and generator versions, model versions and the full prompts.

**Analysis** (`data/results/analysis/`). Each variant is paired with its contrast version of the same passage. For *name spelling*, the contrast is the consistently spelled Somali name, so name origin is held constant. The two interpreter variants share the baseline as contrast, so comparing them separates the effect of mentioning an interpreter from that of the language named. The analysis reports:

- per factor, the mean paired change in credibility and in P(grant), with 95% bootstrap CIs over passages and Wilcoxon signed-rank p-values, Holm-corrected across the five factors;
- decision flip rates, with an exact binomial test on the direction of the flips;
- what each model *says* (stated principles) against what it *does* (the direction of a significant effect);
- how baseline scores relate to the tribunal's real outcome (AUC, agreement);
- agreement between the two models.

Outputs: `summary.json` (every statistic, plus the models' free-text explanations), `effects.csv` (one row per model × factor), `scores.csv` (one row per model × prompt), and figures `credibility_effects.png`, `grant_effects.png` and `decision_flips.png`.

AsyLex (Barale et al., 2023) is CC BY-NC-SA 4.0; the derived dataset uses the same licence.
