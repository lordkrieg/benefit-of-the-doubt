# Benefit of the Doubt

Matched-variant benchmark for bias in LLM credibility judgments on asylum testimony. See [RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md).

## Generating the benchmark

Secrets go in `.env` (`AZURE_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `HUGGINGFACE`); research settings in [config.toml](config.toml). Set `azure.deployment` to your Azure AI Foundry chat deployment (or export `AZURE_DEPLOYMENT`).

```sh
uv run python -m botd.generate              # run, or resume after an interruption / rate limit
uv run python -m botd.generate --status     # progress per stage
uv run python -m botd.generate --stage assemble   # write outputs from whatever is finished
uv run python -m botd.generate --mock       # offline dry run with a fake LLM
```

Pipeline: **prepare** (download AsyLex at a pinned revision, label each first-instance RPD/CRDD decision from its own determination wording, draw a seeded, label-balanced pool) → **extract** (screen and rewrite as a 200–400 word first-person testimony with name placeholders; reject if the LLM reads a different outcome than the label, or if the testimony names the claimant's country) → **annotate** (LLM proposes exact-substring hedging and religious-vocabulary edits, checked by rule: hedges must sit right before a date/number/duration, religious edits may not add actions) → **leakcheck** (second LLM pass for leaked board findings) → **assemble**.

Every LLM stage appends one fsynced JSONL record per case to `data/work/`, so reruns skip finished work. Rate limits are retried with `Retry-After` / backoff; a long quota lockout or repeated failures stop the run cleanly for a later resume.

### Reproducibility

- **Versioned stages.** Each record carries its stage's version: a hash of the stage's prompts, validator code, relevant `config.toml` settings, model settings and the upstream stage's version. Records from other versions are ignored (but kept), so changing a prompt, validator or setting reruns exactly the affected stages, and the dataset never mixes versions. Operational settings (pacing, retries) don't affect versions.
- **Deterministic selection.** The candidate pool depends only on the pinned AsyLex revision, `[dataset]`/`[sample]` settings and the seed; it is rebuilt automatically when those change. Passages are accepted in pool order, so `--stage assemble` over the same work logs always produces identical outputs.
- **LLM outputs.** Requests use `temperature = 0` and a fixed `seed`, and each record stores the model version and `system_fingerprint`, but Azure OpenAI does not guarantee identical outputs. The work logs in `data/work/` (`extract/annotate/leakcheck.jsonl`) are therefore the reproducible record of generation: publish them alongside the dataset. `passages.jsonl` alone (templates + edits + names) is enough to rebuild every prompt.
- **Provenance.** `manifest.json` records the git commit (and whether code/config had uncommitted changes), the AsyLex revision, stage versions, model versions and fingerprints, and the full prompts. Commit the code before a final run so the recorded commit is clean.

All six versions of a passage are built by rule-based substitution from one template, and a diff check undoes each variant's single change to prove it equals its contrast version. Outputs in `data/benchmark/`:

- `benchmark.jsonl`: one row per prompt (`variant`, `factor`, `contrast_with`, `text`, outcome)
- `passages.jsonl`: per-decision template, edits, names, and metadata (citation, country of origin, persecution ground)
- `manual_review.csv`: a seeded 20% sample for hand checking
- `manifest.json`: provenance (commit, AsyLex revision, stage versions, model versions), full prompts, rejection counts, config

AsyLex (Barale et al., 2023) is CC BY-NC-SA 4.0; the derived dataset uses the same licence.
