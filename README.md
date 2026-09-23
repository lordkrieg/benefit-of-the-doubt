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

Pipeline: **prepare** (download AsyLex, draw a seeded, label-balanced pool of first-instance RPD/CRDD decisions) → **extract** (screen and rewrite as a 200–400 word first-person testimony with name placeholders; reject if the decision's outcome disagrees with the AsyLex label) → **annotate** (LLM proposes exact-substring hedging and religious-vocabulary edits) → **leakcheck** (second LLM pass for leaked board findings) → **assemble**.

Every LLM stage appends one fsynced JSONL record per case to `data/work/`, so reruns skip finished work. Rate limits are retried with `Retry-After` / backoff; a long quota lockout or repeated failures stop the run cleanly for a later resume.

All six versions of a passage are built by rule-based substitution from one template, and a diff check undoes each variant's single change to prove it equals its contrast version. Outputs in `data/benchmark/`:

- `benchmark.jsonl`: one row per prompt (`variant`, `factor`, `contrast_with`, `text`, outcome)
- `passages.jsonl`: per-decision template, edits, names, and metadata (citation, country of origin, persecution ground)
- `manual_review.csv`: a seeded 20% sample for hand checking
- `manifest.json`: generator model version, full prompts and hashes, rejection counts, config

AsyLex (Barale et al., 2023) is CC BY-NC-SA 4.0; the derived dataset uses the same licence.
