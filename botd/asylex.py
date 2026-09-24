"""Download AsyLex, label decisions, and draw a balanced candidate pool."""

import csv
import random
import re
import tarfile
import urllib.request
from pathlib import Path

FILES = {
    "texts": "cases_anonymized_txt_raw.tar.gz",
    "gold": "outcome_train_test/test_dataset_gold.csv",
    "silver": "outcome_train_test/train_dataset_silver.csv",
    "case_cover": "case_cover/case_cover_entities_and_decision_outcome.csv",
}

# Checked in order: appeal divisions first, because their headers also mention the RPD.
DIVISIONS = [
    ("rad", ("refugee appeal division", "rad file")),
    ("iad", ("immigration appeal division", "appeal division")),
    ("rpd", ("refugee protection division", "rpd file", "dossier de la spr")),
    ("crdd", ("convention refugee determination division",)),
    ("id", ("immigration division",)),
]


def download(repo: str, revision: str, raw_dir: Path, token: str | None, log=print) -> dict[str, Path]:
    """Fetch the AsyLex files we need at a pinned dataset revision, resuming partial downloads."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, remote in FILES.items():
        dest = raw_dir / Path(remote).name
        paths[key] = dest
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
        have = dest.stat().st_size if dest.exists() else 0
        if total and have == total:
            continue
        log(f"downloading {remote} ({have / 1e6:.0f}/{total / 1e6:.0f} MB present)")
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    return paths


def load_labels(paths: dict[str, Path], sources: list[str]) -> dict[str, tuple[int, str]]:
    """decisionID -> (outcome, source). Earlier sources in `sources` win. Uncertain (2) is dropped."""
    csv.field_size_limit(1 << 30)
    labels: dict[str, tuple[int, str]] = {}
    for source in reversed(sources):
        with open(paths[source], encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                outcome = row["decision_outcome"].strip()
                if outcome in ("0", "1"):
                    labels[row["decisionID"].strip()] = (int(outcome), source)
                else:
                    labels.pop(row["decisionID"].strip(), None)
    return labels


def load_cover_years(path: Path) -> dict[str, str]:
    """decisionID -> year of the citation printed on its case cover ("2013 canlii 100000")."""
    csv.field_size_limit(1 << 30)
    years = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            m = re.search(r"(\d{4}) canlii (\d+)", row["text_case_cover"])
            if m and m.group(2) == row["decisionID"].strip():
                years[m.group(2)] = m.group(1)
    return years


def division_of(text: str) -> str:
    head = re.sub(r"\s+", " ", text[:6000].lower())
    for name, needles in DIVISIONS:
        if any(n in head for n in needles):
            return name
    return "unknown"


# Determination language, matched in the decision's final part. The last match wins, since
# the determination closes the decision (so "not a Convention refugee but is a person in need
# of protection" is a grant). The extract stage re-checks every label with the LLM.
_POSITIVE = re.compile(
    r"\b(is|are) (a )?convention refugees?\b|\b(is|are) (a )?persons? in need of protection"
    r"|(?<!not )(?<!cannot )(?<!can not )\b(accepts?|allows?) (the|his|her|their|your) claims?\b"
    r"|\bclaims? (is|are) (hereby )?(accepted|allowed)",
    re.I,
)
_NEGATIVE = re.compile(
    r"\b(is|are) (neither|not) (a )?(convention refugees?|persons? in need)"
    r"|\b(rejects?|dismiss(es)?) (the|his|her|their|your) claims?\b|\bclaims? (is|are) (hereby )?(rejected|dismissed)"
    r"|\b(not|cannot|can not) accept (the|his|her|their|your) claims?\b",
    re.I,
)


def outcome_from_text(text: str) -> int | None:
    """1 = granted, 0 = refused, None = no determination found."""
    tail = re.sub(r"\s+", " ", text)[-6000:]
    last = {}
    for outcome, pattern in ((1, _POSITIVE), (0, _NEGATIVE)):
        for m in pattern.finditer(tail):
            last[outcome] = m.start()
    return max(last, key=last.get) if last else None


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def build_candidates(paths: dict[str, Path], cfg: dict, log=print) -> list[dict]:
    """Stream the decision archive once and return a seeded, label-balanced candidate pool.

    Order matters: the pipeline accepts candidates in this order until each label's
    target is met, so the final sample is reproducible from the seed.
    """
    ds, sample = cfg["dataset"], cfg["sample"]
    from_text = ds["label_mode"] == "decision_text"
    labels = load_labels(paths, ds["label_sources"])
    cover_years = load_cover_years(paths["case_cover"])
    eligible: dict[int, list[dict]] = {0: [], 1: []}
    seen: dict[str, int] = {}
    with tarfile.open(paths["texts"], "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            m = re.search(r"(\d{4})canlii(\d+)", member.name)
            if m:
                seen[m.group(2)] = seen.get(m.group(2), 0) + 1
            if not m or (not from_text and m.group(2) not in labels):
                continue
            text = decode(tar.extractfile(member).read())
            if len(text) < ds["min_chars"] or division_of(text) not in ds["divisions"]:
                continue
            asylex_label = labels.get(m.group(2))
            if from_text:
                outcome, source = outcome_from_text(text), "decision_text"
                if outcome is None:
                    continue
            else:
                outcome, source = asylex_label
            eligible[outcome].append(
                {
                    "case_id": m.group(2),
                    "citation": f"{m.group(1)} CanLII {m.group(2)} (CA IRB)",
                    "year": int(m.group(1)),
                    "division": division_of(text),
                    "outcome": outcome,
                    "label_source": source,
                    "asylex_label": asylex_label[0] if asylex_label else None,
                    "n_chars": len(text),
                    "text": text,
                }
            )
    if from_text:
        # Labels come from each document itself; only drop numbers repeated among the
        # eligible documents, since the work logs are keyed by case number.
        counts: dict[str, int] = {}
        for c in eligible[0] + eligible[1]:
            counts[c["case_id"]] = counts.get(c["case_id"], 0) + 1
        keep = lambda c: counts[c["case_id"]] == 1  # noqa: E731
    else:
        # CanLII numbers repeat across years, but AsyLex labels carry only the number.
        # Keep a repeated number only when its case-cover citation names this document's
        # year; often it names a different (e.g. IAD) decision, whose label is not ours.
        keep = lambda c: seen[c["case_id"]] == 1 or cover_years.get(c["case_id"]) == str(c["year"])  # noqa: E731
    for outcome in eligible:
        eligible[outcome] = [c for c in eligible[outcome] if keep(c)]
    log(f"eligible decisions: {len(eligible[1])} granted, {len(eligible[0])} refused")

    rng = random.Random(ds["seed"])
    pools = {}
    for outcome, target in ((1, sample["n_granted"]), (0, sample["n_refused"])):
        cases = sorted(eligible[outcome], key=lambda c: int(c["case_id"]))
        rng.shuffle(cases)
        pools[outcome] = cases[: target * sample["candidate_pool_multiplier"]]
        if len(cases) < target:
            log(f"WARNING: only {len(cases)} eligible for outcome={outcome}, target {target}")
    # Interleave labels so a partial run stays roughly balanced.
    out = []
    for i in range(max(len(p) for p in pools.values())):
        for outcome in (1, 0):
            if i < len(pools[outcome]):
                out.append({**pools[outcome][i], "rank": i})
    return out
