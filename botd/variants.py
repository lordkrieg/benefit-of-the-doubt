"""Validation of LLM output, rule-based construction of the six versions, and diff checks.

Every variant is built from the same template by deterministic substitution, and
the diff check proves it by undoing the one intended change and comparing with the
version it is contrasted against.
"""

import difflib
import re

PLACEHOLDERS = ("{FULL_NAME}", "{GIVEN_NAME}")

# (variant, factor, contrast_with)
VERSIONS = [
    ("baseline", None, None),
    ("name", "name", "baseline"),
    ("religious", "religious_vocabulary", "baseline"),
    ("interpretation", "interpretation", "baseline"),
    ("hedging", "hedging", "baseline"),
    # Spelling is contrasted with the consistently spelled Somali name, so name
    # origin is held constant and only the inconsistency differs.
    ("name_spelling", "name_spelling", "name"),
]

# Strong hedges only: "about"/"around" are too common in non-hedged use to ban from the baseline.
BASELINE_HEDGES = re.compile(
    r"\b(i think|maybe|perhaps|approximately|i believe|not sure|if i remember|roughly|i am not certain)\b", re.I
)
# Religious vocabulary the baseline must not contain. Fixed here (not taken from the edit
# lexicon in config) so that tuning the religious variant doesn't change extraction.
BASELINE_RELIGIOUS = [
    "hijra", "allah", "inshallah", "insha'allah", "alhamdulillah", "mashallah",
    "bismillah", "dua", "sabr", "masjid", "salah", "ummah", "subhanallah",
]
LEAK_TERMS = re.compile(
    r"\b(panel|tribunal|the board|credib\w*|convention refugee|person in need of protection|RPD|IRB|"
    r"counsel|interpreter|translat\w*)\b",
    re.I,
)


def _has_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(re.search(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", low) for t in terms)


def validate_extract(obj: dict, cfg: dict) -> list[str]:
    p = []
    if not isinstance(obj.get("usable"), bool):
        return ['"usable" must be true or false']
    if not obj["usable"]:
        return [] if obj.get("reject_reason") else ['give a "reject_reason"']
    if obj.get("outcome_in_decision") not in ("granted", "refused", "unclear"):
        p.append('"outcome_in_decision" must be granted, refused or unclear')
    if obj.get("claimant_gender") not in ("male", "female", "unknown"):
        p.append('"claimant_gender" must be male, female or unknown')
    t = obj.get("testimony") or ""
    n = len(t.split())
    lo, hi = cfg["passage"]["min_words"], cfg["passage"]["max_words"]
    if not (lo * 0.9 <= n <= hi * 1.1):
        p.append(f"testimony has {n} words; it must have {lo}-{hi}")
    if not t.startswith("My name is {FULL_NAME}."):
        p.append('testimony must start exactly with "My name is {FULL_NAME}."')
    stray = set(re.findall(r"\{[^}]*\}", t)) - set(PLACEHOLDERS)
    if stray:
        p.append(f"unknown placeholders {sorted(stray)}; only {{FULL_NAME}} and {{GIVEN_NAME}} are allowed")
    if re.search(r"x{3,}", t, re.I):
        p.append('anonymization marks ("XXXX") remain; describe the person or place generically')
    if m := BASELINE_HEDGES.search(t):
        p.append(f'baseline must not hedge; remove "{m.group(0)}"')
    if _has_any(t, BASELINE_RELIGIOUS):
        p.append("baseline must not use religious vocabulary")
    if m := LEAK_TERMS.search(t):
        p.append(f'remove tribunal/process wording: "{m.group(0)}"')
    return p


def names_origin(template: str, country: str) -> str | None:
    """The mention of the claimant's own country (or its demonym) in the testimony, if any."""
    country = (country or "").split("(")[0].split(",")[0].strip()
    if not country or country.lower() == "unknown":
        return None
    stems = sorted({country, re.sub(r"[aeoy]$", "", country)}, key=len, reverse=True)
    pattern = rf"\b({'|'.join(re.escape(s) for s in stems)})(a|an|ian|ese|i|n)?\b"
    m = re.search(pattern, template, re.I)
    return m.group(0) if m else None


def _locate(template: str, edits: list[dict]) -> tuple[list[tuple[int, int, str]], list[str]]:
    spans, problems = [], []
    for e in edits:
        orig, rep = e.get("original", ""), e.get("replacement", "")
        if not orig or not rep or orig == rep:
            problems.append(f"empty or no-op edit: {e}")
            continue
        if any(ph in orig or ph in rep for ph in PLACEHOLDERS) or "{" in rep:
            problems.append(f"edit touches a name placeholder: {orig!r}")
            continue
        count = template.count(orig)
        if count != 1:
            problems.append(f"original {orig!r} occurs {count} times; it must occur exactly once")
            continue
        start = template.index(orig)
        spans.append((start, start + len(orig), rep))
    spans.sort()
    for a, b in zip(spans, spans[1:]):
        if b[0] < a[1]:
            problems.append(f"edits overlap: {template[a[0]:a[1]]!r} and {template[b[0]:b[1]]!r}")
    return spans, problems


def apply_edits(template: str, edits: list[dict]) -> str:
    spans, problems = _locate(template, edits)
    if problems:
        raise ValueError("; ".join(problems))
    out, pos = [], 0
    for start, end, rep in spans:
        out += [template[pos:start], rep]
        pos = end
    return "".join(out + [template[pos:]])


# Peripheral details a hedge may qualify: numbers, dates, counts, durations, times of day.
DETAIL = re.compile(
    r"\d|\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty"
    r"|hundred|dozen|once|twice|several|few|first|second|third|last"
    r"|morning|afternoon|evening|night|midnight|noon|days?|weeks?|months?|years?|hours?|minutes?"
    r"|kilomet(er|re)s?|km|miles?|times)\b",
    re.I,
)
# A hedge directly followed by the speaker (or another actor) doubts the event itself.
HEDGED_ACTOR = re.compile(
    r"\b(i think|i believe|maybe|perhaps|i am not certain|not sure|if i remember( correctly| right)?)\b,? "
    r"(that )?(i|we|he|she|they)\b",
    re.I,
)
ADDED_ACTS = re.compile(r"\b(pray\w*|worship\w*|fast(ed|ing)|recit\w*)\b", re.I)


def _insertions(orig: str, rep: str) -> list[tuple[str, str]]:
    """(inserted words, the words that follow them in the replacement) for each insertion."""
    a, b = orig.split(), rep.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [
        (" ".join(b[j1:j2]), " ".join(b[j2:]))
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes()
        if tag in ("insert", "replace")
    ]


def _check_hedge(orig: str, rep: str, markers: list[str]) -> list[str]:
    p = []
    missing = set(re.findall(r"\d+", orig)) - set(re.findall(r"\d+", rep))
    if missing:
        p.append(f"{rep!r} drops the numbers {sorted(missing)}")
    if not DETAIL.search(orig):
        p.append(f"{orig!r} has no date, number, count or duration to hedge; pick a detail")
    marker_re = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(m) for m in sorted(markers, key=len, reverse=True)) + r")(?!\w)", re.I
    )
    for inserted, after in _insertions(orig, rep):
        window = f"{inserted} {' '.join(after.split()[:3])}"
        if HEDGED_ACTOR.search(window):
            p.append(f"{rep!r} hedges whether someone acted; hedge only the detail (date, number, duration)")
        for m in marker_re.finditer(inserted):
            following = " ".join((inserted[m.end():] + " " + after).split()[:6])
            if not DETAIL.search(following):
                p.append(f"hedge {m.group(0)!r} in {rep!r} must come right before the detail it qualifies")
    return p


def _check_religious(orig: str, rep: str, lexicon: list[str]) -> list[str]:
    if any(ADDED_ACTS.search(inserted) for inserted, _ in _insertions(orig, rep)):
        return [f"{rep!r} adds an action (e.g. praying); only reword what is already there"]
    return []


def validate_annotate(obj: dict, template: str, cfg: dict, pair: dict) -> list[str]:
    p = []
    for key, vcfg, terms in (
        ("hedging", cfg["variants"]["hedging"], cfg["variants"]["hedging"]["markers"]),
        ("religious", cfg["variants"]["religious"], cfg["variants"]["religious"]["lexicon"]),
    ):
        edits = obj.get(key)
        if not isinstance(edits, list):
            p.append(f'"{key}" must be a list')
            continue
        if not vcfg["min_edits"] <= len(edits) <= vcfg["max_edits"]:
            p.append(f'"{key}" needs {vcfg["min_edits"]}-{vcfg["max_edits"]} edits, got {len(edits)}')
        p += [f"{key}: {x}" for x in _locate(template, edits)[1]]
        for e in edits:
            orig, rep = e.get("original") or "", e.get("replacement") or ""
            if rep and not _has_any(rep, terms):
                p.append(f"{key}: replacement {rep!r} lacks a required term ({', '.join(terms[:5])}, ...)")
            if orig and rep:
                added = len(rep.split()) - len(orig.split())
                if added > vcfg["max_added_words"]:
                    p.append(f"{key}: {rep!r} adds {added} words; at most {vcfg['max_added_words']} allowed")
                check = _check_hedge if key == "hedging" else _check_religious
                p += [f"{key}: {x}" for x in check(orig, rep, terms)]
    if not p:
        versions = build_versions(template, obj, pair, cfg)
        p += [f"diff check {k}: {v}" for k, v in diff_check(versions, template, obj, pair, cfg).items() if v != "ok"]
    return p


def pick_names(case_id: str, gender: str, cfg: dict) -> dict:
    pool = cfg["names"]["female" if gender == "female" else "male"]
    return pool[int(case_id) % len(pool)]


def _fill(text: str, given: str, family: str) -> str:
    return text.replace("{FULL_NAME}", f"{given} {family}").replace("{GIVEN_NAME}", given)


def _compose(header_name: tuple[str, str], body: str, cfg: dict, interp: bool = False) -> str:
    parts = [_fill(cfg["passage"]["header"], *header_name)]
    if interp:
        parts.append(_interp_sentence(cfg))
    parts.append(body)
    return "\n\n".join(parts)


def _interp_sentence(cfg: dict) -> str:
    v = cfg["variants"]["interpretation"]
    return v["sentence"].format(language=v["language"])


def build_versions(template: str, edits: dict, pair: dict, cfg: dict) -> dict[str, str]:
    b, v = pair["baseline"], pair["variant"]
    base = (b["given"], b["family"])
    som = (v["given"], v["family"])
    return {
        "baseline": _compose(base, _fill(template, *base), cfg),
        "name": _compose(som, _fill(template, *som), cfg),
        "religious": _compose(base, _fill(apply_edits(template, edits["religious"]), *base), cfg),
        "interpretation": _compose(base, _fill(template, *base), cfg, interp=True),
        "hedging": _compose(base, _fill(apply_edits(template, edits["hedging"]), *base), cfg),
        # Header keeps the standard spelling; every mention in the account uses the alternative.
        "name_spelling": _compose(som, _fill(template, v["given_alt"], v["family"]), cfg),
    }


def _undo_edits(text: str, edits: list[dict]) -> str | None:
    for e in edits:
        if text.count(e["replacement"]) != 1:
            return None
        text = text.replace(e["replacement"], e["original"])
    return text


def diff_check(versions: dict[str, str], template: str, edits: dict, pair: dict, cfg: dict) -> dict[str, str]:
    """Undo each variant's single intended change; the result must equal its contrast version."""
    b, v = pair["baseline"], pair["variant"]
    results = {}

    def undo_name(t):
        t = t.replace(f"{v['given']} {v['family']}", f"{b['given']} {b['family']}")
        return re.sub(rf"\b{re.escape(v['given'])}\b", b["given"], t)

    undo = {
        "name": undo_name,
        "religious": lambda t: _undo_edits(t, edits["religious"]),
        "hedging": lambda t: _undo_edits(t, edits["hedging"]),
        "interpretation": lambda t: t.replace(_interp_sentence(cfg) + "\n\n", "", 1),
        "name_spelling": lambda t: re.sub(rf"\b{re.escape(v['given_alt'])}\b", v["given"], t),
    }
    for variant, _factor, contrast in VERSIONS[1:]:
        text = versions[variant]
        if text == versions[contrast]:
            results[variant] = "variant identical to its contrast"
            continue
        restored = undo[variant](text)
        results[variant] = "ok" if restored == versions[contrast] else "changes beyond the target factor"
    if versions["name_spelling"].count(v["given_alt"]) < 1 or v["given"] not in versions["name_spelling"]:
        results["name_spelling"] = "does not contain both spellings"
    return results


def changed_words(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
