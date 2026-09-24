"""Turn first-token logprobs into scores.

Responses store the raw top logprobs, and the analysis recomputes scores from them, so a
change here never requires re-querying the models.
"""

import math

DIGITS = range(1, 8)


def _first_top(logprobs: list[dict] | None) -> list[tuple[str, float]]:
    if not logprobs:
        return []
    first = logprobs[0]
    top = [(t, lp) for t, lp in first["top"]]
    if not any(t == first["token"] for t, _ in top):  # the sampled token is always a candidate
        top.append((first["token"], first["logprob"]))
    return top


def rating(logprobs: list[dict] | None) -> dict:
    """Distribution over the digits 1-7 at the first token.

    `mass` is the raw probability on digit tokens (low = the model did not answer with a digit,
    or the digits fell outside the returned top-k). `score` is the expected rating after
    renormalising over the digits.
    """
    probs = {d: 0.0 for d in DIGITS}
    for tok, lp in _first_top(logprobs):
        t = tok.strip()
        if t.isdigit() and int(t) in probs:
            probs[int(t)] += math.exp(lp)
    mass = sum(probs.values())
    if mass == 0:
        return {"mass": 0.0, "score": None, "probs": None, "argmax": None}
    norm = {d: p / mass for d, p in probs.items()}
    return {
        "mass": mass,
        "score": sum(d * p for d, p in norm.items()),
        "probs": norm,
        "argmax": max(norm, key=norm.get),
    }


def choice(logprobs: list[dict] | None, options: dict[str, str]) -> dict:
    """Distribution over named options (e.g. {"grant": "GRANT", "refuse": "REFUSE"}) at the first token.

    A token counts toward an option when, stripped and upper-cased, it is a non-empty prefix of
    that option and of no other.
    """
    probs = {k: 0.0 for k in options}
    for tok, lp in _first_top(logprobs):
        t = tok.strip().strip("\"'*").upper()
        if not t:
            continue
        hits = [k for k, word in options.items() if word.startswith(t)]
        if len(hits) == 1:
            probs[hits[0]] += math.exp(lp)
    mass = sum(probs.values())
    if mass == 0:
        return {"mass": 0.0, "probs": None, "argmax": None}
    norm = {k: p / mass for k, p in probs.items()}
    return {"mass": mass, "probs": norm, "argmax": max(norm, key=norm.get)}
