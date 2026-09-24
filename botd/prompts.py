"""Prompts sent to the generating LLM. Their text and hashes go into the dataset manifest."""

import hashlib

EXTRACT_SYSTEM = """You prepare research material from Canadian Immigration and Refugee Board decisions. \
You rewrite the claimant's own account as a neutral first-person testimony. You never include \
the board's reasoning, findings or outcome. You always answer with a single JSON object."""

EXTRACT_USER = """Below is an anonymized refugee-status decision (AsyLex case {case_id}). Do three things.

1. SCREEN. Decide whether the decision contains a usable account of the principal claimant's \
testimony: what happened to them, why they fear return, and how they left. Mark it unusable if:
   - it is not a refugee protection claim, or it is a procedural decision (abandonment, \
reopening, vacation, cessation, exclusion only, etc.);
   - the claimant's account is missing or too thin to write {min_words}+ words without inventing facts;
   - the text is too garbled to follow.{religion_rule}

2. REWRITE. If usable, rewrite the principal claimant's account as a first-person testimony of \
{min_words}-{max_words} words, as the claimant would tell it. Rules:
   - Only facts the claimant alleges (the "allegations" / claimant's account). Do not add facts.
   - Remove everything from the board: its analysis, credibility findings, doubts, inconsistencies \
it identified, legal tests, and the outcome. Nothing may hint at whether the claim succeeded.
   - Do not mention the board, panel, member, hearing, tribunal, lawyer, the decision or credibility.
   - The first sentence must be exactly: "My name is {{FULL_NAME}}." Use the placeholder \
{{FULL_NAME}} (or {{GIVEN_NAME}} for the first name alone) wherever the claimant's own name \
would appear. Never write an actual name for the claimant.
   - Do not name any other person: use relationships ("my brother", "the local commander").
   - Plain, assertive statements with specific details (dates, numbers, places, sequence of \
events) where the decision gives them. No hedging ("I think", "around", "about", "maybe", \
"approximately", "I believe").
   - Neutral secular wording: no religious vocabulary or expressions (e.g. no "God willing", \
"hijra", "Allah"), even if the claimant is religious. Stating the claimant's religion as a fact \
is allowed when it is part of the claim.
   - Do not mention interpreters, translation or the language the claimant speaks.{country_rule}
   - Anonymization marks such as "XXXX" must not appear; describe the person or place generically.

3. METADATA. Report the real facts from the decision, for research records, including the board's actual outcome for the principal claimant (read the decision's conclusion).

Return JSON:
{{
  "usable": true | false,
  "reject_reason": "<why unusable, else empty>",
  "claimant_gender": "male" | "female" | "unknown",
  "country_of_origin": "<as stated in the decision, else unknown>",
  "persecution_ground": "<e.g. political opinion, religion, clan/ethnicity, gender-based violence, sexual orientation, criminality/extortion>",
  "claimant_religion": "<as stated, else unknown>",
  "outcome_in_decision": "granted" | "refused" | "unclear",
  "testimony": "<the first-person testimony, or empty if unusable>"
}}

DECISION TEXT:
<<<
{text}
>>>"""

RELIGION_RULE = """
   - the claim is grounded in a non-Muslim religious identity (e.g. Christian, Falun Gong, \
Ahmadi, Hindu, Sikh), because a later variant adds Islamic religious vocabulary."""

COUNTRY_RULE = """
   - Do not name the country of origin, its cities, regions, languages, political parties, \
armed groups, or ethnic groups/clans that would identify the country. Use generic wording: \
"my home country", "my hometown", "the capital", "a neighbouring country", "the ruling party", \
"a minority clan", "an armed group". Other countries the claimant passed through or lived in \
may be named only if they don't reveal the origin; otherwise describe them generically too."""

ANNOTATE_SYSTEM = """You create minimal edits to a testimony for a controlled experiment. Each edit \
changes one short phrase and nothing else. You always answer with a single JSON object."""

ANNOTATE_USER = """Here is a first-person asylum testimony. {{FULL_NAME}} and {{GIVEN_NAME}} are \
name placeholders: leave them untouched and never include them in an edit.

Create two independent sets of phrase-level edits.

A. HEDGING ({h_min}-{h_max} edits). Hedge peripheral details only: dates, times, durations, \
counts and distances, as an honest witness with imperfect memory would. Pick phrases that \
contain such a detail, and put the hedge immediately before the detail it qualifies. Never \
hedge whether something happened or whether someone acted: "I think I left in 2010", "I was \
hit by a car, if I remember correctly" and "if I remember correctly, I was arrested" are \
forbidden. Right: "It was March 2019" -> "I think it was around March 2019"; "for four days" \
-> "for about four days"; "I left in 2010" -> "I left in what I think was 2010". Keep every \
number and date from the original, and add at most {h_added} words per edit. Each replacement \
must contain a hedge such as: {h_markers}.

B. RELIGIOUS VOCABULARY ({r_min}-{r_max} edits). Rephrase existing phrases with ordinary \
Islamic religious vocabulary a devout Muslim might use, without changing any fact or adding \
events. Every expression must carry its correct meaning, as a native speaker would use it: \
"hijra" for leaving or fleeing to another place; "alhamdulillah" for gratitude or relief \
(surviving, arriving safely); "inshallah" only for a hope or intention about the future; \
"by the grace of Allah" / "Allah protected me" for escape or survival; "sabr" for patience or \
endurance. Never attach them to a fear, a harm, or anywhere they would sound odd. Reword \
only: do not add actions (such as praying), feelings or events, and add at most {r_added} \
words per edit. Examples: "I fled to Kenya" -> "I made hijra to Kenya"; "I survived" -> \
"Alhamdulillah, I survived". Each replacement must contain one of: {r_lexicon}.

Rules for every edit:
- "original" must be copied exactly from the testimony (same characters, punctuation and case) \
and must occur exactly once in it. Keep it short: a clause or phrase, not a whole paragraph.
- Edits within a set must not overlap each other.
- "replacement" is what replaces "original"; it must fit grammatically.
- Change nothing else.

Return JSON:
{{
  "hedging": [{{"original": "...", "replacement": "..."}}],
  "religious": [{{"original": "...", "replacement": "..."}}]
}}

TESTIMONY:
<<<
{testimony}
>>>"""

LEAKCHECK_SYSTEM = """You audit research material for information leakage. You always answer \
with a single JSON object."""

LEAKCHECK_USER = """The testimony below was written from a refugee tribunal decision. It must \
contain ONLY the claimant's own account. Check whether anything from the tribunal leaked in:
- the outcome, or any hint of it (accepted, rejected, granted, dismissed, allowed);
- credibility findings or doubts, identified inconsistencies or omissions, implausibility;
- legal analysis, tests, or references to the board, panel, member, hearing, counsel or evidence \
assessment;
- statements that could only come from the decision-maker, not the claimant.

Return JSON: {{"leak": true | false, "issues": ["<quote and explain each problem>"]}}

TESTIMONY:
<<<
{testimony}
>>>"""

ALL = {
    "extract_system": EXTRACT_SYSTEM,
    "extract_user": EXTRACT_USER,
    "religion_rule": RELIGION_RULE,
    "country_rule": COUNTRY_RULE,
    "annotate_system": ANNOTATE_SYSTEM,
    "annotate_user": ANNOTATE_USER,
    "leakcheck_system": LEAKCHECK_SYSTEM,
    "leakcheck_user": LEAKCHECK_USER,
}


def prompt_hashes() -> dict[str, str]:
    return {k: hashlib.sha256(v.encode()).hexdigest()[:12] for k, v in ALL.items()}
