# Research Proposal: Bias in LLM Credibility Judgments on Asylum Testimony

## Question

When a small language model assesses asylum testimony, does its credibility judgment change with details that should not matter: the claimant's name, religious vocabulary, whether the testimony was interpreted, or ordinary hedging?

## Why it matters (Q1)

Most refused refugee claims turn on credibility, not on the law. Tribunal guidance (e.g. UNHCR's) warns that hedging, interpreted testimony and small inconsistencies are **not** signs of lying: memory under trauma is imperfect, and interpretation loses nuance. LLMs are already being used to summarize, triage and draft in immigration work. If a model learned the opposite lesson from its training data (that hedged, translated or Muslim-coded testimony is less believable), it would quietly add bias to the most consequential step in the process.

Standard bias benchmarks (BBQ, StereoSet and similar) test short, synthetic sentences about groups. They do not test long, realistic legal narratives where the identity cue is incidental and the decision is a credibility judgment.

## Data

**Source:** a small subset of **AsyLex** (Barale et al., 2023), a corpus of anonymized Canadian refugee-status decisions from the Immigration and Refugee Board, with outcome labels.

**Subset:** about **100 decisions**, balanced between granted and refused claims, chosen for having a clear summary of the claimant's account.

**Preparation (LLM-assisted):** a frontier LLM on Azure AI Foundry (funded by $1,000 in Azure credits) builds the passages from the AsyLex decisions. This model is from a different model family than the one being evaluated. For each decision it:

1. Screens the decision for a usable account of the claimant's testimony.
2. Rewrites that account as a first-person testimony of about 200–400 words.
3. Removes the board's analysis, credibility findings and outcome, so the evaluated model sees only the claim.

**Controlled variants:** each passage is turned into matched versions that differ in exactly one factor.

- Where possible the edit is done by simple, rule-based substitution, not by the LLM: swapping the name, inserting the interpreter sentence, changing a name's spelling. This guarantees nothing else changes.
- The LLM is used only where the edit needs rewriting (hedging, religious vocabulary), with instructions to change nothing else.

| Factor               | Baseline                                    | Variant                                                |
| -------------------- | ------------------------------------------- | ------------------------------------------------------ |
| Name                 | Neutral / European name (e.g. _Lars Olsen_) | Somali name (e.g. _Mahamed Abdi Farah_)                |
| Religious vocabulary | "I fled to Kenya"                           | "I made hijra to Kenya"                                |
| Interpretation       | No mention                                  | "Testimony was given through a Somali interpreter"     |
| Hedging              | "It was March 2019"                         | "I think it was around March 2019"                     |
| Name spelling        | Consistent spelling                         | Two spellings across the account (_Mohamed / Maxamed_) |

That gives roughly 100 passages × 6 versions (the baseline plus five variants) ≈ **600 prompts**. The account of persecution stays the same in every version.

**Quality checks:**

- An automated comparison with the baseline confirms that each variant changes only its target factor.
- A second LLM pass checks that no outcome or credibility finding from the board leaked into the passage.
- About 20% of passages and variants are checked by hand.

**Cost:** 100 decisions is on the order of a few million tokens, well under $100. The remaining credits leave room to scale the dataset in a follow-up study.

## Model

**Primary: DeepSeek V4 Pro** (instruction-tuned).

- Strong reasoning and among the most-downloaded open-weight models, so its behavior is a reasonable stand-in for current frontier models.

**Comparison: GPT-6 Luna**, a model from a different developer, good at high-volume data and preparatory work.

## Method

1. **Credibility score:** prompt the model, acting as a decision assistant, to rate credibility from 1 to 7. Take the probability-weighted average over the digit tokens instead of sampling text, which is deterministic and needs one forward pass per prompt.
2. **Decision:** ask whether to grant or refuse, and score by the log-probabilities of the two answers.
3. **Stated principles:** separately ask the model directly whether hedging, interpretation or a Muslim name should affect credibility. Comparing this answer with steps 1–2 exposes any gap between what the model says and what it does.

**Analysis:**

- For each factor, the mean paired change in credibility score, with a Wilcoxon signed-rank test and bootstrap confidence intervals.
- The rate at which the grant/refuse decision flips.
- A secondary check: how the model's baseline scores relate to the real AsyLex outcomes.

**Stretch goal:** a linear probe on the model's internal activations, to test whether "low credibility" is represented before the model gives an answer.

## Path forward (Q3)

These are speculative and will not be tested:

- **Counterfactual data augmentation:** fine-tune on matched pairs of variants with the same credibility label.
- **Consistency training:** DPO or a consistency penalty that pushes the model toward the same answer on each pair of variants.
- **Adding the relevant guidance to training data:** include credibility-assessment guidance (UNHCR, IRB Chairperson's Guidelines) in instruction tuning, so that the model's stated principles match its behavior.
- **Testing before deployment:** any legal-triage use of an LLM should first pass a matched-variant audit like this one.

## Future directions

This study is a pilot to show the question is worth a full study. If it finds effects, a larger study would add country of origin.

AsyLex has no country-of-origin field, but citizenship can be recovered from its `CLAIMANT_INFO` entities ("citizen of …"). Its `GPE` entities are noisier, since they mix countries of origin and transit.

Comparing raw scores across countries would be confounded: persecution type, strength of evidence and real grant rates all vary by country. Country is also legitimately relevant to _risk_ (whether the fear is well-founded), though not to _credibility_. The larger study would therefore use country in two controlled ways:

- **Sampling across regions, with effects compared by region:** sample decisions across regions, then test whether the within-testimony effects (name, hedging, interpretation) are larger for some groups. For example, does hedging cost a Somali claimant more than a Colombian one?
- **Comparison with real outcomes:** per region, compare the model's grant probability with the tribunal's actual decision. "The model is harsher than the tribunal on claimants from X" is defensible where a raw gap between regions is not.

**A sharper contrast than Global North vs. South:** Canada's Designated Countries of Origin policy (2012–2019) marked mostly European countries, including Hungary and the Czech Republic, as "safe". Claims from those countries, many of them from Roma, were fast-tracked as presumed unfounded. Comparing Roma claimants from Hungary and the Czech Republic with Somali, Nigerian and Colombian claimants would show whether a model copies the tribunal's skepticism toward "safe country" claimants, the general pattern of bias against Global South claimants, or both.

**Related work.** Neither of these papers perturbs testimony to test for bias in LLM credibility judgments:

- _LLMs as annotators of credibility assessment in Danish asylum decisions_ (arXiv 2605.13412, 2026) uses LLMs to classify credibility assessments in 273 Danish appeals decisions. It evaluates accuracy, not bias.
- _When Fairness Isn't Statistical_ (arXiv 2506.03913, 2025) analyses fairness across AsyLex decisions, including clustering by country of citizenship, but does not evaluate LLMs. Its argument that statistical gaps do not prove unfairness supports this study's within-testimony design.

## Deliverables (public Hugging Face Bucket)

- The evaluation dataset: passages, variants and prompts, with a note on AsyLex provenance and licence.
- The generation pipeline: the prompts given to the generating LLM, its model version, and the AsyLex case IDs each passage came from.
- The Colab notebook, and raw scores for every prompt.
- Results figures: effect size per factor and decision-flip rates.
- A README answering Q1 and Q3.

## Ethics and limitations

- AsyLex decisions are anonymized, and no new personal data is collected. The variant names are invented.
- The data is Canadian, not South African. Credibility doctrine is broadly shared across jurisdictions, but conclusions are limited to this setting.
- One hundred decisions is small: the study is a demonstration of a blind spot, not a full audit.
- The testimonies are LLM-rewritten summaries, not verbatim claimant speech. The generating LLM may bring in its own style or biases. Rule-based edits for most factors, diff checks and manual spot-checks limit this, and it applies equally to every version of a passage.
- AsyLex text is sent to Azure only for this research. Azure's standard terms say API data is not used to train models.
- AsyLex is licensed CC BY-NC-SA 4.0 (research use only), so the derived evaluation dataset is released under the same licence.
