# Research Proposal: Bias in LLM Credibility Judgments on Asylum Testimony

## Question

When a language model assesses asylum testimony, does its credibility judgment change with details that should not matter: the claimant's name, religious vocabulary, whether the testimony was interpreted, or how consistently the claimant's name is spelled?

## Summary of the pilot

The pilot has been run. Three models rated 100 asylum testimonies (50 granted, 50 refused, from AsyLex) in six matched versions each.

**Main finding: a Somali interpreter lowers the chance of a grant; a Spanish interpreter barely does.** The sentence "This testimony was given through a Somali interpreter" lowers P(grant) for gpt-6-luna (−9.9 points, p < .001) and Qwen3.5-4B (−4.0 points, p < .001), and decisions flip almost only toward refusal (14 vs 2 passages for Luna, 8 vs 0 for Qwen). The same sentence naming a Spanish interpreter has no significant effect on either model. The penalty therefore attaches to what "Somali" signals about the claimant, not to interpretation itself. Asked directly, both models say an interpreter should make no difference.

DeepSeek-V4-Pro shows no comparable bias. Details are under [Results](#results).

## Why it matters (Q1)

Most refused refugee claims turn on credibility, not on the law. Credibility guidance (UNHCR, 2013; IRB, 2020) warns that interpreted testimony and small inconsistencies, such as a name spelled two ways, are **not** signs of lying: memory under trauma is imperfect, interpretation loses nuance, and transliteration varies. LLMs are already being used to summarize, triage and draft in immigration work. If a model learned the opposite lesson from its training data (that translated, inconsistent or Muslim-coded testimony is less believable), it would quietly add bias to the most consequential step in the process.

Standard bias benchmarks (BBQ: Parrish et al., 2022; StereoSet: Nadeem et al., 2021; and similar) test short, synthetic sentences about groups. They do not test long, realistic legal narratives where the identity cue is incidental and the decision is a credibility judgment.

## Data

**Source:** **AsyLex** (Barale et al., 2023), a corpus of anonymized Canadian refugee-status decisions from the Immigration and Refugee Board, pinned to a fixed dataset revision.

**Subset:** **100 decisions**, 50 granted and 50 refused, from first-instance refugee claims only (the Refugee Protection Division and its pre-2002 predecessor). AsyLex's silver outcome labels proved unreliable for grants, so each decision is labelled from its own determination wording ("is a Convention refugee", "the claim is rejected"); ambiguous decisions are dropped. Candidates are drawn in a seeded, label-balanced order, and extras replace decisions that fail screening.

**Preparation (LLM-assisted):** a frontier LLM on Azure AI Foundry, **gpt-6.1-sol** (low reasoning effort, temperature 0, fixed seed), builds the passages, funded by $1,000 in Azure credits. For each decision it:

1. Screens the decision for a usable account of the claimant's testimony. Claims grounded in a non-Muslim religious identity (e.g. Christian, Falun Gong, Ahmadi) are dropped, since the Islamic-vocabulary variant would be incoherent.
2. Rewrites that account as a first-person testimony of 200–400 words, with placeholders for the claimant's name.
3. Removes the board's analysis, credibility findings and outcome, so the evaluated model sees only the claim. A decision is rejected if the LLM reads a different outcome than the label.
4. Replaces the country of origin, its places and country-identifying groups with generic wording ("my home country"), so the name, interpreter and religious cues do not clash with the story. The real country is kept as metadata.

The generator was meant to come from a different model family than every evaluated model. gpt-6.1-sol and gpt-6-luna are both OpenAI models; see [Limitations](#ethics-and-limitations).

**Controlled variants:** each passage is turned into matched versions that differ in exactly one factor. All six versions are built by rule-based substitution from one template, so nothing else can change. For religious vocabulary, the LLM proposes 2–4 exact-substring phrase edits (at most 5 added words each); a rule rejects edits that add actions or content, and the accepted edits are then applied by substitution.

| Factor                   | Baseline                                    | Variant                                                 | Compared with           |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------- | ----------------------- |
| Name                     | Neutral / European name (e.g. _Lars Olsen_) | Somali name (e.g. _Mahamed Abdi Farah_; 8 per gender)   | Baseline                |
| Religious vocabulary     | "I fled to Kenya"                           | "I made hijra to Kenya"                                 | Baseline                |
| Interpretation           | No mention                                  | "This testimony was given through a Somali interpreter." | Baseline                |
| Interpretation (control) | No mention                                  | "This testimony was given through a Spanish interpreter." | Baseline                |
| Name spelling            | Consistent Somali spelling                  | Two spellings across the account (_Mahamed / Maxamed_)  | Consistent Somali name  |

Comparing name spelling against the consistently spelled Somali name holds name origin constant. Comparing the two interpreter variants separates the effect of mentioning an interpreter from that of the language named.

That gives 100 passages × 6 versions = **600 prompts**. The account of persecution is identical in every version.

**Quality checks:**

- A diff check undoes each variant's single change and confirms the result equals its contrast version.
- A second LLM pass checks that no outcome or credibility finding from the board leaked into the passage.
- A seeded 20% sample (20 passages, all their variants) is set aside for hand checking (`data/benchmark/manual_review.csv`). **This review has not been done yet.**

**Who dropped out during generation.** Reaching 100 passages took 222 screened candidates. The rejections were not even across outcome classes:

| Rejection reason            | Granted | Refused |
| --------------------------- | ------: | ------: |
| Screened out (no usable account, non-Muslim religious claim, …) | 54 | 23 |
| LLM read a different outcome than the label | 32 | 2 |
| Blocked by Azure's content filter | 7 | 4 |

Asylum testimony describes torture, sexual violence and killings, and Azure's default content filters blocked 11 rewrites. Because the final set is balanced by outcome, this cannot shift the grant/refuse mix, but it may remove the most severe persecution accounts, slightly more often among granted claims. The high label-mismatch rate for grants mirrors the unreliable AsyLex labels noted above. Adjusted content-filter settings for research use would remove the filter loss in a follow-up.

**Cost:** extraction used about 1.8 million tokens, well under $100. The remaining credits leave room to scale the dataset.

## Models

| Model | Role | Settings |
| --- | --- | --- |
| **gpt-6-luna** | Frontier, OpenAI | Reasoning model; Azure only returns logprobs with `reasoning_effort = "none"`, temperature fixed at 1, top 5 logprobs (digits outside the top 5 count as zero) |
| **DeepSeek-V4-Pro** | Frontier, open-weight | Temperature 0, top 20 logprobs |
| **Qwen3.5-4B** | Small, open-weight, on Foundry managed compute | Temperature 0, top 20 logprobs, thinking off so the answer is the first token |

Three further small models were evaluated and **dropped because they did not give conclusive answers**: Gemma 4 E2B (`google--gemma-4-e2b-it`), Qwen3.5-0.8B and Qwen3.5-2B. Gemma, for example, rated every baseline passage 7/7, so its credibility scale could not move, and wrote commentary instead of a one-word answer to the grant/refuse and principle questions. None of the three is included in the findings.

## Method

Every request is a separate single-turn conversation with a one-token answer, scored from the first token's logprobs rather than from sampled text. Responses with less than 0.5 probability on valid answer tokens are excluded.

1. **Credibility score:** the model, acting as a decision assistant, rates credibility from 1 to 7. The score is the probability-weighted average over the digit tokens.
2. **Decision:** the model answers GRANT or REFUSE. The score is P(grant), renormalised over the two answers.
3. **Stated principles:** asked directly, with no testimony, whether each factor should make testimony LESS or MORE credible or leave it the SAME (three paraphrases per factor). Comparing this answer with steps 1–2 exposes any gap between what the model says and what it does.

**Analysis:**

- For each factor, the mean paired change in credibility and in P(grant) against the contrast version of the same passage, with 95% bootstrap CIs over passages and Wilcoxon signed-rank tests, Holm-corrected across the five factors within each model and measure.
- Decision flip rates, with an exact binomial test on the direction of the flips.
- Stated principles against observed behavior.
- How baseline scores relate to the tribunal's real outcome (AUC, agreement).
- Agreement between models.

## Results

Full numbers are in [data/results/analysis/](data/results/analysis/) (`summary.json`, `effects.csv`). Mean paired change versus the matched version; **bold** = Holm-corrected p < .05.

| Factor | gpt-6-luna: credibility | gpt-6-luna: P(grant) | Qwen3.5-4B: credibility | Qwen3.5-4B: P(grant) | DeepSeek-V4-Pro: credibility | DeepSeek-V4-Pro: P(grant) |
|---|---:|---:|---:|---:|---:|---:|
| Somali name | **−0.16** | +0.002 | **+0.02** | +0.001 | −0.02 | +0.013 |
| Islamic vocabulary | **−0.13** | −0.000 | −0.00 | −0.003 | +0.01 | **+0.036** |
| Somali interpreter | **−0.13** | **−0.099** | **−0.08** | **−0.040** | −0.02 | −0.013 |
| Spanish interpreter | −0.06 | −0.030 | +0.00 | −0.002 | +0.01 | +0.022 |
| Two spellings of the name | −0.01 | **−0.052** | **−0.02** | **−0.007** | −0.01 | +0.011 |

- **gpt-6-luna** is the most affected. The Somali name, Islamic vocabulary and the Somali interpreter each lower its credibility ratings. The Somali interpreter and inconsistent name spellings also lower its grant probability (spellings: −5.2 points, flips 8 to refuse vs 1 to grant). It states that none of these factors should matter (P(SAME) = 1.00), so for three of the five factors it says one thing and does another.
- **Qwen3.5-4B** reacts strongly and consistently to the Somali interpreter: credibility falls in 90% of passages (Cohen's dz = −1.31). Its other significant effects are tiny (≤ 0.02 on a 1–7 scale). It grants only 18% of baseline passages, so there is little room for decisions to move toward refusal. It too states that an interpreter should make no difference (P(SAME) = 0.99).
- **DeepSeek-V4-Pro** shows no credibility effects. The only significant effect is slightly *higher* grant probability with Islamic vocabulary (+3.6 points, p = .035). Its decisions flip in 16–22% of passages for every factor, but equally often in both directions, which looks like noise rather than bias.

![Change in credibility rating by factor](data/results/analysis/credibility_effects.png)

![Change in grant probability by factor](data/results/analysis/grant_effects.png)

![Decision flips by factor](data/results/analysis/decision_flips.png)

### How much to trust this

- **Baseline judgments track real outcomes only weakly.** The P(grant) AUC against the tribunal's decisions is 0.57 for Qwen, 0.59 for Luna and 0.65 for DeepSeek, with agreement on 56–58% of cases. The models are not good judges of these claims to begin with; the study measures how their judgments *shift*, not whether the judgments are right.
- **The effects are small in absolute terms.** On the 1–7 scale they are at most 0.16 points. The decision effects (up to 10 points of grant probability, and one-directional flips) matter more.
- **The generator and gpt-6-luna are both OpenAI models.** This cannot create the within-passage effects, since every version of a passage comes from the same template, but it could affect Luna's baseline judgments.
- **Content filtering removed a few evaluation prompts:** 6 of DeepSeek's credibility prompts, and 4 each of Luna's credibility and decision prompts.
- **One prompt wording, one seed and 100 passages.** The Holm correction covers the five factors within each model and measure, not all models together. The Somali-interpreter result survives either way (p < .001 in both models).

## Next steps

- Hand-check the 20 manual-review passages before publishing.
- Add a small model that answers in the required format (e.g. Phi-4-mini-instruct) in place of the dropped Gemma and small Qwen models.
- Test whether the Somali-interpreter effect holds with rephrased prompts and other language pairs (e.g. Tigrinya, Dari, Ukrainian).
- **Stretch goal (not attempted):** a linear probe on an open-weight model's internal activations, to test whether "low credibility" is represented before the model gives an answer.

## Path forward (Q3)

These are speculative and will not be tested:

- **Counterfactual data augmentation** (Lu et al., 2020): fine-tune on matched pairs of variants with the same credibility label.
- **Consistency training:** DPO (Rafailov et al., 2023) or a consistency penalty that pushes the model toward the same answer on each pair of variants.
- **Adding the relevant guidance to training data:** include credibility-assessment guidance (UNHCR, 2013; IRB, 2020) in instruction tuning, so that the model's stated principles match its behavior. The pilot shows the gap is real: gpt-6-luna and Qwen3.5-4B both state that an interpreter is irrelevant and then penalize a Somali one.
- **Testing before deployment:** any legal-triage use of an LLM should first pass a matched-variant audit like this one.

## Future directions

The pilot found effects, so a larger study is worth running. It would add country of origin.

AsyLex has no country-of-origin field, but citizenship can be recovered from its `CLAIMANT_INFO` entities ("citizen of …"), and the pilot's generator already records the country for each passage. Its `GPE` entities are noisier, since they mix countries of origin and transit.

Comparing raw scores across countries would be confounded: persecution type, strength of evidence and real grant rates all vary by country. Country is also legitimately relevant to _risk_ (whether the fear is well-founded), though not to _credibility_. The larger study would therefore use country in two controlled ways:

- **Sampling across regions, with effects compared by region:** sample decisions across regions, then test whether the within-testimony effects (name, religious vocabulary, interpretation) are larger for some groups. For example, does an interpreter cost a Somali claimant more than a Colombian one? The pilot's Somali-vs-Spanish interpreter contrast is a first version of this question.
- **Comparison with real outcomes:** per region, compare the model's grant probability with the tribunal's actual decision. "The model is harsher than the tribunal on claimants from X" is defensible where a raw gap between regions is not.

**A sharper contrast than Global North vs. South:** Canada's Designated Countries of Origin policy (2012–2019) marked mostly European countries, including Hungary and the Czech Republic, as "safe". Claims from those countries, many of them from Roma, were fast-tracked as presumed unfounded. Comparing Roma claimants from Hungary and the Czech Republic with Somali, Nigerian and Colombian claimants would show whether a model copies the tribunal's skepticism toward "safe country" claimants, the general pattern of bias against Global South claimants, or both.

**Related work.** Neither of these papers perturbs testimony to test for bias in LLM credibility judgments:

- _LLMs as annotators of credibility assessment in Danish asylum decisions_ (Humblot-Renaux et al., 2026) uses LLMs to classify credibility assessments in 273 Danish appeals decisions. It evaluates accuracy, not bias.
- _When Fairness Isn't Statistical_ (Barale et al., 2025) analyses fairness across AsyLex decisions, including clustering by country of citizenship, but does not evaluate LLMs. Its argument that statistical gaps do not prove unfairness supports this study's within-testimony design.

## Deliverables (public Hugging Face Bucket)

- The evaluation dataset (`data/benchmark/`): passages, variants and prompts, with AsyLex provenance and licence.
- The generation pipeline and its record: the prompts given to the generating LLM, its model version, the AsyLex case IDs each passage came from, and the per-case work logs (`data/work/`), which are the reproducible record of generation since Azure does not guarantee identical outputs.
- The evaluation code and raw responses for every prompt, with their top logprobs (`data/results/`).
- Results figures: effect size per factor and decision-flip rates.
- A README answering Q1 and Q3.

## Ethics and limitations

- AsyLex decisions are anonymized, and no new personal data is collected. The variant names are invented.
- The data is Canadian, not South African. Credibility doctrine is broadly shared across jurisdictions, but conclusions are limited to this setting.
- One hundred decisions is small: the study is a demonstration of a blind spot, not a full audit.
- The testimonies are LLM-rewritten summaries, not verbatim claimant speech. The generating LLM may bring in its own style or biases. Rule-based variants, diff checks and leak checks limit this, and it applies equally to every version of a passage. The manual spot-check is still outstanding.
- The generator (gpt-6.1-sol) shares a developer with one evaluated model (gpt-6-luna).
- Screening and content filtering removed more granted than refused candidates, so the passages may under-represent the most severe accounts (see [Data](#data)).
- AsyLex text is sent to Azure only for this research. Azure's standard terms say API data is not used to train models.
- AsyLex is licensed CC BY-NC-SA 4.0 (research use only), so the derived evaluation dataset is released under the same licence.

## References

- Barale, C., Rovatsos, M., & Bhuta, N. (2023). Automated Refugee Case Analysis: An NLP Pipeline for Supporting Legal Practitioners. _Findings of the Association for Computational Linguistics: ACL 2023_, 2992–3005. https://aclanthology.org/2023.findings-acl.187/
- Barale, C., Rovatsos, M., & Bhuta, N. (2025). When Fairness Isn't Statistical: The Limits of Machine Learning in Evaluating Legal Reasoning. arXiv:2506.03913. https://arxiv.org/abs/2506.03913
- Humblot-Renaux, G., Jahromi, M. N. S., Bakuri-Jørgensen, R., Heyl, M. A., Jarlner, A. S. S., Vlachou, M., Høgenhaug, A. M., Elliott, D., Gammeltoft-Hansen, T., & Moeslund, T. B. (2026). LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics. arXiv:2605.13412. https://arxiv.org/abs/2605.13412
- Immigration and Refugee Board of Canada (2020). _Assessment of Credibility in Claims for Refugee Protection_ (Legal Services reference paper; replaces the 2004 version). https://irb-cisr.gc.ca/en/legal-policy/legal-concepts/Pages/Credib.aspx
- Lu, K., Mardziel, P., Wu, F., Amancharla, P., & Datta, A. (2020). Gender Bias in Neural Natural Language Processing. In _Logic, Language, and Security_ (LNCS 12300), 189–202. Springer. https://doi.org/10.1007/978-3-030-62077-6_14
- Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring stereotypical bias in pretrained language models. _Proceedings of ACL-IJCNLP 2021_, 5356–5371. https://aclanthology.org/2021.acl-long.416/
- Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering. _Findings of the Association for Computational Linguistics: ACL 2022_, 2086–2105. https://aclanthology.org/2022.findings-acl.165/
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _Advances in Neural Information Processing Systems 36 (NeurIPS 2023)_. https://arxiv.org/abs/2305.18290
- UNHCR (2013). _Beyond Proof: Credibility Assessment in EU Asylum Systems_. Brussels: UNHCR. https://www.refworld.org/reference/regionalreport/unhcr/2013/en/104283
