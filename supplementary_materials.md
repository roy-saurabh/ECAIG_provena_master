# Supplementary Materials

**Preventing Evidence Laundering in AI Governance Audits: Epistemic Ceilings for Context-Conditioned Evidence Sufficiency**

These supplementary materials accompany the main manuscript and contain three artifacts referenced from main-text Appendices B, C, and D: the full evidence-family definitions table (Section S1), the full input-mode quality rubric (Section S2), and the full illustrative normalization templates for eight families (Section S3). The main manuscript is self-contained; these materials provide implementation-level detail for adopters who wish to instantiate, modify, or extend the framework. All artifacts here also exist in machine-readable form in `provena_validation.py`.

---

## S1   Evidence-Family Definitions

Compact definitions of all fifteen families: construct, minimum observable unit, candidate metrics, characteristic limitation type, and required-in contexts (from main-text Table 2). Definitions are consistent with the illustrative templates in Section S3 of this supplement.

**Table S1: Evidence-family definitions.**

| F | Family | Construct & minimum observable unit | Candidate metrics | Characteristic limitation | Required-in |
|---|---|---|---|---|---|
| F1 | Data quality | Fidelity of training/eval data to its declared schema and population; observed at the per-record level. | Schema conformance, missingness rate, drift-vs-deployment statistic | Dataset coverage | C1–C8 |
| F2 | Model performance | Predictive accuracy under deployment conditions; observed via per-slice predictions. | Calibration-weighted slice score, group worst-case accuracy | Test-distribution validity | C1–C8 |
| F3 | Drift | Stability of input distribution and model output over time; observed via temporal monitoring. | PSI, KL divergence, MMD | Reference-period ambiguity | C1, C2, C3 (rec.), C4, C5, C8 |
| F4 | Reproducibility | Re-executability of the measurement pipeline under fixed inputs and seeds. | Bit-identical / metric-tolerance reproduction | Pipeline state externality | C2–C5, C7, C8, C10 |
| F5 | Fairness | Disparity across protected or salient groups; observed via group-conditioned metrics. | Equalized-odds gap, calibration deviation per group | Group-coverage and metric-choice | C2–C4, C7, C8 |
| F6 | Explainability | Local or global rationales linking inputs to outputs; observed via attribution or surrogate. | Feature attribution faithfulness, surrogate fidelity | Faithfulness vs. plausibility tradeoff | C2–C8 |
| F7 | Privacy | Resistance to membership and attribute inference; observed via attack-based evaluation or DP bound. | MIA advantage, (ε, δ)-DP parameters | Threat-model scope | C1–C5, C7, C8 |
| F8 | Lineage | Traceability from data source through model artifact to deployment. | Hash chain coverage, source-record-to-prediction trace | Custody-gap risk | All (varies in strength) |
| F9 | Robustness | Stability under documented perturbation; observed via worst-group accuracy and shift sensitivity. | Worst-group accuracy under budgeted shift, shift sensitivity index | Threat-model scope | C2, C4, C5, C8 |
| F10 | Security | Resistance to adversarial inputs, model extraction, and supply-chain compromise. | Red-team pass rate, fuzz coverage, dependency scan | Adversary-budget scope | All |
| F11 | Agent boundary | Scope and authorization of tool-use, multi-step actions, and external calls (agentic systems). | Policy-violation rate, scope-creep incidence | Sandbox fidelity | C4, C5 (req.), C8 |
| F12 | Policy decision | Documented governance decisions and adjudicated trade-offs prior to deployment. | Coverage of policy-decision artifacts vs. risk register | Documentation freshness | C2–C5, C7–C10 |
| F13 | Platform observability | Continuous visibility into latency, error rates, drift, and policy compliance in production. | Telemetry coverage, alerting MTTR | Telemetry-pipeline reliability | C1–C8, C10 |
| F14 | Export integrity | Integrity and completeness of the audit artifact itself (signatures, hashes, manifest). | Signature verification rate, manifest coverage | Transport tampering | All |
| F15 | RAG quality | Groundedness, citation precision, and answer correctness for retrieval-augmented systems. | Groundedness, citation precision, correctness | Retrieval-corpus coverage | C4 (rec.), C5 (rec.), C6 (req.), C7–C8 (rec.) |

---

## S2   Input-Mode Quality Rubric

Each input mode is scored on six quality attributes on a 0, 1, 2, 3 scale. The ceilings `ceil(m, v)` in main-text Table 3 are the worst-case bounds implied by Q(m) across the six attributes (origin, representativeness, measurability, reproducibility, tamper resistance, audit traceability).

**Table S2: Input-mode quality rubric. Verified rows reflect v = 1.**

| Mode | Origin | Repr. | Meas. | Reprod. | Tamper | Trace |
|---|---|---|---|---|---|---|
| Edge Local | 3 | 3 | 3 | 3 | 3 | 3 |
| Federated (unverified) | 3 | 3 | 3 | 2 | 1 | 1 |
| Federated (verified) | 3 | 3 | 3 | 2 | 3 | 3 |
| Uploaded Sample | 2 | 2 | 3 | 2 | 2 | 2 |
| Workspace Asset (unverified) | 2 | 2 | 1 | 1 | 1 | 1 |
| Workspace Asset (verified) | 2 | 2 | 1 | 1 | 2 | 2 |
| External Provider (unverified) | 2 | 2 | 2 | 1 | 1 | 1 |
| External Provider (verified) | 2 | 2 | 2 | 1 | 2 | 2 |
| Declared Supplier Evidence | 1 | 1 | 0 | 0 | 0 | 1 |
| Synthetic Fixture | 0 | 0 | 3 | 3 | 3 | 3 |
| Not Assessable | 0 | 0 | 0 | 0 | 0 | 0 |

---

## S3   Illustrative Normalization Templates

For eight families we supply illustrative templates: candidate mapping g_f, default threshold vector τ_f, adjudication procedure A_f, and documented limitation type L_f. These templates are included to make the reference implementation executable; they are not proposed as consensus governance thresholds. The structural guarantees of main-text Section 9 hold for any choice of τ_f. Empirical threshold calibration is identified as future work in main-text Section 16.

**F1 Data quality.** q = 100 · (w_S·S + w_M·(1 − M) + w_D·(1 − D)); S = schema conformance rate, M = missingness rate, D = clipped drift-vs-deployment statistic; default (w_S, w_M, w_D) = (0.4, 0.3, 0.3). *Limitation:* dataset coverage. *Threshold status:* provisional.

**F2 Model performance.** q = 100 · min over groups g of s_g, where s_g is the calibration-weighted slice score for group g. *Limitation:* test-distribution validity. *Threshold status:* provisional; slice list is deployment-specific.

**F3 Drift.** q = 100 · (1 − clip(PSI/2.5)); PSI substituted by KL or MMD where PSI is undefined. *Limitation:* reference-period ambiguity. *Threshold note:* the 2.5 clip is a conventional "major drift" boundary in operational monitoring practice; it is provisional and subject to calibration.

**F5 Fairness.** q = 100 · (1 − clip(Δ_eo/0.2)), where Δ_eo is the equalized-odds gap; complemented by per-group calibration deviation. *Limitation:* group-coverage and metric-choice. *Threshold note:* the 0.2 value is a provisional operational threshold, inspired by common disparity-screening heuristics but *not equivalent* to the four-fifths rule or any legal test. The EEOC four-fifths rule is a selection-rate ratio comparison and is conceptually distinct from the equalized-odds gap used here.

**F7 Privacy.** q = 100 · (1 − min(MIA/0.5, 1)), where MIA is the membership-inference advantage above random (range 0–0.5). If differential privacy is applied with parameters (ε, δ), q is bounded above by q_DP(ε, δ) = 100 · max(0, 1 − ε/4) · (1 − δ)⁺. This bound is a conservative, monotonic decision rule: q decreases as ε grows or as δ grows. *Limitation:* threat-model scope.

**F9 Robustness.** q derived from worst-group accuracy under a documented threat model with the perturbation budget recorded as part of the limitation. The current template conflates performance and robustness when the threat model is weak; an adequate template requires both a *baseline accuracy gap* (worst-group minus reference) and a *shift sensitivity index* (rate of metric degradation per unit perturbation budget). *Limitation:* threat-model scope.

**F14 Export integrity.** q = 100 if signature verifies, all artifact hashes match, and manifest is complete; intermediate values reflect partial manifest coverage proportional to the fraction of artifacts hashed and signed.

**F15 RAG quality.** q = 100 · (0.5 g_ground + 0.3 g_citation + 0.2 g_correct), computed over a held-out retrieval-paired test set, where g_ground = fraction of generated atoms supported by retrieved passages (human-adjudicated or NLI-adjudicated), g_citation = fraction of cited passages that contain the claimed atom (precision), g_correct = exact-match or graded correctness on the final answer. *Limitation:* retrieval-corpus coverage.
