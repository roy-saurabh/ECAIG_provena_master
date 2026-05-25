# Provena Validation — Anonymous Companion Repository

Anonymous computational companion for a FAccT-targeted manuscript on context-conditioned evidence sufficiency in AI governance auditing.

> **Reviewer note.** During double-blind review, please use the anonymous mirror linked in the manuscript. The de-anonymized repository will be linked after acceptance.
>
> **Scope note.** The computational studies test internal soundness and implementation fidelity under synthetic audit profiles. They do not establish practitioner calibration of the necessity matrix, band thresholds, input-mode quality scores, or evidence-family taxonomy.

---

## What Provena is

Provena is the implementation substrate for the *epistemic ceiling* framework described in the manuscript. The framework bounds the strength of AI governance claims by the provenance, context-specific necessity, and reproducibility of supporting evidence. Its central guarantee is a non-overridable ceiling on claim strength derived from the input mode of each evidence source (Edge Local, Federated, Uploaded Sample, and so on). A supplier-declared robustness report with a perfect score is capped at D; a synthetic fixture remains at C regardless of its raw metric value.

This repository implements the framework in Python and validates that it behaves as designed through five computational studies on synthetic audit profiles.

---

## What this script validates

1. **Framework correctness** — the ceiling, band, and meet functions satisfy their formal definitions.
2. **Sensitivity** — rank-order stability of the composite score under weight perturbation.
3. **Invariant** — the anti-laundering invariant (Corollary 2) holds empirically across constructed weak-evidence profiles.
4. **Ablation** — which framework components are load-bearing for the anti-laundering guarantee.
5. **Laundering-pathology comparison** — metric-only and other baselines versus the full framework on adversarially constructed weak-provenance profiles.
6. **Hard cap vs soft penalty vs metric-only** (Study 5, Tier 3 addition) — comparison on laundering-prone profiles.

## What this script does NOT validate

- That the context–family necessity matrix (Table 2 in the manuscript) is the matrix practitioners would produce in expert elicitation.
- That the band thresholds {0.30, 0.50, 0.70, 0.85} agree with expert-adjudicated audit outcomes.
- That the fifteen evidence families are complete or non-redundant relative to practitioner constructs.
- Any claim about real deployment behaviour.

> **Caution.** These computational studies test internal soundness and implementation fidelity under synthetic audit profiles. They do not establish that the illustrative context-family matrix, band thresholds, input-mode quality scores, or evidence-family taxonomy match practitioner judgment.

---

## Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
pytest>=7.0.0
```

**Why scipy?** Kendall τ and Spearman ρ are required for Study 1. A hand-rolled Kendall τ implementation requires O(n²) pair enumeration and is more error-prone than `scipy.stats.kendalltau`. No other scipy functions are used.

---

## Exact run commands

```bash
python -m pip install -r requirements.txt

# Run all studies (both generators), write validation_results.json
python provena_validation.py --all --deterministic-timestamp --out validation_results.json

# Reproduce the Section 11 worked examples, write sample_audit_report.json
python sample_audit.py

# Run tests
python -m pytest

# Anonymization check
python scripts/check_anonymization.py
```

Additional flags:

```bash
# Specific generator only
python provena_validation.py --all --generator uniform --out results_uniform.json
python provena_validation.py --all --generator context_correlated --out results_cc.json

# Specific studies
python provena_validation.py --study sensitivity --study invariant --out partial.json

# Print summary to stdout
python provena_validation.py --all --print-summary --out validation_results.json
```

---

## Output files

| File | Description |
|------|-------------|
| `validation_results.json` | All study results, framework constants, generator assumptions, manuscript tables |
| `GENERATED_RESULTS_AUDIT.md` | Auto-generated audit comparing generated numbers to manuscript claims |
| `REPO_READINESS.md` | Honest readiness declaration with per-item checklist |

---

## The two generators

### Uniform generator (seed 99)
Context sampled uniformly from C1–C10. Mode sampled uniformly from 7 assessable modes. Verification: Bernoulli(0.30). Score q: Uniform(0, 100). Reproducibility: Bernoulli(0.60). Coverage: at least ⌈0.5 × n_required⌉ required families selected; 0–3 optional families added.

### Context-correlated generator (seed 101)
Context-specific mode probability distributions (e.g., C9 vendor/supplier has 35% Declared Supplier Evidence; C4 health has 25% Federated). Mode-specific verification and reproducibility probabilities. Score q from Normal distributions by mode (e.g., Synthetic Fixture: Normal(88, 10); Edge Local: Normal(75, 15)), clipped to [0, 100]. Coverage: 50–95% of required families depending on context risk level. This generator intentionally makes Declared Supplier Evidence and Synthetic Fixture produce numerically high scores, which tests the anti-laundering guarantee under realistic-looking but epistemically weak evidence.

---

## Five studies

### Study 1: Sensitivity decomposition
Weight perturbation (δ ∈ {0.05, 0.10, 0.15}, 12 perturbations per δ, seed 7) on the composite score G = w₁·C_comp + w₂·C_iq + w₃·C_rep. Reports Kendall τ and Spearman ρ between baseline and perturbed G rankings, mean/max absolute G shift, pre-cap band-change rate, and final ClaimLevel-change rate. The three-layer decomposition (G → band → ClaimLevel) makes visible whether the hard cap absorbs most weight sensitivity.

### Study 2: Invariant testing
500 profiles per generator (seed 13). Each profile has at least one required-family finding forced to q=100, mode=Synthetic Fixture or Declared Supplier Evidence, verified=0. Checks that ClaimLevel ≤ ceil(forced_mode, 0) in every case.

### Study 3: Laundering-pathology comparison
300 profiles (seed 29) where all required families are present with q=100 and modes drawn only from Synthetic Fixture or Declared Supplier Evidence. Compares all 9 variants. The central headline: metric-only should promote all to A; the full framework caps at C or D.

### Study 4: Ablation
Six ablation/variant checks on the 500 main-generator profiles: (A) broaden missingness to all applicable families (R>0) — conservative, produces demotions only; (B) no context matrix and no missingness; (C) no input-mode ceiling; (D) no reproducibility term; (E) no hard cap on composite; (F) no missingness synthesis with R kept. Reports disagreement rate, promotions, demotions, and overstatement magnitude.

### Study 5: Hard cap vs soft penalty vs metric-only (Tier 3 addition)
500 laundering-prone profiles (seed 43): all required families present, q ~ Normal(90, 8) clipped [70, 100], 80–90% weak-provenance modes (Synthetic Fixture, Declared Supplier Evidence, Workspace Asset, External Provider unverified), 10–20% strong modes (Edge Local, Uploaded Sample). Compares all 9 variants. The key question: does soft_penalty_only (full G, no hard cap) still over-promote relative to the full framework?

---

## Baselines

| Variant | Description |
|---------|-------------|
| `full_framework` | Missingness synthesis + input-mode ceilings + hard cap + composite G |
| `metric_only` | band(mean observed q); ignores provenance, context, missingness |
| `soft_penalty_only` | Full G with missingness synthesis; no hard cap. ClaimLevel = band(100·G) |
| `completeness_only_checklist` | ClaimLevel = band(100·C_comp); ignores provenance and q |
| `composite_without_cap` | Full G, missingness synthesis, no ceiling meet. Equivalent to soft_penalty_only |
| `evidence_present` | A if all required families have observed evidence; E otherwise |
| `supplier_attestation` | Coverage × avg_q; DSE verified=1 explicitly acceptable; no ceilings |
| `assurance_case_style` | Stylized: coverage × (avg_q/100); no mode ceilings; E only if all absent |
| `control_presence` | Fraction of required families present → threshold-based level |

**Note:** `soft_penalty_only` and `composite_without_cap` produce identical results by design — they are included separately to name two different rhetorical framings of the same relaxation.

---

## Interpreting validation_results.json

```json
{
  "metadata": { ... },          // script name, timestamp, python/numpy versions, seeds
  "framework": { ... },         // all constants: ceilings, quality scores, necessity matrix
  "generators": { ... },        // generator assumptions serialised
  "baselines": { ... },         // description of each variant
  "studies": {
    "uniform": {                 // results for uniform generator
      "study_1_sensitivity_decomposition": { ... },
      "study_2_invariant_testing": { ... },
      "study_3_laundering_pathology": { ... },
      "study_4_ablation": { ... },
      "study_5_hardcap_vs_softpenalty": { ... }
    },
    "context_correlated": { ... } // same structure for correlated generator
  },
  "manuscript_tables": { ... }   // ready-to-cite table rows for each study
}
```

Seeds are documented in `metadata.seeds`. All numbers are generated by the script; none are hardcoded.

---

## How to regenerate manuscript tables

```bash
python provena_validation.py --all --deterministic-timestamp --out validation_results.json
```

`manuscript_tables` in the JSON contains row-level data for each table. `GENERATED_RESULTS_AUDIT.md` lists every manuscript claim with its generated value and a recommended replacement sentence.

---

## Anonymization

This repository is double-blind safe. It contains no author names, organization names, institutional affiliations, grant numbers, private emails, internal product names other than "Provena", or local filesystem paths. Run `python scripts/check_anonymization.py` to verify.

---

## Limitations of synthetic validation

The computational studies establish that the implementation satisfies its formal specification under synthetic audit profiles. They do not establish:

- That the necessity matrix is calibrated against practitioner judgment (identified as future work in the manuscript).
- That the band thresholds are calibrated against expert-adjudicated audit vignettes.
- That the fifteen evidence families are the families practitioners would identify.
- Any claim about external validity in real deployments.

The framework's structural guarantees hold for any consistent choice of necessity matrix, band thresholds, and family taxonomy. Calibration of these specific instantiations is identified as future work.
