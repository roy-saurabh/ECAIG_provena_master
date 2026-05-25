#!/usr/bin/env python3
"""
sample_audit.py — Worked-example demonstrator for the Provena framework.

Reproduces the two worked numerical examples from Section 11 of the companion
manuscript end-to-end and emits a structured JSON audit report.

  Example 11.1  HR resume screening (C3)
                Illustrates missingness synthesis and hard-cap enforcement
                when a required family (F6 Explainability) is absent.
                Expected: ClaimLevel = E

  Example 11.2  Supplier-led vendor AI (C9)
                Illustrates weak-provenance capping when all required families
                are covered by Declared Supplier Evidence.
                Expected: ClaimLevel = D

Output: sample_audit_report.json  (or --out <path>)

Usage:
    python sample_audit.py
    python sample_audit.py --out my_report.json

Cross-references:
    Manuscript Section 11.1, Table 6 (HR example)
    Manuscript Section 11.2, Table 7 (Vendor AI example)
    Manuscript Appendix E (audit-trace schema)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from provena_validation import (
    DEFAULT_WEIGHTS,
    CONTEXT_NAMES,
    FAMILY_NAMES,
    AuditProfile,
    Finding,
    augment_missing_findings,
    band,
    ceiling,
    compute_C_comp,
    compute_C_iq,
    compute_C_rep,
    compute_G,
    finding_level,
    meet,
    required_families,
    weakest_critical_ceiling,
)

# ---------------------------------------------------------------------------
# Worked-example profile definitions
# (numbers copied verbatim from manuscript Tables 6 and 7)
# ---------------------------------------------------------------------------

# Section 11.1 — HR resume screening, context C3
# Ten observed findings; F6 (Explainability) is required but absent.
EXAMPLE_11_1 = AuditProfile(
    profile_id="worked_example_11_1_HR_C3",
    context=3,
    findings=[
        # f     mode               ver  q     reprod  is_synth  limitation
        Finding(1,  "Edge Local",          1, 88.0, True),   # F1  Data quality
        Finding(2,  "Edge Local",          1, 92.0, True),   # F2  Model performance
        Finding(4,  "Edge Local",          1, 84.0, True),   # F4  Reproducibility
        Finding(5,  "Edge Local",          1, 86.0, True),   # F5  Fairness
        Finding(7,  "Edge Local",          1, 90.0, True),   # F7  Privacy
        Finding(8,  "Workspace Asset",     0, 70.0, False),  # F8  Lineage
        Finding(10, "Synthetic Fixture",   0, 100.0, False), # F10 Security
        Finding(12, "Workspace Asset",     0, 65.0, False),  # F12 Policy decision
        Finding(13, "Edge Local",          1, 80.0, True),   # F13 Platform observability
        Finding(14, "Edge Local",          1, 95.0, True),   # F14 Export integrity
        # F6 (Explainability) intentionally absent → synthesized by framework
    ],
)

# Section 11.2 — Supplier-led vendor AI, context C9
# Four required families all present with Declared Supplier Evidence; no
# missingness synthesis required.  Required families for C9: F8, F10, F12, F14.
EXAMPLE_11_2 = AuditProfile(
    profile_id="worked_example_11_2_vendor_C9",
    context=9,
    findings=[
        Finding(8,  "Declared Supplier Evidence", 0, 95.0, False),  # F8  Lineage
        Finding(10, "Declared Supplier Evidence", 0, 92.0, False),  # F10 Security
        Finding(12, "Declared Supplier Evidence", 0, 90.0, False),  # F12 Policy decision
        Finding(14, "Declared Supplier Evidence", 0, 98.0, False),  # F14 Export integrity
    ],
)


# ---------------------------------------------------------------------------
# Core computation: step-by-step, recording all intermediate values
# ---------------------------------------------------------------------------

def run_worked_example(
    profile: AuditProfile,
    weights: tuple = DEFAULT_WEIGHTS,
) -> Dict[str, Any]:
    """Reproduce one worked example and return a structured report dict."""
    ctx = profile.context
    req = required_families(ctx)

    # Step 1 — missingness synthesis
    augmented = augment_missing_findings(profile)
    observed_fams = {f.family for f in profile.findings}
    missing_req_fams = sorted(req - observed_fams)

    # Step 2 — composite score
    c_comp = compute_C_comp(augmented, ctx)
    c_iq   = compute_C_iq(augmented)
    c_rep  = compute_C_rep(augmented)
    g      = compute_G(augmented, ctx, weights)
    pre_cap_band = band(100.0 * g)

    # Step 3 — hard cap
    weakest_ceil = weakest_critical_ceiling(augmented, ctx)
    claim_level  = meet(pre_cap_band, weakest_ceil)

    # Per-finding detail (schema: Appendix E "Finding")
    findings_out: List[Dict[str, Any]] = []
    for f in augmented:
        ceil_val = "E" if f.is_missing_synthetic else ceiling(f.mode, f.verified)
        level    = finding_level(f)
        findings_out.append({
            "family": f"F{f.family}",
            "family_name": FAMILY_NAMES[f.family],
            "required": f.family in req,
            "input_mode": f.mode,
            "verified": bool(f.verified),
            "normalized_q": round(f.q, 4),
            "reproducible": f.reproducible,
            "ceiling": ceil_val,
            "per_finding_level": level,
            "is_missing_synthetic": f.is_missing_synthetic,
            "limitation": f.limitation or None,
        })

    # Limitation set: all required-family limitations (Section 13 rule)
    limitation_set = [
        d["limitation"]
        for d in findings_out
        if d["limitation"] and d["required"]
    ]

    return {
        "profile_id": profile.profile_id,
        "context": f"C{ctx}",
        "context_name": CONTEXT_NAMES[ctx],
        "required_families": [f"F{f}" for f in sorted(req)],
        "observed_families": [f"F{f}" for f in sorted(observed_fams)],
        "missing_required_families": [f"F{f}" for f in missing_req_fams],
        "n_findings_observed": len(profile.findings),
        "n_findings_augmented": len(augmented),
        "weights": {"C_comp": weights[0], "C_iq": weights[1], "C_rep": weights[2]},
        "step_1_missingness_synthesis": {
            "synthesized": [f"F{f}" for f in missing_req_fams],
            "note": (
                "Each absent required family synthesized as Not Assessable "
                "(mode=Not Assessable, verified=0, q=0, reproducible=False)."
            ),
        },
        "step_2_composite_score": {
            "C_comp": round(c_comp, 6),
            "C_comp_detail": f"{len(observed_fams & req)}/{len(req)}",
            "C_iq": round(c_iq, 6),
            "C_rep": round(c_rep, 6),
            "C_rep_detail": (
                f"{sum(1 for f in augmented if f.reproducible)}/{len(augmented)}"
            ),
            "G": round(g, 6),
            "G_times_100": round(100.0 * g, 4),
            "pre_cap_band": pre_cap_band,
        },
        "step_3_hard_cap": {
            "weakest_critical_ceiling": weakest_ceil,
            "pre_cap_band": pre_cap_band,
            "claim_level": claim_level,
            "formula": f"ClaimLevel = band(100·G) ∧ weakest_critical_ceiling"
                        f" = {pre_cap_band} ∧ {weakest_ceil} = {claim_level}",
        },
        "claim_level": claim_level,
        "findings": findings_out,
        "limitation_set": limitation_set,
    }


# ---------------------------------------------------------------------------
# Verification: assert computed values match manuscript
# ---------------------------------------------------------------------------

EXPECTED: Dict[str, Dict[str, Any]] = {
    "worked_example_11_1_HR_C3": {
        "claim_level": "E",
        "pre_cap_band": "B",
        "weakest_critical_ceiling": "E",
        # Manuscript Section 11.2 Step 2 values (rounded to 3 d.p.)
        "C_comp_approx": 0.909,
        "C_iq_approx": 0.755,
        "C_rep_approx": 0.636,
        "G_approx": 0.779,
    },
    "worked_example_11_2_vendor_C9": {
        "claim_level": "D",
        "pre_cap_band": "D",
        "weakest_critical_ceiling": "D",
        "C_comp_approx": 1.000,
        "C_iq_approx": 0.300,
        "C_rep_approx": 0.000,
        "G_approx": 0.470,
    },
}


def verify(report: Dict[str, Any]) -> List[str]:
    """Return a list of verification failures (empty = all pass)."""
    pid = report["profile_id"]
    exp = EXPECTED[pid]
    failures: List[str] = []

    s3 = report["step_3_hard_cap"]
    s2 = report["step_2_composite_score"]

    if s3["claim_level"] != exp["claim_level"]:
        failures.append(
            f"ClaimLevel: got {s3['claim_level']!r}, expected {exp['claim_level']!r}"
        )
    if s3["pre_cap_band"] != exp["pre_cap_band"]:
        failures.append(
            f"pre_cap_band: got {s3['pre_cap_band']!r}, expected {exp['pre_cap_band']!r}"
        )
    if s3["weakest_critical_ceiling"] != exp["weakest_critical_ceiling"]:
        failures.append(
            f"weakest_critical_ceiling: got {s3['weakest_critical_ceiling']!r}, "
            f"expected {exp['weakest_critical_ceiling']!r}"
        )
    for key in ("C_comp_approx", "C_iq_approx", "C_rep_approx", "G_approx"):
        field = key.replace("_approx", "")
        got = s2[field]
        want = exp[key]
        if abs(got - want) > 0.001:
            failures.append(f"{field}: got {got:.4f}, expected ≈{want:.3f}")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the Section 11 worked examples and write a JSON report."
    )
    parser.add_argument(
        "--out",
        default="sample_audit_report.json",
        metavar="PATH",
        help="Output file path (default: sample_audit_report.json)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip assertion checks against manuscript values.",
    )
    args = parser.parse_args()

    print("Provena — worked-example demonstrator", flush=True)
    print("=" * 52, flush=True)

    reports = []
    all_ok = True

    for profile in [EXAMPLE_11_1, EXAMPLE_11_2]:
        report = run_worked_example(profile)
        reports.append(report)

        pid = report["profile_id"]
        level = report["claim_level"]
        s2 = report["step_2_composite_score"]
        s3 = report["step_3_hard_cap"]

        print(f"\n{pid}")
        print(f"  Context : {report['context']} — {report['context_name']}")
        print(f"  Required: {', '.join(report['required_families'])}")
        print(f"  Missing : {', '.join(report['missing_required_families']) or 'none'}")
        print(
            f"  C_comp={s2['C_comp']:.4f}  C_iq={s2['C_iq']:.4f}"
            f"  C_rep={s2['C_rep']:.4f}  G={s2['G']:.4f}"
        )
        print(
            f"  band(100·G)={s3['pre_cap_band']}"
            f"  ∧  ceil_min={s3['weakest_critical_ceiling']}"
            f"  →  ClaimLevel={level}"
        )

        if not args.no_verify:
            failures = verify(report)
            if failures:
                all_ok = False
                for msg in failures:
                    print(f"  FAIL: {msg}", file=sys.stderr)
            else:
                print("  PASS: all values match manuscript Section 11 (±0.001)")

    output = {
        "generated_by": "sample_audit.py",
        "manuscript_sections": ["11.1", "11.2"],
        "description": (
            "End-to-end reproduction of the two worked examples in Section 11 "
            "of the companion manuscript.  Values match Tables 6–7 and the "
            "step-by-step computations in Sections 11.1–11.2."
        ),
        "reports": reports,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {args.out}")

    if not all_ok:
        print("ERROR: verification failures — see stderr.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
