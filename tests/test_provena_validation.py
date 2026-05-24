"""
Tests for provena_validation.py
Anonymous companion repository — pytest test suite.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provena_validation import (
    LEVELS,
    LEVEL_INT,
    NECESSITY,
    CC_MODE_PROBS,
    CC_VERIFY_PROBS,
    CC_REPRO_PROBS,
    AuditProfile,
    Finding,
    band,
    ceiling,
    meet,
    required_families,
    recommended_families,
    augment_missing_findings,
    finding_level,
    compute_C_comp,
    compute_C_iq,
    compute_C_rep,
    compute_G,
    weakest_critical_ceiling,
    compute_claim_level_full,
    compute_variant,
    generate_uniform_profiles,
    generate_context_correlated_profiles,
    overstatement_magnitude,
    level_distribution,
    _compute_soft_penalty_only,
    _compute_full_framework,
    DEFAULT_WEIGHTS,
    INPUT_MODES,
)


# ---------------------------------------------------------------------------
# 1. Band function boundaries
# ---------------------------------------------------------------------------


class TestBandFunction:
    def test_100_is_A(self):
        assert band(100) == "A"

    def test_85_is_A(self):
        assert band(85) == "A"

    def test_84_999_is_B(self):
        assert band(84.999) == "B"

    def test_70_is_B(self):
        assert band(70) == "B"

    def test_69_999_is_C(self):
        assert band(69.999) == "C"

    def test_50_is_C(self):
        assert band(50) == "C"

    def test_49_999_is_D(self):
        assert band(49.999) == "D"

    def test_30_is_D(self):
        assert band(30) == "D"

    def test_29_999_is_E(self):
        assert band(29.999) == "E"

    def test_0_is_E(self):
        assert band(0) == "E"


# ---------------------------------------------------------------------------
# 2. Ceiling function
# ---------------------------------------------------------------------------


class TestCeilingFunction:
    def test_synthetic_fixture_unverified_is_C(self):
        assert ceiling("Synthetic Fixture", 0) == "C"

    def test_synthetic_fixture_verified_is_C(self):
        assert ceiling("Synthetic Fixture", 1) == "C"

    def test_declared_supplier_unverified_is_D(self):
        assert ceiling("Declared Supplier Evidence", 0) == "D"

    def test_declared_supplier_verified_is_D(self):
        assert ceiling("Declared Supplier Evidence", 1) == "D"

    def test_not_assessable_unverified_is_E(self):
        assert ceiling("Not Assessable", 0) == "E"

    def test_not_assessable_verified_is_E(self):
        assert ceiling("Not Assessable", 1) == "E"

    def test_federated_unverified_is_B(self):
        assert ceiling("Federated", 0) == "B"

    def test_federated_verified_is_A(self):
        assert ceiling("Federated", 1) == "A"

    def test_workspace_asset_unverified_is_C(self):
        assert ceiling("Workspace Asset", 0) == "C"

    def test_workspace_asset_verified_is_B(self):
        assert ceiling("Workspace Asset", 1) == "B"

    def test_external_provider_unverified_is_C(self):
        assert ceiling("External Provider", 0) == "C"

    def test_external_provider_verified_is_B(self):
        assert ceiling("External Provider", 1) == "B"

    def test_edge_local_both_are_A(self):
        assert ceiling("Edge Local", 0) == "A"
        assert ceiling("Edge Local", 1) == "A"

    def test_uploaded_sample_both_are_B(self):
        assert ceiling("Uploaded Sample", 0) == "B"
        assert ceiling("Uploaded Sample", 1) == "B"


# ---------------------------------------------------------------------------
# 3. Necessity matrix
# ---------------------------------------------------------------------------


class TestNecessityMatrix:
    def test_all_10_contexts_present(self):
        assert len(NECESSITY) == 10

    def test_all_15_families_per_row(self):
        for row in NECESSITY:
            assert len(row) == 15

    def test_values_only_in_0_1_2(self):
        for row in NECESSITY:
            for v in row:
                assert v in {0, 1, 2}

    def test_required_families_returns_set(self):
        for ctx in range(1, 11):
            req = required_families(ctx)
            assert isinstance(req, set)
            assert all(1 <= f <= 15 for f in req)

    def test_c3_requires_F6_explainability(self):
        # C3 HR/workforce: row[5] = NECESSITY[2][5] = 2
        assert 6 in required_families(3)

    def test_c2_requires_all_key_families(self):
        # Financial services: F1-F10, F12-F14 all required
        req = required_families(2)
        for f in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]:
            assert f in req

    def test_c9_required_includes_F8_F10_F12_F14(self):
        req = required_families(9)
        for f in [8, 10, 12, 14]:
            assert f in req


# ---------------------------------------------------------------------------
# 4. Context-correlated probability distributions
# ---------------------------------------------------------------------------


class TestContextCorrelatedDistributions:
    def test_mode_probs_sum_to_1_for_each_context(self):
        for ctx, probs in CC_MODE_PROBS.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Context C{ctx} mode probs sum to {total}, not 1.0"
            )

    def test_all_10_contexts_have_distributions(self):
        assert set(CC_MODE_PROBS.keys()) == set(range(1, 11))

    def test_no_negative_probabilities(self):
        for ctx, probs in CC_MODE_PROBS.items():
            for mode, p in probs.items():
                assert p >= 0, f"C{ctx} mode {mode} has negative probability {p}"

    def test_verify_probs_in_unit_interval(self):
        for mode, p in CC_VERIFY_PROBS.items():
            assert 0.0 <= p <= 1.0, f"Verify prob for {mode} out of range: {p}"

    def test_repro_probs_in_unit_interval(self):
        for mode, p in CC_REPRO_PROBS.items():
            assert 0.0 <= p <= 1.0, f"Repro prob for {mode} out of range: {p}"

    def test_c9_has_high_declared_supplier_evidence(self):
        # Vendor context: DSE should dominate
        assert CC_MODE_PROBS[9].get("Declared Supplier Evidence", 0) >= 0.30

    def test_all_modes_in_verify_probs(self):
        assessable = [m for m in INPUT_MODES if m != "Not Assessable"]
        for mode in assessable:
            assert mode in CC_VERIFY_PROBS, f"Missing verify prob for {mode}"


# ---------------------------------------------------------------------------
# 5. Missingness rule
# ---------------------------------------------------------------------------


class TestMissingnessRule:
    def _make_profile_without_F6(self) -> AuditProfile:
        # C3 requires F6 (Explainability)
        findings = [
            Finding(family=1, mode="Edge Local", verified=1, q=90, reproducible=True),
            Finding(family=2, mode="Edge Local", verified=1, q=85, reproducible=True),
            Finding(family=4, mode="Edge Local", verified=1, q=80, reproducible=True),
            Finding(family=5, mode="Edge Local", verified=1, q=88, reproducible=True),
            Finding(family=7, mode="Edge Local", verified=1, q=92, reproducible=True),
        ]
        return AuditProfile(profile_id="test_c3", context=3, findings=findings)

    def test_synthesized_F6_not_assessable_appears_in_augmented(self):
        profile = self._make_profile_without_F6()
        augmented = augment_missing_findings(profile)
        f6 = [f for f in augmented if f.family == 6]
        assert len(f6) == 1
        assert f6[0].mode == "Not Assessable"
        assert f6[0].is_missing_synthetic is True

    def test_missing_required_family_sets_claim_level_to_E(self):
        profile = self._make_profile_without_F6()
        lv = compute_claim_level_full(profile)
        assert lv == "E", (
            f"Expected E when required F6 is absent in C3; got {lv}"
        )

    def test_no_missing_required_families_does_not_synthesize(self):
        # Profile with all required families for C9 (has fewer required)
        req = sorted(required_families(9))
        findings = [
            Finding(family=f, mode="Edge Local", verified=1, q=90, reproducible=True)
            for f in req
        ]
        profile = AuditProfile(profile_id="test_c9_full", context=9, findings=findings)
        augmented = augment_missing_findings(profile)
        synthetic = [f for f in augmented if f.is_missing_synthetic]
        assert len(synthetic) == 0


# ---------------------------------------------------------------------------
# 6. Anti-laundering invariant
# ---------------------------------------------------------------------------


class TestAntiLaundering:
    def test_synthetic_fixture_q100_cannot_exceed_C(self):
        f = Finding(
            family=1, mode="Synthetic Fixture", verified=0, q=100.0, reproducible=True
        )
        lv = finding_level(f)
        assert LEVEL_INT[lv] <= LEVEL_INT["C"], (
            f"Synthetic Fixture q=100 yielded {lv}, expected <= C"
        )

    def test_declared_supplier_q100_cannot_exceed_D(self):
        f = Finding(
            family=1, mode="Declared Supplier Evidence", verified=0, q=100.0, reproducible=True
        )
        lv = finding_level(f)
        assert LEVEL_INT[lv] <= LEVEL_INT["D"], (
            f"Declared Supplier Evidence q=100 yielded {lv}, expected <= D"
        )

    def test_declared_supplier_verified_q100_cannot_exceed_D(self):
        f = Finding(
            family=1, mode="Declared Supplier Evidence", verified=1, q=100.0, reproducible=True
        )
        lv = finding_level(f)
        assert LEVEL_INT[lv] <= LEVEL_INT["D"]

    def test_synthetic_fixture_verified_q100_cannot_exceed_C(self):
        f = Finding(
            family=1, mode="Synthetic Fixture", verified=1, q=100.0, reproducible=True
        )
        lv = finding_level(f)
        assert LEVEL_INT[lv] <= LEVEL_INT["C"]


# ---------------------------------------------------------------------------
# 7. Hard cap
# ---------------------------------------------------------------------------


class TestHardCap:
    def test_high_G_capped_at_weakest_critical_ceiling(self):
        # Profile where G should band to A, but a required family is SF (ceiling C)
        # Use C3 (HR), provide all required families but one as SF
        req = sorted(required_families(3))
        findings = []
        for i, fam in enumerate(req):
            if i == 0:
                # Force this to Synthetic Fixture with high q
                findings.append(
                    Finding(family=fam, mode="Synthetic Fixture", verified=0,
                            q=100.0, reproducible=True)
                )
            else:
                findings.append(
                    Finding(family=fam, mode="Edge Local", verified=1,
                            q=100.0, reproducible=True)
                )
        profile = AuditProfile(profile_id="hardcap_test", context=3, findings=findings)
        lv = compute_claim_level_full(profile)
        # G should be very high -> band A; weakest ceiling = C (from Synthetic Fixture)
        aug = augment_missing_findings(profile)
        g = compute_G(aug, 3, DEFAULT_WEIGHTS)
        assert band(100 * g) == "A", f"Expected pre-cap band A; got {band(100*g)}"
        assert lv == "C", f"Expected ClaimLevel C (hard cap from SF); got {lv}"

    def test_hard_cap_meet_b_and_c_gives_c(self):
        assert meet("B", "C") == "C"

    def test_hard_cap_meet_a_and_e_gives_e(self):
        assert meet("A", "E") == "E"


# ---------------------------------------------------------------------------
# 8. Soft penalty baseline
# ---------------------------------------------------------------------------


class TestSoftPenaltyBaseline:
    def _make_sf_profile(self) -> AuditProfile:
        req = sorted(required_families(3))
        findings = [
            Finding(family=f, mode="Synthetic Fixture", verified=0, q=100.0,
                    reproducible=True)
            for f in req
        ]
        return AuditProfile(profile_id="sf_test", context=3, findings=findings)

    def test_soft_penalty_does_not_apply_hard_cap(self):
        profile = self._make_sf_profile()
        # Full framework should cap at C (SF ceiling)
        full_lv = _compute_full_framework(profile)
        soft_lv = _compute_soft_penalty_only(profile)
        # soft_penalty should not be worse than full_framework in general
        # (it ignores ceiling so may be HIGHER)
        assert LEVEL_INT[soft_lv] >= LEVEL_INT[full_lv], (
            f"Soft penalty {soft_lv} should be >= full {full_lv} on laundering profiles"
        )

    def test_soft_penalty_can_exceed_weak_critical_ceiling(self):
        # Profile with all SF findings (ceiling C) and high q
        profile = self._make_sf_profile()
        full_lv = _compute_full_framework(profile)
        soft_lv = _compute_soft_penalty_only(profile)
        # Full should be at most C; soft may be higher
        assert LEVEL_INT[full_lv] <= LEVEL_INT["C"]
        # soft penalty might be higher because it ignores the ceiling
        # (it uses G which may band to B or A if q=100 and coverage=1)
        aug = augment_missing_findings(profile)
        g = compute_G(aug, 3, DEFAULT_WEIGHTS)
        assert soft_lv == band(100.0 * g)


# ---------------------------------------------------------------------------
# 9. Study 5 profile properties
# ---------------------------------------------------------------------------


class TestStudy5:
    def test_laundering_prone_profiles_have_high_q(self):
        from provena_validation import _build_study5_profile
        rng = np.random.default_rng(43)
        ctx = 1
        p = _build_study5_profile(0, ctx, rng)
        for f in p.findings:
            assert f.q >= 70.0, f"Finding q={f.q} < 70 in Study 5 profile"

    def test_laundering_prone_profiles_have_all_required_families(self):
        from provena_validation import _build_study5_profile
        rng = np.random.default_rng(43)
        for ctx in range(1, 11):
            p = _build_study5_profile(0, ctx, rng)
            req = required_families(ctx)
            obs_fams = {f.family for f in p.findings}
            assert req.issubset(obs_fams), (
                f"C{ctx}: missing required families {req - obs_fams}"
            )

    def test_study5_dominant_weak_provenance(self):
        from provena_validation import _build_study5_profile
        weak_modes = {
            "Synthetic Fixture", "Declared Supplier Evidence",
            "Workspace Asset", "External Provider"
        }
        rng = np.random.default_rng(43)
        weak_count = 0
        total = 0
        for i in range(50):
            ctx = int(rng.integers(1, 11))
            p = _build_study5_profile(i, ctx, rng)
            for f in p.findings:
                total += 1
                if f.mode in weak_modes:
                    weak_count += 1
        # Should be 80-90% weak modes
        assert weak_count / total >= 0.75, (
            f"Weak-mode fraction {weak_count/total:.2f} < 0.75"
        )

    def test_study5_variants_produce_serialisable_output(self):
        import json
        from provena_validation import run_study5_hardcap, _NumpyEncoder
        result = run_study5_hardcap(n=20, seed=43)
        serialized = json.dumps(result, cls=_NumpyEncoder)
        parsed = json.loads(serialized)
        assert "variants" in parsed
        for v in ["full_framework", "soft_penalty_only", "metric_only"]:
            assert v in parsed["variants"]


# ---------------------------------------------------------------------------
# 10. Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_uniform_generator_same_seed_identical_profiles(self):
        p1 = generate_uniform_profiles(n=50, seed=99)
        p2 = generate_uniform_profiles(n=50, seed=99)
        for a, b in zip(p1, p2):
            assert a.context == b.context
            assert len(a.findings) == len(b.findings)
            for fa, fb in zip(a.findings, b.findings):
                assert fa.family == fb.family
                assert fa.mode == fb.mode
                assert fa.verified == fb.verified
                assert abs(fa.q - fb.q) < 1e-12

    def test_cc_generator_same_seed_identical_profiles(self):
        p1 = generate_context_correlated_profiles(n=50, seed=101)
        p2 = generate_context_correlated_profiles(n=50, seed=101)
        for a, b in zip(p1, p2):
            assert a.context == b.context
            assert len(a.findings) == len(b.findings)

    def test_different_seeds_differ(self):
        p1 = generate_uniform_profiles(n=50, seed=99)
        p2 = generate_uniform_profiles(n=50, seed=100)
        contexts_same = sum(
            1 for a, b in zip(p1, p2) if a.context == b.context
        )
        # Should differ substantially
        assert contexts_same < 45, "Different seeds produced nearly identical profiles"


# ---------------------------------------------------------------------------
# 11. Anonymization check sanity
# ---------------------------------------------------------------------------


class TestAnonymizationScript:
    def test_check_script_exists(self):
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "check_anonymization.py"
        )
        assert os.path.exists(script), "check_anonymization.py not found"

    def test_check_script_passes_on_clean_file(self, tmp_path):
        import subprocess
        clean_file = tmp_path / "clean.py"
        clean_file.write_text('# anonymous code\nresult = 42\n')
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "check_anonymization.py"
        )
        ret = subprocess.run(
            [sys.executable, script, str(tmp_path)],
            capture_output=True,
        )
        assert ret.returncode == 0, f"Expected clean pass; got: {ret.stdout}{ret.stderr}"

    def test_check_script_fails_on_banned_token(self, tmp_path):
        import subprocess
        dirty_file = tmp_path / "dirty.py"
        dirty_file.write_text('# AffectLog internal tool\n')
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "check_anonymization.py"
        )
        ret = subprocess.run(
            [sys.executable, script, str(tmp_path)],
            capture_output=True,
        )
        assert ret.returncode != 0, "Expected failure on banned token 'AffectLog'"


# ---------------------------------------------------------------------------
# Additional: overstatement magnitude
# ---------------------------------------------------------------------------


class TestOverstatementMagnitude:
    def test_same_level_is_zero(self):
        assert overstatement_magnitude("C", "C") == 0

    def test_A_over_D_is_3(self):
        assert overstatement_magnitude("A", "D") == 3

    def test_E_under_A_is_negative(self):
        assert overstatement_magnitude("E", "A") == -4


# ---------------------------------------------------------------------------
# Additional: level_distribution
# ---------------------------------------------------------------------------


class TestLevelDistribution:
    def test_counts_all_levels(self):
        levels = ["A", "B", "C", "A", "E", "D", "B"]
        dist = level_distribution(levels)
        assert dist == {"A": 2, "B": 2, "C": 1, "D": 1, "E": 1}

    def test_empty_list(self):
        dist = level_distribution([])
        assert all(v == 0 for v in dist.values())
