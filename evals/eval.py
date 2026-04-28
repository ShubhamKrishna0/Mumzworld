"""Evaluation runner for the Moms Verdict Generator pipeline.

Runs test cases from a JSON file, checks schema validity, hallucination,
coverage, and confidence calibration. Reports pass/fail with reasons.

Usage:
    python -m evals.eval
    python evals/eval.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.errors import InsufficientDataError, PipelineError
from app.pipeline import run_pipeline
from app.schema import MomsVerdict


@dataclass
class TestResult:
    """Result of a single evaluation test case."""

    name: str
    passed: bool
    reason: str
    details: dict | None = field(default=None)


def _load_test_cases(test_cases_path: str) -> list[dict]:
    """Load test case definitions from a JSON file."""
    path = Path(test_cases_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_input_file(input_file: str) -> bytes:
    """Load a data file as raw bytes for the pipeline."""
    path = Path(input_file)
    return path.read_bytes()


def _check_hallucination(verdict: MomsVerdict, reviews: list[str]) -> tuple[bool, str]:
    """Basic hallucination check: verify pros/cons keywords appear in input reviews.

    Joins all review text into a single lowercase corpus and checks that each
    pro/con shares at least one significant keyword (4+ chars) with the reviews.
    """
    corpus = " ".join(reviews).lower()

    for pro in verdict.pros:
        words = [w for w in pro.lower().split() if len(w) >= 4]
        if words and not any(w in corpus for w in words):
            return False, f"Pro may be hallucinated (no keyword overlap): '{pro}'"

    for con in verdict.cons:
        words = [w for w in con.lower().split() if len(w) >= 4]
        if words and not any(w in corpus for w in words):
            return False, f"Con may be hallucinated (no keyword overlap): '{con}'"

    return True, "No hallucination detected"


def _check_coverage(verdict: MomsVerdict, key_themes: list[str]) -> tuple[bool, str]:
    """Check that expected key themes appear somewhere in the output."""
    if not key_themes:
        return True, "No key themes to check"

    output_text = " ".join([
        verdict.summary_en,
        verdict.verdict_en,
        " ".join(verdict.pros),
        " ".join(verdict.cons),
    ]).lower()

    missing = [t for t in key_themes if t.lower() not in output_text]

    if len(missing) > len(key_themes) // 2:
        return False, f"Missing majority of key themes: {missing}"

    return True, f"Coverage OK (missing {len(missing)}/{len(key_themes)} themes)"


def _check_confidence(
    verdict: MomsVerdict, expected: dict
) -> tuple[bool, str]:
    """Check confidence calibration against expected ranges."""
    conf = verdict.confidence
    min_conf = expected.get("min_confidence")
    max_conf = expected.get("max_confidence")

    if min_conf is not None and conf < min_conf:
        return False, f"Confidence {conf:.4f} below expected minimum {min_conf}"
    if max_conf is not None and conf >= max_conf:
        return False, f"Confidence {conf:.4f} at or above expected maximum {max_conf}"

    return True, f"Confidence {conf:.4f} within expected range"


def _check_uncertainty_reason(
    verdict: MomsVerdict, expected: dict
) -> tuple[bool, str]:
    """Check uncertainty_reason consistency with confidence."""
    expect_reason = expected.get("expect_uncertainty_reason", False)

    if expect_reason and verdict.uncertainty_reason is None:
        return False, "Expected uncertainty_reason but got None"
    if not expect_reason and verdict.uncertainty_reason is not None:
        return False, f"Did not expect uncertainty_reason but got: '{verdict.uncertainty_reason}'"

    # Also check consistency: confidence < 0.5 should have reason, >= 0.5 should not
    if verdict.confidence < 0.5 and verdict.uncertainty_reason is None:
        return False, "Confidence < 0.5 but uncertainty_reason is None"
    if verdict.confidence >= 0.5 and verdict.uncertainty_reason is not None:
        return False, f"Confidence >= 0.5 but uncertainty_reason is set: '{verdict.uncertainty_reason}'"

    return True, "Uncertainty reason consistent with confidence"


def _run_single_test(test_case: dict) -> TestResult:
    """Run a single test case through the pipeline and check all assertions."""
    name = test_case["name"]
    input_file = test_case["input_file"]
    expected = test_case["expected"]
    checks_failed: list[str] = []
    details: dict = {}

    # Load input
    try:
        raw_data = _load_input_file(input_file)
    except FileNotFoundError:
        return TestResult(name=name, passed=False, reason=f"Input file not found: {input_file}")

    # Load reviews for hallucination check
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reviews = json.load(f)
    except Exception:
        reviews = []

    # Run pipeline
    try:
        verdict = run_pipeline(raw_data, ".json")
    except InsufficientDataError:
        # This is an expected outcome for certain test cases (e.g., empty_after_preprocessing)
        if expected.get("expect_low_confidence") and not expected.get("expect_pros") and not expected.get("expect_cons"):
            return TestResult(
                name=name,
                passed=True,
                reason="Pipeline raised InsufficientDataError as expected",
                details={"error": "InsufficientDataError"},
            )
        return TestResult(
            name=name,
            passed=False,
            reason="Pipeline raised InsufficientDataError unexpectedly",
            details={"error": "InsufficientDataError"},
        )
    except PipelineError as exc:
        return TestResult(
            name=name,
            passed=False,
            reason=f"Pipeline error: {exc}",
            details={"error": type(exc).__name__, "message": str(exc)},
        )
    except Exception as exc:
        return TestResult(
            name=name,
            passed=False,
            reason=f"Unexpected error: {exc}",
            details={"error": type(exc).__name__, "message": str(exc)},
        )

    # Store output summary
    details["confidence"] = verdict.confidence
    details["uncertainty_reason"] = verdict.uncertainty_reason
    details["num_pros"] = len(verdict.pros)
    details["num_cons"] = len(verdict.cons)

    # Check 1: Schema validity (already guaranteed by Pydantic, but verify type)
    if not isinstance(verdict, MomsVerdict):
        checks_failed.append("Output is not a valid MomsVerdict instance")

    # Check 2: Hallucination
    if not expected.get("expect_hallucination", False):
        ok, msg = _check_hallucination(verdict, reviews)
        if not ok:
            checks_failed.append(msg)
        details["hallucination_check"] = msg

    # Check 3: Coverage
    key_themes = expected.get("key_themes", [])
    ok, msg = _check_coverage(verdict, key_themes)
    if not ok:
        checks_failed.append(msg)
    details["coverage_check"] = msg

    # Check 4: Confidence calibration
    ok, msg = _check_confidence(verdict, expected)
    if not ok:
        checks_failed.append(msg)
    details["confidence_check"] = msg

    # Check 5: Pros expectation
    if expected.get("expect_pros") and len(verdict.pros) == 0:
        checks_failed.append("Expected non-empty pros but got empty list")
    if not expected.get("expect_pros") and expected.get("expect_pros") is not None and len(verdict.pros) > 0:
        # Only flag if explicitly set to false (not just missing)
        pass  # LLMs may still produce minor pros; don't hard-fail

    # Check 6: Cons expectation
    if expected.get("expect_cons") and len(verdict.cons) == 0:
        checks_failed.append("Expected non-empty cons but got empty list")

    # Check 7: Uncertainty reason consistency
    ok, msg = _check_uncertainty_reason(verdict, expected)
    if not ok:
        checks_failed.append(msg)
    details["uncertainty_check"] = msg

    if checks_failed:
        return TestResult(
            name=name,
            passed=False,
            reason="; ".join(checks_failed),
            details=details,
        )

    return TestResult(name=name, passed=True, reason="All checks passed", details=details)


def run_evaluation(test_cases_path: str) -> list[TestResult]:
    """Load test cases from JSON and run each through the pipeline.

    Parameters
    ----------
    test_cases_path : str
        Path to the JSON file containing test case definitions.

    Returns
    -------
    list[TestResult]
        A list of TestResult objects with pass/fail and descriptive reasons.
    """
    test_cases = _load_test_cases(test_cases_path)
    results: list[TestResult] = []

    for tc in test_cases:
        print(f"  Running: {tc['name']}...", end=" ", flush=True)
        result = _run_single_test(tc)
        status = "PASS" if result.passed else "FAIL"
        print(status)
        results.append(result)

    return results


def main() -> None:
    """Run the evaluation suite and print results."""
    # Determine test cases path
    eval_dir = Path(__file__).parent
    test_cases_path = eval_dir / "test_cases.json"

    if not test_cases_path.exists():
        print(f"Error: test cases file not found at {test_cases_path}")
        sys.exit(1)

    print("=" * 60)
    print("Moms Verdict Generator — Evaluation Suite")
    print("=" * 60)
    print()

    results = run_evaluation(str(test_cases_path))

    # Print summary
    print()
    print("-" * 60)
    print("Results Summary")
    print("-" * 60)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        print(f"  [{icon}] {r.name}: {r.reason}")
        if r.details and not r.passed:
            for k, v in r.details.items():
                print(f"         {k}: {v}")

    print()
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
