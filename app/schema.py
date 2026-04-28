"""MomsVerdict Pydantic v2 schema and validation utilities."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError, field_validator

from app.errors import ValidationError


class MomsVerdict(BaseModel):
    """Structured output schema for the Moms Verdict pipeline."""

    summary_en: str
    summary_ar: str
    verdict_en: str
    verdict_ar: str
    pros: list[str]
    cons: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_reason: str | None

    @field_validator("summary_en", "summary_ar", "verdict_en", "verdict_ar")
    @classmethod
    def no_empty_strings(cls, v: str) -> str:
        if v is not None and v.strip() == "":
            raise ValueError("Empty strings not allowed; use None")
        return v

    @field_validator("uncertainty_reason")
    @classmethod
    def uncertainty_not_empty(cls, v: str | None) -> str | None:
        if v is not None and v.strip() == "":
            raise ValueError("uncertainty_reason must be non-empty string or None")
        return v


def validate_and_retry(generate_fn, max_retries: int = 2) -> MomsVerdict:
    """Call generate_fn, validate with MomsVerdict. Retry up to max_retries on failure.

    Parameters
    ----------
    generate_fn : callable
        A callable that takes no arguments and returns a dict.
    max_retries : int
        Number of additional attempts after the first failure (total = 1 + max_retries).

    Returns
    -------
    MomsVerdict
        A validated MomsVerdict instance.

    Raises
    ------
    app.errors.ValidationError
        If validation fails after all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        data = generate_fn()
        try:
            return MomsVerdict.model_validate(data)
        except PydanticValidationError as exc:
            last_error = exc
            if attempt < max_retries:
                print(
                    f"[validate_and_retry] Attempt {attempt + 1} failed validation, "
                    f"retrying ({max_retries - attempt} retries left): {exc}"
                )
            else:
                print(
                    f"[validate_and_retry] Attempt {attempt + 1} failed validation, "
                    f"no retries left: {exc}"
                )

    raise ValidationError(
        f"Schema validation failed after {1 + max_retries} attempts. "
        f"Last error: {last_error}"
    )
