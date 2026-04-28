"""Custom error classes for the Moms Verdict pipeline."""


class PipelineError(Exception):
    """Base class for pipeline errors."""


class EmptyInputError(PipelineError):
    """Raised when input contains no reviews."""


class ParseError(PipelineError):
    """Raised when input file cannot be parsed."""


class InsufficientDataError(PipelineError):
    """Raised when all reviews are discarded during preprocessing."""


class GenerationError(PipelineError):
    """Raised when LLM generation fails after retries."""


class ValidationError(PipelineError):
    """Raised when output fails schema validation after retries."""
