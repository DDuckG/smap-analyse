class SmapError(Exception):
    """Base exception for recoverable application errors."""


class ValidationFailure(SmapError):
    """Raised when one or more UAP records fail validation."""


class OntologyError(SmapError):
    """Raised when ontology configuration is invalid."""


class PipelineError(SmapError):
    """Raised when a pipeline stage cannot complete."""


class DatabaseNotInitializedError(SmapError):
    """Raised when the metadata database is missing or behind migrations."""


class ReviewValidationError(SmapError):
    """Raised when a review action or reviewed overlay contribution is invalid."""


class ReviewConflictError(SmapError):
    """Raised when a review action conflicts with existing reviewed knowledge."""
