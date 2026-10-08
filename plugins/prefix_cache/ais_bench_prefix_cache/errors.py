class PrefixCacheError(Exception):
    """Base user-facing plugin error."""


class ScenarioValidationError(PrefixCacheError):
    """Invalid scenario or source data."""


class ArtifactValidationError(PrefixCacheError):
    """Generated artifact is incomplete or inconsistent."""


class PromptRoundTripError(ArtifactValidationError):
    """Composed prompt tokens do not survive decode/re-encode."""


class RuntimeCapabilityError(PrefixCacheError):
    """The inference service cannot satisfy a required runtime capability."""
