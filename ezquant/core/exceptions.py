class EZQuantError(Exception):
    """Base exception for EZQUANT."""


class PolicyViolationError(EZQuantError):
    """Raised when policy constraints are violated."""


class ExportBlockedError(EZQuantError):
    """Raised when export is blocked by QC gates."""


class MissingMetadataError(EZQuantError):
    """Raised when required metadata is absent."""
