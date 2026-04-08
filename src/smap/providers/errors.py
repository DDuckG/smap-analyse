from __future__ import annotations


class ProviderError(RuntimeError):
    """Base provider error."""


class ProviderUnavailableError(ProviderError):
    """Raised when an optional provider dependency is unavailable locally."""

