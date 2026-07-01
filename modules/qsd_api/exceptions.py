from __future__ import annotations


class TranscriptionCancelled(RuntimeError):
    """Raised when cooperative QSD task cancellation is requested."""
