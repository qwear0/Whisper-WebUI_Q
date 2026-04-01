import os
from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def allow_trusted_checkpoint_loading() -> Iterator[None]:
    """
    Temporarily restore pre-Torch-2.6 checkpoint loading behavior.

    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True`` when the callsite
    does not explicitly pass ``weights_only``. Pyannote/Lightning still load full
    trusted checkpoints during diarization model initialization, so we temporarily
    force the legacy behavior for that narrow code path and then restore the
    previous environment.
    """

    previous_force_weights_only: Optional[str] = os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    previous_force_no_weights_only: Optional[str] = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    try:
        yield
    finally:
        if previous_force_no_weights_only is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_force_no_weights_only

        if previous_force_weights_only is not None:
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = previous_force_weights_only
