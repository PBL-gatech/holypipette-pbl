
from __future__ import absolute_import

from typing import TYPE_CHECKING

from .pipetteDetector import PipetteDetector, PipetteDetector1, PipetteDetector2
from .pipetteFocuser import PipetteFocuser

__all__ = [
    "PipetteDetector",
    "PipetteDetector1",
    "PipetteDetector2",
    "PipetteFocuser",
    "CellSegmentor2",
    "CellSegmentor3",
]

if TYPE_CHECKING:
    from .cellSegmentor import CellSegmentor2, CellSegmentor3  # pragma: no cover


def __getattr__(name: str):
    if name in {"CellSegmentor2", "CellSegmentor3"}:
        from .cellSegmentor import CellSegmentor2, CellSegmentor3
        globals().update({
            "CellSegmentor2": CellSegmentor2,
            "CellSegmentor3": CellSegmentor3,
        })
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
