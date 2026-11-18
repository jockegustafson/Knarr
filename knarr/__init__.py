"""Knarr: creak (vocal fry) word detection for conversational speech, with Knarr (default) and F4N2.

    from knarr import CreakDetector
    det = CreakDetector()                                   # Knarr; GPU if available; downloads WavLM-large once
    det.score_file("talk.wav", [("hej", 0.51, 0.80), ("då", 1.20, 1.55)])
"""
from .detector import CreakDetector
from .model import INSUFFICIENT_SUPPORT

__version__ = "1.0.0"
__all__ = ["CreakDetector", "INSUFFICIENT_SUPPORT", "__version__"]
