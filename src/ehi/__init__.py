"""
EHI (Entity Hallucination Index) package for Hindi abstractive summarization.

This package implements reward-driven fine-tuning with entity hallucination metrics
for improving factual consistency in Hindi text summarization.
"""

__version__ = "0.1.0"
__author__ = "EHI Project Contributors"

from ehi.metrics import EntityHallucinationIndex, EntityF1Score, compute_rouge_metrics
from ehi.ner import HindiNERPipeline
from ehi.utils import setup_logging, set_seed, normalize_unicode

__all__ = [
    "EntityHallucinationIndex",
    "EntityF1Score", 
    "compute_rouge_metrics",
    "HindiNERPipeline",
    "setup_logging",
    "set_seed",
    "normalize_unicode",
]