"""
Narrative Intelligence System (NIS)

A comprehensive story analysis toolkit combining deterministic heuristics
with optional LLM semantic understanding for reliable, cost-efficient content analysis.
"""

__version__ = "0.1.0"

from .engagement_analyzer import EngagementAnalyzer
from .ending_predictor import EndingPredictor
from .genre_classifier import GenreClassifier
from .character_mapper import CharacterMapper
from .narrative_intelligence import NarrativeIntelligence

__all__ = [
    "EngagementAnalyzer",
    "EndingPredictor",
    "GenreClassifier",
    "CharacterMapper",
    "NarrativeIntelligence",
]
