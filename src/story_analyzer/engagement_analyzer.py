"""
Engagement Analyzer Module

Analyzes story engagement using deterministic heuristics such as:
- Pacing analysis (sentence length variation)
- Dialogue density
- Action word frequency
- Emotional intensity
- Reader hook detection
"""

import re
from typing import Dict, List, Optional, Any
from collections import Counter


class EngagementAnalyzer:
    """Analyzes story engagement metrics using deterministic heuristics."""

    # Common action verbs that indicate high engagement
    ACTION_VERBS = {
        'run', 'jump', 'fight', 'chase', 'escape', 'attack', 'defend',
        'explode', 'crash', 'shout', 'scream', 'grab', 'throw', 'strike'
    }

    # Emotional intensity words
    EMOTIONAL_WORDS = {
        'love', 'hate', 'fear', 'anger', 'joy', 'sad', 'happy', 'terrified',
        'furious', 'ecstatic', 'devastated', 'thrilled', 'anxious', 'worried'
    }

    def __init__(self):
        """Initialize the engagement analyzer."""
        pass

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze story engagement using multiple heuristics.

        Args:
            text: The story text to analyze

        Returns:
            Dictionary containing engagement metrics
        """
        if not text or not text.strip():
            return self._empty_result()

        sentences = self._split_sentences(text)
        words = self._tokenize(text)

        return {
            'engagement_score': self._calculate_engagement_score(text, sentences, words),
            'pacing_variance': self._analyze_pacing(sentences),
            'dialogue_density': self._calculate_dialogue_density(text),
            'action_density': self._calculate_action_density(words),
            'emotional_intensity': self._calculate_emotional_intensity(words),
            'hook_quality': self._assess_hook(sentences),
            'readability': self._calculate_readability(sentences, words),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for invalid input."""
        return {
            'engagement_score': 0.0,
            'pacing_variance': 0.0,
            'dialogue_density': 0.0,
            'action_density': 0.0,
            'emotional_intensity': 0.0,
            'hook_quality': 0.0,
            'readability': 0.0,
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _analyze_pacing(self, sentences: List[str]) -> float:
        """
        Analyze pacing by measuring sentence length variation.
        Higher variance indicates more dynamic pacing.
        """
        if len(sentences) < 2:
            return 0.0

        lengths = [len(s.split()) for s in sentences]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)

        # Normalize to 0-1 scale (good pacing has variance between 5-50)
        normalized = min(variance / 50.0, 1.0)
        return round(normalized, 3)

    def _calculate_dialogue_density(self, text: str) -> float:
        """
        Calculate dialogue density using quote detection.
        More dialogue often increases engagement.
        """
        # Count dialogue markers
        dialogue_chars = text.count('"') + text.count("'")
        total_chars = len(text)

        if total_chars == 0:
            return 0.0

        # Normalize: typical engaging stories have 10-30% dialogue
        density = (dialogue_chars / total_chars) * 100
        normalized = min(density / 30.0, 1.0)
        return round(normalized, 3)

    def _calculate_action_density(self, words: List[str]) -> float:
        """Calculate density of action verbs."""
        if not words:
            return 0.0

        action_count = sum(1 for word in words if word in self.ACTION_VERBS)
        density = action_count / len(words)

        # Normalize: good action density is around 2-5%
        normalized = min(density / 0.05, 1.0)
        return round(normalized, 3)

    def _calculate_emotional_intensity(self, words: List[str]) -> float:
        """Calculate emotional intensity using emotional word frequency."""
        if not words:
            return 0.0

        emotional_count = sum(1 for word in words if word in self.EMOTIONAL_WORDS)
        intensity = emotional_count / len(words)

        # Normalize: good emotional intensity is around 3-8%
        normalized = min(intensity / 0.08, 1.0)
        return round(normalized, 3)

    def _assess_hook(self, sentences: List[str]) -> float:
        """
        Assess the quality of the opening hook.
        Good hooks are typically short, punchy, and engaging.
        """
        if not sentences:
            return 0.0

        first_sentence = sentences[0]
        word_count = len(first_sentence.split())

        # Good hooks are typically 5-15 words
        if 5 <= word_count <= 15:
            hook_score = 1.0
        elif word_count < 5:
            hook_score = 0.6
        elif word_count <= 20:
            hook_score = 0.7
        else:
            hook_score = 0.4

        # Bonus for questions, action words, or emotional words
        first_words = set(self._tokenize(first_sentence))
        if '?' in first_sentence:
            hook_score = min(hook_score + 0.2, 1.0)
        if first_words & self.ACTION_VERBS:
            hook_score = min(hook_score + 0.1, 1.0)
        if first_words & self.EMOTIONAL_WORDS:
            hook_score = min(hook_score + 0.1, 1.0)

        return round(hook_score, 3)

    def _calculate_readability(self, sentences: List[str], words: List[str]) -> float:
        """
        Calculate readability score (simplified Flesch-Kincaid).
        Higher scores indicate easier reading.
        """
        if not sentences or not words:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Simplified readability: prefer moderate sentence length (10-20 words)
        # and moderate word length (4-6 characters)
        sentence_score = 1.0 - min(abs(avg_sentence_length - 15) / 15, 1.0)
        word_score = 1.0 - min(abs(avg_word_length - 5) / 5, 1.0)

        readability = (sentence_score + word_score) / 2
        return round(readability, 3)

    def _calculate_engagement_score(self, text: str, sentences: List[str], words: List[str]) -> float:
        """Calculate overall engagement score as weighted average of all metrics."""
        pacing = self._analyze_pacing(sentences)
        dialogue = self._calculate_dialogue_density(text)
        action = self._calculate_action_density(words)
        emotion = self._calculate_emotional_intensity(words)
        hook = self._assess_hook(sentences)
        readability = self._calculate_readability(sentences, words)

        # Weighted average
        engagement = (
            pacing * 0.20 +
            dialogue * 0.15 +
            action * 0.20 +
            emotion * 0.20 +
            hook * 0.15 +
            readability * 0.10
        )

        return round(engagement, 3)
