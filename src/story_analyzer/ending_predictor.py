"""
Ending Predictor Module

Predicts story endings using pattern recognition and narrative arc analysis:
- Story arc identification (e.g., rising action, climax, resolution)
- Conflict resolution patterns
- Character trajectory analysis
- Tension curve analysis
"""

import re
from typing import Dict, List, Optional, Any
from enum import Enum


class EndingType(Enum):
    """Possible story ending types."""
    HAPPY = "happy"
    TRAGIC = "tragic"
    BITTERSWEET = "bittersweet"
    OPEN = "open"
    TWIST = "twist"
    AMBIGUOUS = "ambiguous"


class EndingPredictor:
    """Predicts story endings based on narrative patterns."""

    # Thresholds for ending detection
    MIN_TWIST_INDICATORS = 2
    MIN_AMBIGUOUS_INDICATORS = 2

    # Words associated with positive endings
    POSITIVE_WORDS = {
        'success', 'victory', 'triumph', 'happiness', 'love', 'joy', 'peace',
        'reunion', 'achieve', 'win', 'rescue', 'save', 'celebrate', 'wedding'
    }

    # Words associated with negative endings
    NEGATIVE_WORDS = {
        'death', 'loss', 'defeat', 'tragedy', 'fail', 'sacrifice', 'die',
        'destroy', 'end', 'darkness', 'despair', 'farewell', 'goodbye'
    }

    # Words indicating twists or surprises
    TWIST_WORDS = {
        'suddenly', 'however', 'but', 'unexpectedly', 'surprise', 'reveal',
        'actually', 'twist', 'shock', 'never', 'all along'
    }

    # Words indicating ambiguity
    AMBIGUOUS_WORDS = {
        'maybe', 'perhaps', 'might', 'could', 'uncertain', 'unclear',
        'wonder', 'question', 'mystery', 'unknown'
    }

    def __init__(self):
        """Initialize the ending predictor."""
        pass

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the story ending based on narrative patterns.

        Args:
            text: The story text to analyze

        Returns:
            Dictionary containing ending predictions and confidence scores
        """
        if not text or not text.strip():
            return self._empty_result()

        sentences = self._split_sentences(text)
        words = self._tokenize(text)

        # Analyze different sections of the story
        opening = ' '.join(sentences[:len(sentences)//3]) if len(sentences) >= 3 else text
        middle = ' '.join(sentences[len(sentences)//3:2*len(sentences)//3]) if len(sentences) >= 3 else text
        ending = ' '.join(sentences[2*len(sentences)//3:]) if len(sentences) >= 3 else text

        tension_curve = self._analyze_tension_curve(sentences)
        ending_type = self._predict_ending_type(ending, words)
        
        return {
            'predicted_ending': ending_type.value,
            'confidence': self._calculate_confidence(ending, words),
            'tension_curve': tension_curve,
            'arc_completion': self._analyze_arc_completion(tension_curve),
            'resolution_likelihood': self._predict_resolution(ending, words),
            'emotional_trajectory': self._analyze_emotional_trajectory(sentences),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for invalid input."""
        return {
            'predicted_ending': EndingType.AMBIGUOUS.value,
            'confidence': 0.0,
            'tension_curve': [],
            'arc_completion': 0.0,
            'resolution_likelihood': 0.0,
            'emotional_trajectory': 'neutral',
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _analyze_tension_curve(self, sentences: List[str]) -> List[float]:
        """
        Analyze tension curve throughout the story.
        Returns normalized tension values for each segment.
        """
        if not sentences:
            return []

        # Divide story into 10 segments
        segment_size = max(1, len(sentences) // 10)
        tension_curve = []

        for i in range(0, len(sentences), segment_size):
            segment = ' '.join(sentences[i:i+segment_size])
            tension = self._calculate_segment_tension(segment)
            tension_curve.append(tension)

        return tension_curve

    def _calculate_segment_tension(self, segment: str) -> float:
        """Calculate tension level for a text segment."""
        words = self._tokenize(segment)
        
        # Count tension indicators
        action_words = {'run', 'fight', 'chase', 'escape', 'attack', 'danger'}
        conflict_words = {'argue', 'conflict', 'problem', 'challenge', 'threat'}
        emotional_words = {'fear', 'worry', 'anxiety', 'panic', 'dread'}
        
        tension_count = sum(1 for word in words if 
                           word in action_words or 
                           word in conflict_words or 
                           word in emotional_words)
        
        # Check for exclamation marks and questions (indicate tension)
        punctuation_tension = segment.count('!') * 0.1 + segment.count('?') * 0.05
        
        if not words:
            return 0.0
        
        tension = min((tension_count / len(words)) * 10 + punctuation_tension, 1.0)
        return round(tension, 3)

    def _predict_ending_type(self, ending_text: str, all_words: List[str]) -> EndingType:
        """Predict the type of ending based on text patterns."""
        ending_words = self._tokenize(ending_text)
        
        # Count different types of indicators
        positive_count = sum(1 for word in ending_words if word in self.POSITIVE_WORDS)
        negative_count = sum(1 for word in ending_words if word in self.NEGATIVE_WORDS)
        twist_count = sum(1 for word in ending_words if word in self.TWIST_WORDS)
        ambiguous_count = sum(1 for word in ending_words if word in self.AMBIGUOUS_WORDS)
        
        # Decision logic
        if twist_count >= self.MIN_TWIST_INDICATORS:
            return EndingType.TWIST
        
        if ambiguous_count >= self.MIN_AMBIGUOUS_INDICATORS:
            return EndingType.AMBIGUOUS
        
        if positive_count > negative_count * 1.5:
            return EndingType.HAPPY
        elif negative_count > positive_count * 1.5:
            return EndingType.TRAGIC
        elif positive_count > 0 and negative_count > 0:
            return EndingType.BITTERSWEET
        else:
            return EndingType.OPEN

    def _calculate_confidence(self, ending_text: str, all_words: List[str]) -> float:
        """Calculate confidence in the ending prediction."""
        if not ending_text:
            return 0.0
        
        ending_words = self._tokenize(ending_text)
        if not ending_words:
            return 0.0
        
        # Count total indicator words
        indicator_count = sum(1 for word in ending_words if 
                            word in self.POSITIVE_WORDS or
                            word in self.NEGATIVE_WORDS or
                            word in self.TWIST_WORDS or
                            word in self.AMBIGUOUS_WORDS)
        
        # Confidence based on indicator density
        confidence = min(indicator_count / len(ending_words) * 2, 1.0)
        
        # Boost confidence if ending is substantial
        if len(ending_words) > 20:
            confidence = min(confidence * 1.2, 1.0)
        
        return round(confidence, 3)

    def _analyze_arc_completion(self, tension_curve: List[float]) -> float:
        """
        Analyze how complete the narrative arc is.
        Complete arcs typically rise then fall.
        """
        if len(tension_curve) < 3:
            return 0.5  # Not enough data
        
        # Check for rise-then-fall pattern
        mid_point = len(tension_curve) // 2
        first_half_avg = sum(tension_curve[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(tension_curve[mid_point:]) / (len(tension_curve) - mid_point)
        
        # Check if there's a peak in the middle-to-late portion
        max_tension = max(tension_curve) if tension_curve else 0
        max_index = tension_curve.index(max_tension) if max_tension in tension_curve else 0
        
        # Good arc has peak in 40-80% range and falls afterward
        peak_position = max_index / len(tension_curve)
        has_fall = tension_curve[-1] < max_tension * 0.8 if max_tension > 0 else False
        
        completion = 0.0
        if 0.4 <= peak_position <= 0.8:
            completion += 0.5
        if has_fall:
            completion += 0.5
        
        return round(completion, 3)

    def _predict_resolution(self, ending_text: str, all_words: List[str]) -> float:
        """Predict likelihood of a resolved ending vs. open ending."""
        ending_words = self._tokenize(ending_text)
        
        # Words indicating resolution
        resolution_words = {
            'finally', 'end', 'concluded', 'resolved', 'finished', 'complete',
            'settled', 'decided', 'closure', 'peace', 'rest'
        }
        
        # Words indicating open ending
        open_words = {
            'continue', 'journey', 'begin', 'start', 'next', 'future',
            'tomorrow', 'onward', 'ahead'
        }
        
        resolution_count = sum(1 for word in ending_words if word in resolution_words)
        open_count = sum(1 for word in ending_words if word in open_words)
        
        if not ending_words:
            return 0.5
        
        # Calculate resolution likelihood
        if resolution_count + open_count == 0:
            return 0.5
        
        likelihood = resolution_count / (resolution_count + open_count)
        return round(likelihood, 3)

    def _analyze_emotional_trajectory(self, sentences: List[str]) -> str:
        """Analyze the overall emotional trajectory of the story."""
        if len(sentences) < 3:
            return 'neutral'
        
        # Analyze beginning and end emotions
        beginning = ' '.join(sentences[:len(sentences)//3])
        ending = ' '.join(sentences[2*len(sentences)//3:])
        
        begin_words = set(self._tokenize(beginning))
        end_words = set(self._tokenize(ending))
        
        begin_positive = len(begin_words & self.POSITIVE_WORDS)
        begin_negative = len(begin_words & self.NEGATIVE_WORDS)
        end_positive = len(end_words & self.POSITIVE_WORDS)
        end_negative = len(end_words & self.NEGATIVE_WORDS)
        
        # Determine trajectory
        if end_positive > end_negative and begin_positive <= begin_negative:
            return 'ascending'  # Gets better
        elif end_negative > end_positive and begin_negative <= begin_positive:
            return 'descending'  # Gets worse
        elif end_positive > end_negative and begin_positive > begin_negative:
            return 'positive'  # Stays positive
        elif end_negative > end_positive and begin_negative > begin_positive:
            return 'negative'  # Stays negative
        else:
            return 'neutral'  # No clear trajectory
