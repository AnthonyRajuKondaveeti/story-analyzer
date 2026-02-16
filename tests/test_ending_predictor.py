"""Tests for the Ending Predictor module."""

import pytest
from story_analyzer.ending_predictor import EndingPredictor, EndingType


class TestEndingPredictor:
    """Test suite for EndingPredictor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = EndingPredictor()

    def test_empty_text(self):
        """Test with empty text."""
        result = self.predictor.predict("")
        assert result['predicted_ending'] == EndingType.AMBIGUOUS.value
        assert result['confidence'] == 0.0

    def test_happy_ending(self):
        """Test prediction of happy ending."""
        text = """
        The hero faced many challenges but persevered.
        In the end, they achieved victory and found love.
        Everyone celebrated their success and lived happily.
        """
        result = self.predictor.predict(text)
        
        assert 'predicted_ending' in result
        assert 'confidence' in result
        assert result['predicted_ending'] in [e.value for e in EndingType]

    def test_tragic_ending(self):
        """Test prediction of tragic ending."""
        text = """
        Despite their efforts, everything fell apart.
        The hero faced defeat and ultimate loss.
        Death and tragedy marked the end of the story.
        """
        result = self.predictor.predict(text)
        
        assert 'predicted_ending' in result
        # Should detect tragic elements
        assert result['predicted_ending'] in [EndingType.TRAGIC.value, EndingType.BITTERSWEET.value]

    def test_twist_ending(self):
        """Test prediction of twist ending."""
        text = """
        Everything seemed normal until the very end.
        However, suddenly everything changed unexpectedly.
        The shocking twist revealed the truth all along.
        But wait, there was yet another surprise twist.
        """
        result = self.predictor.predict(text)
        
        assert 'predicted_ending' in result
        # Should detect twist indicators (needs at least 2 twist words)
        assert result['predicted_ending'] == EndingType.TWIST.value

    def test_tension_curve(self):
        """Test tension curve analysis."""
        text = """
        It was a peaceful day. The calm before the storm.
        Suddenly danger struck! Fear and panic ensued.
        They fought bravely against impossible odds.
        Finally, peace returned to the land.
        """
        result = self.predictor.predict(text)
        
        assert 'tension_curve' in result
        assert isinstance(result['tension_curve'], list)
        # Tension curve should have values
        assert len(result['tension_curve']) > 0

    def test_arc_completion(self):
        """Test narrative arc completion analysis."""
        text = """
        The journey began with hope and excitement.
        Challenges arose and tension mounted steadily.
        The climax was intense with danger everywhere.
        Resolution came and the story concluded peacefully.
        """
        result = self.predictor.predict(text)
        
        assert 'arc_completion' in result
        assert 0 <= result['arc_completion'] <= 1

    def test_resolution_likelihood(self):
        """Test resolution prediction."""
        text = """
        The story reached its end and concluded finally.
        All conflicts were resolved and peace was achieved.
        Everything was finished and complete.
        """
        result = self.predictor.predict(text)
        
        assert 'resolution_likelihood' in result
        assert 0 <= result['resolution_likelihood'] <= 1

    def test_emotional_trajectory(self):
        """Test emotional trajectory analysis."""
        text = """
        In the beginning, there was darkness and despair.
        Through the middle, hope began to grow.
        By the end, joy and happiness prevailed.
        """
        result = self.predictor.predict(text)
        
        assert 'emotional_trajectory' in result
        assert result['emotional_trajectory'] in [
            'ascending', 'descending', 'positive', 'negative', 'neutral'
        ]

    def test_short_text(self):
        """Test with very short text."""
        text = "The end came swiftly."
        result = self.predictor.predict(text)
        
        # Should still return valid results
        assert 'predicted_ending' in result
        assert 'confidence' in result
