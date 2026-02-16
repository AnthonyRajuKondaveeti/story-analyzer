"""Tests for the Engagement Analyzer module."""

import pytest
from story_analyzer.engagement_analyzer import EngagementAnalyzer


class TestEngagementAnalyzer:
    """Test suite for EngagementAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EngagementAnalyzer()

    def test_empty_text(self):
        """Test with empty text."""
        result = self.analyzer.analyze("")
        assert result['engagement_score'] == 0.0
        assert result['pacing_variance'] == 0.0

    def test_basic_analysis(self):
        """Test basic story analysis."""
        text = """
        The hero ran through the dark forest. 
        Fear gripped his heart as shadows moved around him.
        "Help!" he screamed into the night.
        """
        result = self.analyzer.analyze(text)
        
        assert 'engagement_score' in result
        assert 'pacing_variance' in result
        assert 'dialogue_density' in result
        assert 'action_density' in result
        assert 'emotional_intensity' in result
        assert 'hook_quality' in result
        assert 'readability' in result
        
        # Check that scores are normalized 0-1
        assert 0 <= result['engagement_score'] <= 1
        assert 0 <= result['pacing_variance'] <= 1

    def test_high_action_text(self):
        """Test text with high action content."""
        text = """
        They run and jump through the burning building.
        The hero fights the villain in an epic chase.
        Explosions strike all around as they escape.
        """
        result = self.analyzer.analyze(text)
        
        # High action should have good action density
        assert result['action_density'] > 0

    def test_dialogue_detection(self):
        """Test dialogue density calculation."""
        text = """
        "Hello," she said. "How are you?"
        "I'm fine," he replied. "Thank you for asking."
        """
        result = self.analyzer.analyze(text)
        
        # Should detect dialogue
        assert result['dialogue_density'] > 0

    def test_emotional_intensity(self):
        """Test emotional word detection."""
        text = """
        She was terrified and filled with fear.
        Her heart was full of love and joy.
        The sadness and anger overwhelmed him.
        """
        result = self.analyzer.analyze(text)
        
        # Should detect emotional content
        assert result['emotional_intensity'] > 0

    def test_hook_quality_short(self):
        """Test hook assessment with short opening."""
        text = """
        The world ended today.
        Nobody seemed to notice at first.
        """
        result = self.analyzer.analyze(text)
        
        # Short, punchy opening should score well
        assert result['hook_quality'] > 0.5

    def test_hook_quality_long(self):
        """Test hook assessment with long opening."""
        text = """
        It was a beautiful morning in the small village nestled between 
        the mountains where everyone knew each other and life moved slowly.
        """
        result = self.analyzer.analyze(text)
        
        # Long opening scores lower
        assert 'hook_quality' in result

    def test_pacing_variance(self):
        """Test pacing analysis with varied sentence lengths."""
        text = """
        Short sentence. 
        This is a medium length sentence with more words.
        This is a very long sentence that goes on and on with many words and clauses.
        Brief.
        """
        result = self.analyzer.analyze(text)
        
        # Varied lengths should show pacing variance
        assert result['pacing_variance'] > 0

    def test_readability(self):
        """Test readability calculation."""
        text = """
        The cat sat on the mat.
        It was a sunny day.
        Birds sang in the trees.
        """
        result = self.analyzer.analyze(text)
        
        # Simple text should have good readability
        assert result['readability'] > 0
