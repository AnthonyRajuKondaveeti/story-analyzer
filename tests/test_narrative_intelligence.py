"""Tests for the Narrative Intelligence main module."""

import pytest
from story_analyzer import NarrativeIntelligence


class TestNarrativeIntelligence:
    """Test suite for NarrativeIntelligence."""

    def setup_method(self):
        """Set up test fixtures."""
        self.nis = NarrativeIntelligence()

    def test_initialization(self):
        """Test NIS initialization."""
        assert self.nis is not None
        assert self.nis.engagement_analyzer is not None
        assert self.nis.ending_predictor is not None
        assert self.nis.genre_classifier is not None
        assert self.nis.character_mapper is not None

    def test_empty_text(self):
        """Test with empty text."""
        result = self.nis.analyze("")
        assert 'error' in result or result['word_count'] == 0

    def test_full_analysis(self):
        """Test complete story analysis."""
        text = """
        Alice was a brave detective in the magical city.
        She investigated mysterious crimes using her magic powers.
        "I will solve this case," Alice declared confidently.
        Bob, her partner, helped her chase the criminal.
        In the end, they achieved victory and justice prevailed.
        """
        result = self.nis.analyze(text)
        
        assert 'text_length' in result
        assert 'word_count' in result
        assert 'analysis_mode' in result
        assert result['analysis_mode'] == 'deterministic'
        
        # Should have all component results
        assert 'engagement' in result
        assert 'ending_prediction' in result
        assert 'genre' in result
        assert 'characters' in result
        assert 'summary' in result

    def test_selective_analysis(self):
        """Test running only selected components."""
        text = """
        A simple story for testing selective analysis.
        It has some content but nothing complex.
        """
        
        # Only engagement
        result = self.nis.analyze(text, components=['engagement'])
        assert 'engagement' in result
        assert 'genre' not in result
        assert 'characters' not in result
        
        # Only genre
        result = self.nis.analyze(text, components=['genre'])
        assert 'genre' in result
        assert 'engagement' not in result

    def test_engagement_only(self):
        """Test engagement-only analysis."""
        text = """
        The hero ran through danger. "Help!" he screamed.
        Fear and excitement filled every moment.
        """
        result = self.nis.analyze_engagement(text)
        
        assert 'engagement_score' in result
        assert 'pacing_variance' in result

    def test_ending_only(self):
        """Test ending prediction only."""
        text = """
        The story built up tension gradually.
        Finally, everything was resolved happily.
        Success and joy marked the conclusion.
        """
        result = self.nis.predict_ending(text)
        
        assert 'predicted_ending' in result
        assert 'confidence' in result

    def test_genre_only(self):
        """Test genre classification only."""
        text = """
        Magic and wizards filled the fantasy realm.
        Dragons and spells created an epic adventure.
        """
        result = self.nis.classify_genre(text)
        
        assert 'primary_genre' in result
        assert 'genre_scores' in result

    def test_characters_only(self):
        """Test character mapping only."""
        text = """
        Sarah and Tom were best friends.
        Sarah met Tom every day at school.
        Tom helped Sarah with her homework.
        """
        result = self.nis.map_characters(text)
        
        assert 'characters' in result
        assert 'relationships' in result

    def test_character_network(self):
        """Test character network data extraction."""
        text = """
        Alice worked with Bob on the project.
        Bob introduced Alice to Carol.
        Carol and Alice became friends.
        """
        result = self.nis.get_character_network(text)
        
        assert 'nodes' in result
        assert 'edges' in result

    def test_summary_generation(self):
        """Test summary generation."""
        text = """
        A magical detective story with Alice and Bob.
        They solved mysteries using magic and courage.
        "We did it!" Alice said as they succeeded.
        Victory and happiness concluded their adventure.
        """
        result = self.nis.analyze(text)
        
        assert 'summary' in result
        summary = result['summary']
        
        # Summary should have key information
        assert 'engagement_level' in summary or len(summary) == 0
        assert 'primary_genre' in summary or len(summary) == 0

    def test_llm_initialization(self):
        """Test initialization with LLM mode."""
        nis_llm = NarrativeIntelligence(use_llm=True)
        assert nis_llm.use_llm is True
        
        text = "A simple test story."
        result = nis_llm.analyze(text)
        assert result['analysis_mode'] == 'llm_enhanced'

    def test_metadata(self):
        """Test that basic metadata is included."""
        text = "This is a test story with several words."
        result = self.nis.analyze(text)
        
        assert result['text_length'] > 0
        assert result['word_count'] > 0
        assert result['word_count'] == len(text.split())

    def test_long_story_analysis(self):
        """Test analysis of longer story."""
        text = """
        Once upon a time in a magical kingdom far away, there lived a brave knight named Sir George.
        Sir George was known throughout the land for his courage and honor.
        One day, a terrible dragon threatened the peaceful village.
        The villagers were terrified and called upon Sir George for help.
        "I will protect you all," Sir George declared bravely.
        He mounted his horse and rode toward the dragon's lair.
        The battle was fierce and dangerous.
        Sir George fought with all his strength and skill.
        After a long struggle, he finally defeated the evil dragon.
        The village celebrated his victory with a grand feast.
        Sir George was hailed as a hero forever.
        """
        result = self.nis.analyze(text)
        
        # Should successfully analyze longer text
        assert result['word_count'] > 50
        assert 'engagement' in result
        assert 'genre' in result
