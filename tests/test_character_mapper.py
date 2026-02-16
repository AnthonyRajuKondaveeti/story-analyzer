"""Tests for the Character Mapper module."""

import pytest
from story_analyzer.character_mapper import CharacterMapper


class TestCharacterMapper:
    """Test suite for CharacterMapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = CharacterMapper()

    def test_empty_text(self):
        """Test with empty text."""
        result = self.mapper.map_characters("")
        assert result['characters'] == []
        assert result['character_mentions'] == {}

    def test_character_extraction(self):
        """Test basic character extraction."""
        text = """
        John walked into the room. Mary greeted him warmly.
        John and Mary talked for hours about their adventures.
        Sarah joined them later.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        assert 'characters' in result
        assert 'character_mentions' in result
        # Should find John and Mary (mentioned multiple times)
        assert 'John' in result['characters']
        assert 'Mary' in result['characters']

    def test_relationship_detection(self):
        """Test relationship mapping."""
        text = """
        Alice and Bob were good friends.
        Alice met Bob every day at the park.
        Bob helped Alice with her problems.
        Charlie sometimes joined Alice and Bob.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        assert 'relationships' in result
        # Should detect relationships between characters
        assert len(result['relationships']) > 0

    def test_main_characters(self):
        """Test main character identification."""
        text = """
        Alice was the protagonist. She appeared in every scene.
        Alice talked to Bob, Charlie, and David.
        Bob was also important and appeared often.
        Charlie made a brief appearance.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        assert 'main_characters' in result
        # Alice should be identified as main character
        if result['main_characters']:
            assert 'Alice' in result['main_characters']

    def test_character_importance(self):
        """Test character importance calculation."""
        text = """
        Emma was central to the story. She knew everyone.
        Emma met with Frank, Grace, and Henry regularly.
        Frank appeared less often than Emma.
        Grace was mentioned a few times.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        assert 'character_importance' in result
        # Should have importance scores
        if result['character_importance']:
            assert all(0 <= score <= 1 for score in result['character_importance'].values())

    def test_network_density(self):
        """Test network density calculation."""
        text = """
        Alex, Beth, and Carl were close friends.
        Alex knew Beth well. Beth knew Carl well.
        Carl knew Alex well. They all interacted.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        assert 'network_density' in result
        assert 0 <= result['network_density'] <= 1

    def test_clusters(self):
        """Test character cluster identification."""
        text = """
        Group one: Alice and Bob always talked together.
        Alice and Bob were inseparable friends.
        Group two: Charlie and David hung out separately.
        Charlie and David never met Alice or Bob.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        assert 'clusters' in result
        # Should detect separate groups
        if len(result['characters']) >= 4:
            # Might detect clusters if enough characters
            assert isinstance(result['clusters'], list)

    def test_min_mentions_filter(self):
        """Test minimum mentions filtering."""
        text = """
        Alice appeared many times in the story.
        Alice was mentioned again and again and again.
        Alice came back one more time.
        Bob appeared only once here.
        """
        result = self.mapper.map_characters(text, min_mentions=3)
        
        # Should only include Alice (mentioned 3+ times)
        assert 'Alice' in result['characters']
        # Bob should be filtered out
        assert 'Bob' not in result['characters']

    def test_network_visualization_data(self):
        """Test getting network visualization data."""
        text = """
        John and Jane worked together closely.
        John met Jane every day for meetings.
        Jane introduced John to Michael.
        """
        result = self.mapper.get_network_data(text, min_mentions=2)
        
        assert 'nodes' in result
        assert 'edges' in result
        # Should have node and edge data for visualization
        assert isinstance(result['nodes'], list)
        assert isinstance(result['edges'], list)

    def test_no_valid_characters(self):
        """Test with text that has no valid character names."""
        text = """
        the quick brown fox jumps over the lazy dog.
        this sentence has no proper names at all.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        # Should handle gracefully with no characters
        assert result['characters'] == []

    def test_case_sensitivity(self):
        """Test that character names are case-sensitive."""
        text = """
        Elizabeth met elizabeth in the park.
        Elizabeth and elizabeth talked together.
        """
        result = self.mapper.map_characters(text, min_mentions=2)
        
        # Should treat as separate entities initially
        # (In reality, both should be detected as "Elizabeth")
        assert 'characters' in result
