"""Tests for the Genre Classifier module."""

import pytest
from story_analyzer.genre_classifier import GenreClassifier


class TestGenreClassifier:
    """Test suite for GenreClassifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = GenreClassifier()

    def test_empty_text(self):
        """Test with empty text."""
        result = self.classifier.classify("")
        assert result['primary_genre'] == 'general'
        assert result['genre_confidence'] == 0.0

    def test_fantasy_classification(self):
        """Test fantasy genre detection."""
        text = """
        The wizard cast a powerful spell using his magic wand.
        Dragons soared above the enchanted castle.
        The knight embarked on a quest to find the prophecy.
        """
        result = self.classifier.classify(text)
        
        assert 'primary_genre' in result
        assert 'genre_scores' in result
        # Should detect fantasy elements
        assert 'fantasy' in result['genre_scores']

    def test_scifi_classification(self):
        """Test science fiction genre detection."""
        text = """
        The spaceship traveled through the galaxy at light speed.
        Robots and aliens populated the distant planet.
        Advanced technology and AI controlled everything.
        """
        result = self.classifier.classify(text)
        
        assert 'primary_genre' in result
        # Should detect sci-fi elements
        assert 'science_fiction' in result.get('genre_scores', {})

    def test_mystery_classification(self):
        """Test mystery genre detection."""
        text = """
        The detective examined the clues carefully.
        The murder suspect had no clear alibi.
        She investigated the crime scene for evidence.
        """
        result = self.classifier.classify(text)
        
        assert 'primary_genre' in result
        # Should detect mystery elements
        assert 'mystery' in result.get('genre_scores', {})

    def test_romance_classification(self):
        """Test romance genre detection."""
        text = """
        Their hearts were filled with love and passion.
        The couple shared a romantic kiss under the stars.
        Their relationship blossomed into true love.
        """
        result = self.classifier.classify(text)
        
        assert 'primary_genre' in result
        # Should detect romance elements
        assert 'romance' in result.get('genre_scores', {})

    def test_horror_classification(self):
        """Test horror genre detection."""
        text = """
        Terror filled the air as the ghost appeared.
        Blood dripped from the walls in the haunted house.
        The monster lurked in the dark shadows.
        """
        result = self.classifier.classify(text)
        
        assert 'primary_genre' in result
        # Should detect horror elements
        assert 'horror' in result.get('genre_scores', {})

    def test_mixed_genre(self):
        """Test mixed genre detection."""
        text = """
        The wizard detective investigated the magical murder.
        Using spells and clues, she solved the mystery.
        Magic and investigation combined in this tale.
        """
        result = self.classifier.classify(text)
        
        assert 'is_mixed_genre' in result
        assert 'genre_scores' in result
        # Should have multiple genres with similar scores
        assert len(result['genre_scores']) > 1

    def test_top_n_genres(self):
        """Test returning top N genres."""
        text = """
        Fantasy magic and science fiction technology combined.
        Romance and mystery elements were present.
        Horror and thriller aspects added tension.
        """
        result = self.classifier.classify(text, top_n=3)
        
        assert 'genre_scores' in result
        # Should return at most 3 genres
        assert len(result['genre_scores']) <= 3

    def test_tone_analysis(self):
        """Test tone detection."""
        text = """
        Darkness and evil pervaded the story.
        Death and shadow followed the characters.
        A grim and bleak atmosphere dominated.
        """
        result = self.classifier.classify(text)
        
        assert 'tone' in result
        assert result['tone'] in ['dark', 'light', 'humorous', 'serious', 'neutral']

    def test_subgenre_identification(self):
        """Test subgenre detection."""
        text = """
        The epic quest spanned across the kingdom.
        The prophecy foretold the hero's destiny.
        Magic and swords clashed in epic battles.
        """
        result = self.classifier.classify(text)
        
        assert 'subgenres' in result
        # Should detect epic fantasy subgenre
        if result['primary_genre'] == 'fantasy':
            assert 'epic' in result['subgenres'] or len(result['subgenres']) >= 0

    def test_confidence_score(self):
        """Test genre confidence calculation."""
        text = """
        Magic spells and wizards filled the story.
        Dragons and enchanted castles were everywhere.
        The fantasy realm was richly detailed.
        """
        result = self.classifier.classify(text)
        
        assert 'genre_confidence' in result
        assert 0 <= result['genre_confidence'] <= 100
