"""
Genre Classifier Module

Classifies story genres using rule-based heuristics and pattern matching:
- Keyword and phrase detection for genre indicators
- Narrative structure analysis
- Setting and atmosphere detection
- Character archetype identification
"""

import re
from typing import Dict, List, Set, Tuple, Any
from collections import Counter


class GenreClassifier:
    """Classifies story genres using deterministic heuristics."""

    # Thresholds for tone detection
    MIN_HUMOROUS_INDICATORS = 2
    MIN_SERIOUS_INDICATORS = 2
    LONG_SENTENCE_THRESHOLD = 20

    # Genre-specific keyword sets
    GENRE_KEYWORDS = {
        'fantasy': {
            'magic', 'wizard', 'dragon', 'sword', 'kingdom', 'quest', 'spell',
            'enchanted', 'castle', 'knight', 'elf', 'dwarf', 'prophecy', 'realm'
        },
        'science_fiction': {
            'space', 'robot', 'alien', 'technology', 'future', 'spaceship',
            'laser', 'planet', 'galaxy', 'android', 'cyborg', 'quantum', 'AI'
        },
        'mystery': {
            'detective', 'clue', 'suspect', 'investigate', 'murder', 'crime',
            'mystery', 'evidence', 'witness', 'alibi', 'case', 'solve'
        },
        'romance': {
            'love', 'heart', 'kiss', 'romance', 'passion', 'relationship',
            'boyfriend', 'girlfriend', 'wedding', 'date', 'attraction', 'desire'
        },
        'horror': {
            'fear', 'terror', 'scream', 'blood', 'dark', 'monster', 'ghost',
            'haunted', 'nightmare', 'evil', 'demon', 'death', 'shadow'
        },
        'thriller': {
            'danger', 'chase', 'escape', 'threat', 'conspiracy', 'assassin',
            'spy', 'mission', 'enemy', 'urgent', 'pursue', 'trap'
        },
        'adventure': {
            'journey', 'explore', 'discover', 'treasure', 'expedition', 'quest',
            'adventure', 'travel', 'mountain', 'island', 'map', 'wilderness'
        },
        'historical': {
            'war', 'ancient', 'century', 'empire', 'revolution', 'king',
            'queen', 'historical', 'era', 'dynasty', 'colonial', 'medieval'
        },
        'comedy': {
            'funny', 'laugh', 'joke', 'humor', 'silly', 'ridiculous',
            'comedy', 'hilarious', 'amusing', 'grin', 'chuckle', 'wit'
        },
        'drama': {
            'conflict', 'emotion', 'struggle', 'tension', 'crisis', 'family',
            'relationship', 'society', 'personal', 'moral', 'choice', 'decision'
        },
    }

    def __init__(self):
        """Initialize the genre classifier."""
        pass

    def classify(self, text: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Classify the story into one or more genres.

        Args:
            text: The story text to classify
            top_n: Number of top genres to return

        Returns:
            Dictionary containing genre classifications and confidence scores
        """
        if not text or not text.strip():
            return self._empty_result()

        words = self._tokenize(text)
        sentences = self._split_sentences(text)

        # Calculate genre scores
        genre_scores = self._calculate_genre_scores(words)
        
        # Get top N genres
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        top_genres = sorted_genres[:top_n]

        # Normalize scores to sum to 1.0
        total_score = sum(score for _, score in top_genres)
        if total_score > 0:
            normalized_genres = {
                genre: round(score / total_score, 3) 
                for genre, score in top_genres
            }
        else:
            normalized_genres = {}

        return {
            'primary_genre': top_genres[0][0] if top_genres else 'general',
            'genre_scores': normalized_genres,
            'genre_confidence': top_genres[0][1] if top_genres else 0.0,
            'is_mixed_genre': self._is_mixed_genre(top_genres),
            'subgenres': self._identify_subgenres(words, top_genres[0][0] if top_genres else 'general'),
            'tone': self._analyze_tone(words, sentences),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for invalid input."""
        return {
            'primary_genre': 'general',
            'genre_scores': {},
            'genre_confidence': 0.0,
            'is_mixed_genre': False,
            'subgenres': [],
            'tone': 'neutral',
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _calculate_genre_scores(self, words: List[str]) -> Dict[str, float]:
        """Calculate scores for each genre based on keyword matches."""
        if not words:
            return {}

        word_set = set(words)
        word_counts = Counter(words)
        genre_scores = {}

        for genre, keywords in self.GENRE_KEYWORDS.items():
            # Count matches with frequency weighting
            match_score = 0
            for keyword in keywords:
                if keyword in word_set:
                    # Weight by frequency (more mentions = stronger signal)
                    match_score += word_counts[keyword] * 0.1
            
            # Normalize by total words
            genre_scores[genre] = match_score / len(words) * 100

        return genre_scores

    def _is_mixed_genre(self, top_genres: List[Tuple[str, float]]) -> bool:
        """Determine if the story is a mixed-genre work."""
        if len(top_genres) < 2:
            return False
        
        # If top two genres are close in score, it's mixed genre
        if top_genres[1][1] >= top_genres[0][1] * 0.7:
            return True
        
        return False

    def _identify_subgenres(self, words: List[str], primary_genre: str) -> List[str]:
        """Identify subgenres within the primary genre."""
        subgenres = []
        word_set = set(words)

        # Define subgenres for each main genre
        subgenre_keywords = {
            'fantasy': {
                'epic': {'quest', 'kingdom', 'prophecy', 'destiny'},
                'urban': {'city', 'modern', 'street', 'urban'},
                'dark': {'dark', 'evil', 'shadow', 'curse'},
            },
            'science_fiction': {
                'cyberpunk': {'cyber', 'hacker', 'virtual', 'network'},
                'space_opera': {'space', 'galaxy', 'empire', 'fleet'},
                'dystopian': {'dystopia', 'oppression', 'control', 'regime'},
            },
            'mystery': {
                'detective': {'detective', 'investigator', 'sleuth'},
                'noir': {'dark', 'shadow', 'rain', 'night'},
                'cozy': {'village', 'amateur', 'tea', 'small'},
            },
            'horror': {
                'psychological': {'mind', 'madness', 'insane', 'paranoia'},
                'supernatural': {'ghost', 'spirit', 'demon', 'haunted'},
                'slasher': {'killer', 'murder', 'blood', 'victim'},
            },
        }

        if primary_genre in subgenre_keywords:
            for subgenre, keywords in subgenre_keywords[primary_genre].items():
                if len(word_set & keywords) >= 2:
                    subgenres.append(subgenre)

        return subgenres

    def _analyze_tone(self, words: List[str], sentences: List[str]) -> str:
        """Analyze the overall tone of the story."""
        word_set = set(words)
        
        # Define tone indicators
        dark_words = {'dark', 'death', 'fear', 'evil', 'shadow', 'grim', 'bleak'}
        light_words = {'light', 'hope', 'joy', 'bright', 'happy', 'cheerful'}
        serious_words = {'grave', 'serious', 'solemn', 'important', 'critical'}
        humorous_words = {'funny', 'laugh', 'joke', 'silly', 'amusing', 'comic'}
        
        dark_count = len(word_set & dark_words)
        light_count = len(word_set & light_words)
        serious_count = len(word_set & serious_words)
        humorous_count = len(word_set & humorous_words)
        
        # Check punctuation for tone hints
        exclamations = sum(s.count('!') for s in sentences)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Determine tone
        if dark_count > light_count * 1.5:
            return 'dark'
        elif light_count > dark_count * 1.5:
            return 'light'
        elif humorous_count >= self.MIN_HUMOROUS_INDICATORS:
            return 'humorous'
        elif serious_count >= self.MIN_SERIOUS_INDICATORS or avg_sentence_length > self.LONG_SENTENCE_THRESHOLD:
            return 'serious'
        else:
            return 'neutral'
