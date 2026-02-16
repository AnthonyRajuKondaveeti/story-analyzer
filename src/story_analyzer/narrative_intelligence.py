"""
Narrative Intelligence Module

Main orchestrator for the Narrative Intelligence System (NIS).
Combines all analysis components with optional LLM enhancement.
"""

from typing import Dict, Optional, List, Any
from .engagement_analyzer import EngagementAnalyzer
from .ending_predictor import EndingPredictor
from .genre_classifier import GenreClassifier
from .character_mapper import CharacterMapper


class NarrativeIntelligence:
    """
    Main interface for the Narrative Intelligence System.
    
    Combines deterministic heuristics with optional LLM semantic understanding
    for comprehensive story analysis.
    """

    def __init__(self, use_llm: bool = False, llm_config: Optional[Dict] = None):
        """
        Initialize the Narrative Intelligence System.

        Args:
            use_llm: Whether to use LLM for enhanced semantic understanding
            llm_config: Configuration for LLM (API keys, model selection, etc.)
        """
        self.use_llm = use_llm
        self.llm_config = llm_config or {}
        
        # Initialize core components
        self.engagement_analyzer = EngagementAnalyzer()
        self.ending_predictor = EndingPredictor()
        self.genre_classifier = GenreClassifier()
        self.character_mapper = CharacterMapper()
        
        # Initialize LLM components if requested
        self.llm_client = None
        if use_llm:
            self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM client (optional enhancement)."""
        # This would initialize OpenAI or other LLM clients
        # For now, it's a placeholder for future LLM integration
        pass

    def analyze(self, text: str, components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive narrative analysis.

        Args:
            text: The story text to analyze
            components: List of components to run. If None, runs all.
                       Options: ['engagement', 'ending', 'genre', 'characters']

        Returns:
            Dictionary containing analysis results from all requested components
        """
        if not text or not text.strip():
            return self._empty_result()

        # Default to all components if not specified
        if components is None:
            components = ['engagement', 'ending', 'genre', 'characters']

        results = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'analysis_mode': 'llm_enhanced' if self.use_llm else 'deterministic',
        }

        # Run requested analyses
        if 'engagement' in components:
            results['engagement'] = self.engagement_analyzer.analyze(text)

        if 'ending' in components:
            results['ending_prediction'] = self.ending_predictor.predict(text)

        if 'genre' in components:
            results['genre'] = self.genre_classifier.classify(text)

        if 'characters' in components:
            results['characters'] = self.character_mapper.map_characters(text)

        # Add overall summary
        results['summary'] = self._generate_summary(results)

        return results

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for invalid input."""
        return {
            'text_length': 0,
            'word_count': 0,
            'analysis_mode': 'llm_enhanced' if self.use_llm else 'deterministic',
            'error': 'Empty or invalid text provided',
        }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of the analysis."""
        summary = {}

        # Engagement summary
        if 'engagement' in results:
            eng = results['engagement']
            summary['engagement_level'] = self._categorize_score(eng.get('engagement_score', 0))

        # Genre summary
        if 'genre' in results:
            gen = results['genre']
            summary['primary_genre'] = gen.get('primary_genre', 'unknown')
            summary['is_mixed_genre'] = gen.get('is_mixed_genre', False)

        # Ending summary
        if 'ending_prediction' in results:
            end = results['ending_prediction']
            summary['predicted_ending'] = end.get('predicted_ending', 'unknown')
            summary['ending_confidence'] = end.get('confidence', 0.0)

        # Character summary
        if 'characters' in results:
            char = results['characters']
            summary['character_count'] = len(char.get('characters', []))
            summary['main_characters'] = char.get('main_characters', [])

        return summary

    def _categorize_score(self, score: float) -> str:
        """Categorize a numerical score into descriptive levels."""
        if score >= 0.8:
            return 'very_high'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'moderate'
        elif score >= 0.2:
            return 'low'
        else:
            return 'very_low'

    def analyze_engagement(self, text: str) -> Dict[str, Any]:
        """Analyze only story engagement."""
        return self.engagement_analyzer.analyze(text)

    def predict_ending(self, text: str) -> Dict[str, Any]:
        """Predict only story ending."""
        return self.ending_predictor.predict(text)

    def classify_genre(self, text: str) -> Dict[str, Any]:
        """Classify only story genre."""
        return self.genre_classifier.classify(text)

    def map_characters(self, text: str) -> Dict[str, Any]:
        """Map only character relationships."""
        return self.character_mapper.map_characters(text)

    def get_character_network(self, text: str, min_mentions: int = 2) -> Dict[str, Any]:
        """Get character network visualization data."""
        return self.character_mapper.get_network_data(text, min_mentions)
