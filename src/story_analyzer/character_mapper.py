"""
Character Mapper Module

Maps character relationships using network analysis:
- Character co-occurrence detection
- Relationship type inference
- Character importance ranking
- Social network visualization data
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import networkx as nx


class CharacterMapper:
    """Maps character relationships and generates network analysis."""

    # Common pronouns to filter out
    PRONOUNS = {
        'he', 'she', 'they', 'him', 'her', 'his', 'hers', 'their', 'theirs',
        'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours'
    }

    # Relationship indicator words
    RELATIONSHIP_INDICATORS = {
        'family': {'mother', 'father', 'sister', 'brother', 'son', 'daughter', 'parent', 'child', 'family'},
        'friend': {'friend', 'buddy', 'companion', 'ally', 'partner'},
        'romantic': {'love', 'lover', 'husband', 'wife', 'girlfriend', 'boyfriend', 'romance'},
        'enemy': {'enemy', 'rival', 'foe', 'opponent', 'adversary', 'hate'},
        'professional': {'colleague', 'boss', 'employee', 'coworker', 'mentor', 'student'},
    }

    def __init__(self):
        """Initialize the character mapper."""
        pass

    def map_characters(self, text: str, min_mentions: int = 2) -> Dict[str, any]:
        """
        Map characters and their relationships in the story.

        Args:
            text: The story text to analyze
            min_mentions: Minimum number of mentions to consider a character

        Returns:
            Dictionary containing character network and relationship data
        """
        if not text or not text.strip():
            return self._empty_result()

        # Extract potential character names
        characters = self._extract_characters(text, min_mentions)
        
        if not characters:
            return self._empty_result()

        # Build relationship graph
        graph = self._build_relationship_graph(text, characters)
        
        # Analyze the network
        return {
            'characters': list(characters.keys()),
            'character_mentions': characters,
            'relationships': self._extract_relationships(graph),
            'main_characters': self._identify_main_characters(characters, graph),
            'character_importance': self._calculate_importance(characters, graph),
            'network_density': self._calculate_network_density(graph),
            'clusters': self._identify_clusters(graph),
        }

    def _empty_result(self) -> Dict[str, any]:
        """Return empty result for invalid input."""
        return {
            'characters': [],
            'character_mentions': {},
            'relationships': [],
            'main_characters': [],
            'character_importance': {},
            'network_density': 0.0,
            'clusters': [],
        }

    def _extract_characters(self, text: str, min_mentions: int) -> Dict[str, int]:
        """
        Extract character names from text.
        Uses capitalized words as potential names.
        """
        # Find all capitalized words (potential names)
        # This is a simple heuristic - could be improved with NER
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Count occurrences
        name_counts = Counter(potential_names)
        
        # Filter by minimum mentions and remove common words
        common_words = {
            'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'He', 'She',
            'It', 'They', 'We', 'I', 'You', 'Chapter', 'Part'
        }
        
        characters = {
            name: count 
            for name, count in name_counts.items() 
            if count >= min_mentions and name not in common_words
        }
        
        return characters

    def _build_relationship_graph(self, text: str, characters: Dict[str, int]) -> nx.Graph:
        """
        Build a graph representing character relationships.
        Characters appearing in the same sentence are considered related.
        """
        graph = nx.Graph()
        
        # Add all characters as nodes
        for character in characters:
            graph.add_node(character, mentions=characters[character])
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Find co-occurrences within sentences
        for sentence in sentences:
            # Find which characters appear in this sentence
            chars_in_sentence = [char for char in characters if char in sentence]
            
            # Add edges for all pairs
            for i, char1 in enumerate(chars_in_sentence):
                for char2 in chars_in_sentence[i+1:]:
                    if graph.has_edge(char1, char2):
                        # Increment weight if edge exists
                        graph[char1][char2]['weight'] += 1
                    else:
                        # Create new edge
                        graph.add_edge(char1, char2, weight=1)
        
        return graph

    def _extract_relationships(self, graph: nx.Graph) -> List[Dict[str, any]]:
        """Extract relationship information from the graph."""
        relationships = []
        
        for edge in graph.edges(data=True):
            char1, char2, data = edge
            relationships.append({
                'character1': char1,
                'character2': char2,
                'strength': data['weight'],
                'type': 'associated',  # Default type
            })
        
        # Sort by strength
        relationships.sort(key=lambda x: x['strength'], reverse=True)
        
        return relationships

    def _identify_main_characters(self, characters: Dict[str, int], graph: nx.Graph) -> List[str]:
        """
        Identify main characters based on mentions and connections.
        """
        if not characters:
            return []
        
        # Calculate a score combining mentions and connections
        scores = {}
        for char in characters:
            mention_score = characters[char]
            connection_score = graph.degree(char) if char in graph else 0
            scores[char] = mention_score + connection_score * 2
        
        # Sort by score and take top characters
        sorted_chars = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 or top 30% of characters, whichever is smaller
        top_count = min(5, max(1, len(characters) // 3))
        return [char for char, _ in sorted_chars[:top_count]]

    def _calculate_importance(self, characters: Dict[str, int], graph: nx.Graph) -> Dict[str, float]:
        """
        Calculate importance score for each character.
        Combines mention frequency with network centrality.
        """
        importance = {}
        
        # Calculate centrality if graph has edges
        if graph.number_of_edges() > 0:
            try:
                centrality = nx.degree_centrality(graph)
            except:
                centrality = {char: 0 for char in characters}
        else:
            centrality = {char: 0 for char in characters}
        
        # Normalize mention counts
        max_mentions = max(characters.values()) if characters else 1
        
        for char in characters:
            mention_score = characters[char] / max_mentions
            centrality_score = centrality.get(char, 0)
            
            # Weighted combination
            importance[char] = round(0.6 * mention_score + 0.4 * centrality_score, 3)
        
        return importance

    def _calculate_network_density(self, graph: nx.Graph) -> float:
        """Calculate the density of the character network."""
        if graph.number_of_nodes() < 2:
            return 0.0
        
        try:
            density = nx.density(graph)
            return round(density, 3)
        except:
            return 0.0

    def _identify_clusters(self, graph: nx.Graph) -> List[List[str]]:
        """
        Identify clusters/communities of characters.
        Returns groups of characters that are closely connected.
        """
        if graph.number_of_nodes() < 2:
            return []
        
        try:
            # Use connected components for simple clustering
            components = list(nx.connected_components(graph))
            
            # Only return clusters with more than 1 character
            clusters = [list(component) for component in components if len(component) > 1]
            
            # Sort clusters by size
            clusters.sort(key=len, reverse=True)
            
            return clusters
        except:
            return []

    def get_network_data(self, text: str, min_mentions: int = 2) -> Dict[str, any]:
        """
        Get network data in a format suitable for visualization.

        Args:
            text: The story text to analyze
            min_mentions: Minimum number of mentions to consider a character

        Returns:
            Dictionary with nodes and edges for network visualization
        """
        characters = self._extract_characters(text, min_mentions)
        graph = self._build_relationship_graph(text, characters)
        
        # Prepare node data
        nodes = []
        for char in graph.nodes():
            nodes.append({
                'id': char,
                'label': char,
                'mentions': characters.get(char, 0),
                'connections': graph.degree(char),
            })
        
        # Prepare edge data
        edges = []
        for char1, char2, data in graph.edges(data=True):
            edges.append({
                'source': char1,
                'target': char2,
                'weight': data['weight'],
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
        }
