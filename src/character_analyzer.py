"""
Character Relationship Analyzer
Extracts characters and their relationships from story text using LLM.
"""

import os
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class Relationship(BaseModel):
    """Single relationship between two characters"""
    from_character: str = Field(description="First character name")
    to_character: str = Field(description="Second character name")
    relationship_type: str = Field(description="Type of relationship")
    is_blood_relation: bool = Field(description="Whether this is a blood relation")
    description: str = Field(description="Brief description of their relationship")


class CharacterMap(BaseModel):
    """Complete character relationship map"""
    characters: List[str] = Field(description="List of all character names")
    relationships: List[Relationship] = Field(description="List of relationships")


class CharacterRelationshipAnalyzer:
    """
    Analyzes story text to extract character relationships.
    Returns interactive graph data for visualization.
    """
    
    # Relationship type categories with colors
    RELATIONSHIP_COLORS = {
        # Blood relations
        'sibling': '#E74C3C',        # Bright red
        'parent': '#FF8E8E',         # Light red
        'child': '#FF8E8E',          # Light red
        'cousin': '#FFB3B3',         # Lighter red
        'grandparent': '#FFC9C9',    # Very light red
        'grandchild': '#FFC9C9',     # Very light red
        
        # Romance
        'romantic': '#EC407A',       # Bright magenta-pink (more vibrant)
        'lovers': '#FF1493',         # Deep pink
        'ex-lovers': '#C71585',      # Medium violet red
        
        # Friendship
        'friend': '#26C6DA',         # Cyan
        'best_friend': '#0097A7',    # Dark cyan
        'ally': '#00897B',           # Teal-green (distinctly different)
        
        # Family (non-blood)
        'adopted': '#FFA500',        # Orange
        'step-sibling': '#FFB347',   # Light orange
        'in-law': '#FFCC99',        # Very light orange
        
        # Antagonistic
        'enemy': '#8B0000',          # Dark red
        'rival': '#DC143C',          # Crimson
        
        # Professional
        'mentor': '#FFB300',         # Amber/Gold (completely different from purple)
        'student': '#FDD835',        # Yellow
        'colleague': '#D7BDE2',      # Very light purple
        'boss': '#6C3483',           # Dark purple
        
        # Default
        'other': '#95A5A6'           # Gray
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize analyzer with Mistral API."""
        # Load environment variables
        load_dotenv()
        
        try:
            from mistralai import Mistral
            self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
            
            if not self.api_key:
                raise ValueError("No Mistral API key found")
            
            self.client = Mistral(api_key=self.api_key)
            self.model = "mistral-large-latest"
            
            # Import guardrails
            from .guardrails import GuardrailsManager
            self.guardrails = GuardrailsManager()
            
        except ImportError:
            raise ImportError("Please install mistralai: pip install mistralai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize analyzer: {e}")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Extract character relationships from text.
        
        Args:
            text: Story text to analyze
            
        Returns:
            {
                'success': bool,
                'characters': List[str],
                'relationships': List[Dict],
                'graph_data': Dict (for visualization)
            }
        """
        prompt_id = "character_map_v1"
        
        # Pre-check
        pre_check = self.guardrails.pre_check(text, prompt_id)
        
        if pre_check['cached_response']:
            cached = pre_check['cached_response']
            return {
                'success': True,
                'characters': cached['characters'],
                'relationships': cached['relationships'],
                'graph_data': self._build_graph_data(cached),
                'source': 'cache'
            }
        
        if not pre_check['can_proceed']:
            return {
                'success': False,
                'error': pre_check['error'],
                'characters': [],
                'relationships': [],
                'graph_data': None
            }
        
        # Build prompt
        prompt = self._build_prompt(text)
        
        try:
            # Call LLM
            response_text = self._call_with_retry(prompt)
            
            # Validate
            post_check = self.guardrails.post_check(response_text, CharacterMap)
            
            if not post_check['is_valid']:
                return {
                    'success': False,
                    'error': post_check['error'],
                    'characters': [],
                    'relationships': [],
                    'graph_data': None
                }
            
            parsed: CharacterMap = post_check['parsed_data']
            
            # Convert to dict format
            result = {
                'characters': parsed.characters,
                'relationships': [
                    {
                        'from': r.from_character,
                        'to': r.to_character,
                        'type': r.relationship_type,
                        'is_blood': r.is_blood_relation,
                        'description': r.description
                    }
                    for r in parsed.relationships
                ]
            }
            
            # Cache
            self.guardrails.record_api_call()
            self.guardrails.cache_response(text, prompt_id, result)
            
            return {
                'success': True,
                'characters': result['characters'],
                'relationships': result['relationships'],
                'graph_data': self._build_graph_data(result),
                'source': 'llm'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'characters': [],
                'relationships': [],
                'graph_data': None
            }
    
    def _call_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call Mistral with retry logic."""
        import time
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(kw in error_str for kw in ['429', 'rate', 'quota'])
                
                if is_rate_limit and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 2
                    print(f"  ⏳ Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
    
    def _build_prompt(self, text: str) -> str:
        """Build character extraction prompt."""
        return f"""Extract ALL characters and their relationships from this story.

RELATIONSHIP TYPES:
Blood Relations: sibling, parent, child, cousin, grandparent, grandchild
Romance: romantic, lovers, ex-lovers
Friendship: friend, best_friend, ally
Family (non-blood): adopted, step-sibling, in-law
Antagonistic: enemy, rival
Professional: mentor, student, colleague, boss

STORY TEXT:
{text}

INSTRUCTIONS:
1. Identify all named characters (ignore unnamed like "the man")
2. For each relationship, determine the type
3. Mark if it's a blood relation (true/false)
4. Provide brief description

Return valid JSON only:
{{
  "characters": ["Name1", "Name2", "Name3"],
  "relationships": [
    {{
      "from_character": "Name1",
      "to_character": "Name2",
      "relationship_type": "<type from list above>",
      "is_blood_relation": <true or false>,
      "description": "<brief description>"
    }}
  ]
}}"""
    
    def _build_graph_data(self, result: Dict) -> Dict[str, Any]:
        """
        Convert character map to Plotly graph format.
        
        Returns:
            {
                'nodes': [...],  # For Plotly
                'edges': [...],  # For Plotly
            }
        """
        characters = result.get('characters', [])
        relationships = result.get('relationships', [])
        
        if not characters:
            return {'nodes': [], 'edges': []}
        
        # Build nodes (characters)
        nodes = []
        for i, char in enumerate(characters):
            nodes.append({
                'id': char,
                'label': char,
                'x': i,  # Will be repositioned by layout algorithm
                'y': 0,
                'size': 20
            })
        
        # Build edges (relationships)
        edges = []
        for rel in relationships:
            rel_type = rel.get('type', 'other')
            color = self.RELATIONSHIP_COLORS.get(rel_type, self.RELATIONSHIP_COLORS['other'])
            
            # Add badge if blood relation
            label = rel_type
            if rel.get('is_blood', False):
                label += " 🩸"
            
            edges.append({
                'from': rel.get('from'),
                'to': rel.get('to'),
                'label': label,
                'color': color,
                'description': rel.get('description', ''),
                'is_blood': rel.get('is_blood', False)
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.guardrails.get_stats()
