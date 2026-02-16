"""
LLM Analyzer - IMPROVED VERSION
Better prompts with examples and genre awareness
Uses Mistral AI API
"""

import os
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mistralai import Mistral


# Response schemas (unchanged)
class EngagementAnalysis(BaseModel):
    score: float = Field(ge=0, le=10, description="Engagement score 0-10")
    is_interesting: bool = Field(description="Binary interesting decision")
    confidence: float = Field(ge=0, le=1, description="Confidence in assessment")
    reasoning: str = Field(description="Explanation for the score")
    key_factors: List[str] = Field(description="List of engagement factors detected")


class EndingAnalysis(BaseModel):
    is_ending: bool = Field(description="Whether story is reaching its ending")
    probability: float = Field(ge=0, le=1, description="Probability story is ending")
    confidence: float = Field(ge=0, le=1, description="Confidence in assessment")
    reasoning: str = Field(description="Explanation for the decision")
    ending_markers: List[str] = Field(description="Detected ending indicators")


class ContinuationRecommendation(BaseModel):
    action: str = Field(description="CONTINUE, CONCLUDE, or AMBIGUOUS")
    confidence: float = Field(ge=0, le=1, description="Confidence in recommendation")
    reasoning: str = Field(description="Explanation for recommendation")
    considerations: List[str] = Field(description="Key factors in decision")


class LLMAnalyzer:
    """
    IMPROVED semantic analysis with:
    1. Better prompts with examples
    2. Genre-aware analysis
    3. Clearer error messages
    """
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        # Load environment variables from .env file
        load_dotenv()
        
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Mistral API key found. Please either:\n"
                "  1. Set MISTRAL_API_KEY environment variable, OR\n"
                "  2. Pass api_key parameter to LLMAnalyzer()\n"
                "Get your API key at: https://console.mistral.ai/"
            )
        
        # Configure Mistral client
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-large-latest"
        
        # Import guardrails
        from src.guardrails import GuardrailsManager
        self.guardrails = GuardrailsManager()
        self.use_cache = use_cache
        
        # Generation config
        self.generation_config = None  # Mistral uses different parameters
    
    def _call_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call Mistral with exponential backoff for rate limit handling."""
        import time as _time
        
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
                is_rate_limit = any(kw in error_str for kw in ['429', 'resource_exhausted', 'quota', 'rate'])
                
                if is_rate_limit and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 2  # 2s, 4s, 8s
                    print(f"  ⏳ Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    _time.sleep(wait)
                else:
                    raise
    
    def analyze_engagement(
        self, 
        text: str, 
        genre_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze story engagement with genre awareness and error recovery."""
        prompt_id = "engagement_v2"
        
        # Pre-LLM guardrails
        pre_check = self.guardrails.pre_check(text, prompt_id)
        
        if pre_check['cached_response'] and self.use_cache:
            return {
                'success': True,
                'data': pre_check['cached_response'],
                'source': 'cache'
            }
        
        if not pre_check['can_proceed']:
            return {
                'success': False,
                'error': pre_check['error'],
                'data': None
            }
        
        # Try full prompt first
        prompt = self._build_engagement_prompt_v2(text, genre_hint)
        
        try:
            response_text = self._call_with_retry(prompt)
            post_check = self.guardrails.post_check(response_text, EngagementAnalysis)
            
            if post_check['is_valid']:
                parsed: EngagementAnalysis = post_check['parsed_data']
                result = {
                    'score': parsed.score,
                    'is_interesting': parsed.is_interesting,
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning,
                    'key_factors': parsed.key_factors
                }
                
                self.guardrails.record_api_call()
                if self.use_cache:
                    self.guardrails.cache_response(text, prompt_id, result)
                
                return {'success': True, 'data': result, 'source': 'mistral'}
            
            # FALLBACK 1: Try simplified prompt
            print("  ⚠️  Full prompt failed, trying simplified version...")
            simple_prompt = self._build_engagement_prompt_simple(text)
            
            try:
                response_text = self._call_with_retry(simple_prompt, max_retries=2)
                post_check = self.guardrails.post_check(response_text, EngagementAnalysis)
                
                if post_check['is_valid']:
                    parsed: EngagementAnalysis = post_check['parsed_data']
                    result = {
                        'score': parsed.score,
                        'is_interesting': parsed.is_interesting,
                        'confidence': max(0.0, parsed.confidence - 0.1),  # Lower confidence
                        'reasoning': parsed.reasoning + " (simplified analysis)",
                        'key_factors': parsed.key_factors
                    }
                    
                    self.guardrails.record_api_call()
                    return {'success': True, 'data': result, 'source': 'mistral-simplified'}
                
                # FALLBACK 2: Use partial data if available
                if post_check['partial_data']:
                    partial = post_check['partial_data']
                    if 'score' in partial:
                        result = {
                            'score': partial.get('score', 5.0),
                            'is_interesting': partial.get('is_interesting', partial.get('score', 5.0) > 5.0),
                            'confidence': max(0.3, partial.get('confidence', 0.5) - 0.2),
                            'reasoning': partial.get('reasoning', 'Partial LLM analysis (degraded mode)'),
                            'key_factors': ['partial_response']
                        }
                        return {'success': True, 'data': result, 'source': 'mistral-partial'}
            
            except Exception as e:
                print(f"  ⚠️  Simplified prompt also failed: {str(e)}")
            
            # FALLBACK 3: Return error with suggestion
            return {
                'success': False,
                'error': f"LLM analysis failed after retries. Consider using heuristic-only mode.",
                'data': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Mistral API call failed: {str(e)}",
                'data': None
            }
    
    def analyze_ending(self, text: str) -> Dict[str, Any]:
        """Analyze if story is reaching its ending with error recovery."""
        prompt_id = "ending_v2"
        
        pre_check = self.guardrails.pre_check(text, prompt_id)
        
        if pre_check['cached_response'] and self.use_cache:
            return {'success': True, 'data': pre_check['cached_response'], 'source': 'cache'}
        
        if not pre_check['can_proceed']:
            return {'success': False, 'error': pre_check['error'], 'data': None}
        
        prompt = self._build_ending_prompt_v2(text)
        
        try:
            response_text = self._call_with_retry(prompt)
            post_check = self.guardrails.post_check(response_text, EndingAnalysis)
            
            if post_check['is_valid']:
                parsed: EndingAnalysis = post_check['parsed_data']
                result = {
                    'is_ending': parsed.is_ending,
                    'probability': parsed.probability,
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning,
                    'ending_markers': parsed.ending_markers
                }
                
                self.guardrails.record_api_call()
                if self.use_cache:
                    self.guardrails.cache_response(text, prompt_id, result)
                
                return {'success': True, 'data': result, 'source': 'mistral'}
            
            # FALLBACK 1: Simplified prompt
            print("  ⚠️  Full ending prompt failed, trying simplified...")
            simple_prompt = self._build_ending_prompt_simple(text)
            
            try:
                response_text = self._call_with_retry(simple_prompt, max_retries=2)
                post_check = self.guardrails.post_check(response_text, EndingAnalysis)
                
                if post_check['is_valid']:
                    parsed: EndingAnalysis = post_check['parsed_data']
                    result = {
                        'is_ending': parsed.is_ending,
                        'probability': parsed.probability,
                        'confidence': max(0.0, parsed.confidence - 0.1),
                        'reasoning': parsed.reasoning + " (simplified)",
                        'ending_markers': parsed.ending_markers
                    }
                    return {'success': True, 'data': result, 'source': 'mistral-simplified'}
                
                # FALLBACK 2: Partial data
                if post_check['partial_data']:
                    partial = post_check['partial_data']
                    if 'probability' in partial or 'is_ending' in partial:
                        prob = partial.get('probability', 0.5 if partial.get('is_ending', False) else 0.3)
                        result = {
                            'is_ending': partial.get('is_ending', prob > 0.5),
                            'probability': prob,
                            'confidence': max(0.3, partial.get('confidence', 0.5) - 0.2),
                            'reasoning': partial.get('reasoning', 'Partial analysis (degraded)'),
                            'ending_markers': ['partial_response']
                        }
                        return {'success': True, 'data': result, 'source': 'mistral-partial'}
            
            except Exception as e:
                print(f"  ⚠️  Simplified ending prompt failed: {str(e)}")
            
            return {'success': False, 'error': 'Ending analysis failed after retries', 'data': None}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'data': None}
    
    def recommend_continuation(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate continuation recommendation with error recovery."""
        prompt_id = "continuation_v2"
        
        pre_check = self.guardrails.pre_check(text, prompt_id)
        
        if pre_check['cached_response'] and self.use_cache:
            return {'success': True, 'data': pre_check['cached_response'], 'source': 'cache'}
        
        if not pre_check['can_proceed']:
            return {'success': False, 'error': pre_check['error'], 'data': None}
        
        prompt = self._build_continuation_prompt_v2(text, context)
        
        try:
            response_text = self._call_with_retry(prompt)
            post_check = self.guardrails.post_check(response_text, ContinuationRecommendation)
            
            if post_check['is_valid']:
                parsed: ContinuationRecommendation = post_check['parsed_data']
                result = {
                    'action': parsed.action,
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning,
                    'considerations': parsed.considerations
                }
                
                self.guardrails.record_api_call()
                if self.use_cache:
                    self.guardrails.cache_response(text, prompt_id, result)
                
                return {'success': True, 'data': result, 'source': 'mistral'}
            
            # FALLBACK 1: Simplified prompt
            print("  ⚠️  Full continuation prompt failed, trying simplified...")
            simple_prompt = self._build_continuation_prompt_simple(text)
            
            try:
                response_text = self._call_with_retry(simple_prompt, max_retries=2)
                post_check = self.guardrails.post_check(response_text, ContinuationRecommendation)
                
                if post_check['is_valid']:
                    parsed: ContinuationRecommendation = post_check['parsed_data']
                    result = {
                        'action': parsed.action,
                        'confidence': max(0.0, parsed.confidence - 0.1),
                        'reasoning': parsed.reasoning + " (simplified)",
                        'considerations': parsed.considerations
                    }
                    return {'success': True, 'data': result, 'source': 'mistral-simplified'}
                
                # FALLBACK 2: Partial data
                if post_check['partial_data']:
                    partial = post_check['partial_data']
                    if 'action' in partial:
                        result = {
                            'action': partial.get('action', 'AMBIGUOUS'),
                            'confidence': max(0.3, partial.get('confidence', 0.5) - 0.2),
                            'reasoning': partial.get('reasoning', 'Partial recommendation'),
                            'considerations': ['partial_response']
                        }
                        return {'success': True, 'data': result, 'source': 'mistral-partial'}
            
            except Exception as e:
                print(f"  ⚠️  Simplified continuation prompt failed: {str(e)}")
            
            return {'success': False, 'error': 'Continuation analysis failed after retries', 'data': None}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'data': None}
    
    def _build_engagement_prompt_simple(self, text: str) -> str:
        """Simplified prompt for fallback when full prompt fails."""
        return f"""Rate this story segment's engagement level from 0-10.

Story: {text}

Return ONLY valid JSON:
{{
  "score": <0-10>,
  "is_interesting": <true/false>,
  "confidence": <0-1>,
  "reasoning": "<brief explanation>",
  "key_factors": ["<factor1>", "<factor2>"]
}}"""
    
    def _build_engagement_prompt_v2(self, text: str, genre_hint: Optional[str] = None) -> str:
        """
        Optimized prompt: shorter, clearer, with chain-of-thought reasoning.
        """
        genre_context = f" (Genre: {genre_hint})" if genre_hint else ""
        
        return f"""Rate story engagement 0-10. Consider ALL sources: action, psychological tension, mystery, emotion, stakes, pacing{genre_context}.

CALIBRATION:
8/10: "Who are you?" he typed. "Someone you already chose not to be." [mystery + existential stakes]
8/10: "Get down!" Bullets tore through the wall. She dove. [action + danger]
5/10: She sat by the window, watching rain, thinking about yesterday. [reflective, low stakes]
2/10: He walked to the store on Main Street. He bought milk. [no tension]

ANALYZE: {text}

THINK STEP-BY-STEP:
1. What engagement sources are present?
2. How strong is each (action, tension, mystery, emotion, stakes, pacing)?
3. Final score considering all factors

Return valid JSON only:
{{"score": <0-10>, "is_interesting": <bool>, "confidence": <0-1>, "reasoning": "<2-3 sentences>", "key_factors": ["<factor>", ...]}}"""
    
    def _build_ending_prompt_simple(self, text: str) -> str:
        """Simplified ending prompt for fallback."""
        return f"""Is this story reaching its ending? Respond with probability 0-1.

Story: {text}

Return ONLY valid JSON:
{{
  "is_ending": <true/false>,
  "probability": <0-1>,
  "confidence": <0-1>,
  "reasoning": "<brief explanation>",
  "ending_markers": ["<marker1>", "<marker2>"]
}}"""
    
    def _build_ending_prompt_v2(self, text: str) -> str:
        """Optimized ending detection with chain-of-thought."""
        return f"""Is this story reaching its ENDING?

ENDING SIGNALS: Resolution, closure, reflection, arc completion, loose ends tied, falling tension
CONTINUING SIGNALS: New questions, escalating conflict, rising action, unresolved struggle

EXAMPLES:
0.85: "She closed the journal. Outside, the sun rose. It was over. Finally, completely, over." [resolution + closure]
0.15: "The message: 'This is just the beginning.' Her stomach tightened." [new mystery + rising tension]

ANALYZE: {text}

THINK:
1. What conflicts remain unresolved?
2. Is tension rising or falling?
3. Are loose ends being tied or new threads introduced?
4. What's the probability this is an ending (0-1)?

Return valid JSON only:
{{"is_ending": <bool>, "probability": <0-1>, "confidence": <0-1>, "reasoning": "<explanation>", "ending_markers": ["<marker>", ...]}}"""
    
    def _build_continuation_prompt_simple(self, text: str) -> str:
        """Simplified continuation prompt for fallback."""
        return f"""Should this story CONTINUE or CONCLUDE?

Story: {text}

Return ONLY valid JSON:
{{
  "action": "<CONTINUE or CONCLUDE or AMBIGUOUS>",
  "confidence": <0-1>,
  "reasoning": "<brief explanation>",
  "considerations": ["<factor1>", "<factor2>"]
}}"""
    
    def _build_continuation_prompt_v2(self, text: str, context: Optional[Dict]) -> str:
        """Optimized continuation recommendation with context awareness."""
        
        # Build compact context string
        context_hints = ""
        if context:
            eng_score = context.get('engagement_score', 'unknown')
            ending_prob = context.get('ending_probability', 'unknown')
            if eng_score != 'unknown' or ending_prob != 'unknown':
                context_hints = f" [Engagement: {eng_score}, Ending prob: {ending_prob}]"
        
        return f"""Should this story CONTINUE, CONCLUDE, or is it AMBIGUOUS?{context_hints}

CRITERIA:
- CONTINUE: High engagement + unresolved conflicts + momentum
- CONCLUDE: Low engagement OR natural ending OR diminishing returns
- AMBIGUOUS: Mixed signals

ANALYZE: {text}

THINK:
1. Current engagement level?
2. Story momentum (rising/falling)?
3. Conflicts resolved or ongoing?
4. Recommendation?

Return valid JSON only:
{{"action": "<CONTINUE or CONCLUDE or AMBIGUOUS>", "confidence": <0-1>, "reasoning": "<explanation>", "considerations": ["<factor>", ...]}}"""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        return self.guardrails.get_stats()