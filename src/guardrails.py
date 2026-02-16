"""
Guardrails - Simplified for Free Deployment
Rate limiting and basic validation only.
"""

import re
import json
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError


@dataclass
class CostTracker:
    """Track API usage and rate limits"""
    total_requests: int = 0
    total_tokens: int = 0
    requests_last_hour: list = field(default_factory=list)
    
    MAX_REQUESTS_PER_HOUR = 60  # Rate limit for API
    
    def can_make_request(self) -> Tuple[bool, str]:
        """Check if we're within rate limits"""
        cutoff = datetime.now() - timedelta(hours=1)
        self.requests_last_hour = [ts for ts in self.requests_last_hour if ts > cutoff]
        
        if len(self.requests_last_hour) >= self.MAX_REQUESTS_PER_HOUR:
            return False, f"Rate limit exceeded: {self.MAX_REQUESTS_PER_HOUR}/hour"
        return True, "OK"
    
    def record_request(self, tokens: int = 0):
        """Record a new API request"""
        self.total_requests += 1
        self.total_tokens += tokens
        self.requests_last_hour.append(datetime.now())


class ResponseCache:
    """LRU (least-recently-used) in-memory cache for LLM responses with TTL"""
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        from collections import OrderedDict
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}  # Track when entries were added
        self.max_size = max_size
        self.ttl_hours = ttl_hours
    
    def _hash_input(self, text: str, prompt_id: str) -> str:
        """Create cache key from input"""
        content = f"{prompt_id}:{text[:500]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired"""
        if key not in self.timestamps:
            return True
        age = datetime.now() - self.timestamps[key]
        return age > timedelta(hours=self.ttl_hours)
    
    def get(self, text: str, prompt_id: str) -> Optional[Any]:
        """Retrieve cached response (moves to end = most-recently-used)"""
        key = self._hash_input(text, prompt_id)
        
        # Check expiration
        if key in self.cache and self._is_expired(key):
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, text: str, prompt_id: str, response: Any):
        """Store response in cache, evicting LRU entry if full"""
        key = self._hash_input(text, prompt_id)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = response
        self.timestamps[key] = datetime.now()
        
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)  # Evict LRU (oldest)
            if oldest_key in self.timestamps:
                del self.timestamps[oldest_key]


class PreLLMGuardrails:
    """Guardrails applied BEFORE calling the LLM"""
    
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 5000  # Smaller for free tier
    
    SUSPICIOUS_PATTERNS = [
        r'ignore previous instructions',
        r'disregard.*rules',
        r'you are now',
        r'system:.*',
    ]
    
    def __init__(self, cost_tracker: CostTracker, cache: ResponseCache):
        self.cost_tracker = cost_tracker
        self.cache = cache
    
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """Validate input text before sending to LLM"""
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return False, f"Text too short (min {self.MIN_TEXT_LENGTH} chars)"
        if len(text) > self.MAX_TEXT_LENGTH:
            return False, f"Text too long (max {self.MAX_TEXT_LENGTH} chars)"
        
        text_lower = text.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower):
                return False, "Suspicious pattern detected"
        return True, "OK"
    
    def check_rate_limit(self) -> Tuple[bool, str]:
        """Check if we can make another API call"""
        return self.cost_tracker.can_make_request()
    
    def check_cache(self, text: str, prompt_id: str) -> Optional[Any]:
        """Check if we have a cached response"""
        return self.cache.get(text, prompt_id)


class PostLLMGuardrails:
    """Guardrails applied AFTER receiving LLM response"""
    
    def validate_json_schema(
        self, 
        response: str, 
        schema_model: BaseModel
    ) -> Tuple[bool, Optional[BaseModel], str]:
        """Validate LLM response against Pydantic schema"""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            model = schema_model(**data)
            return True, model, "OK"
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except ValidationError as e:
            return False, None, f"Schema validation failed: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    def extract_partial_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract partial valid data from malformed response.
        Used for graceful degradation when full parsing fails.
        """
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            # Extract any valid fields we can find
            partial = {}
            
            # Common fields across all response types
            if 'score' in data and isinstance(data['score'], (int, float)):
                partial['score'] = float(data['score'])
            if 'confidence' in data and isinstance(data['confidence'], (int, float)):
                partial['confidence'] = float(data['confidence'])
            if 'reasoning' in data and isinstance(data['reasoning'], str):
                partial['reasoning'] = data['reasoning']
            if 'is_interesting' in data and isinstance(data['is_interesting'], bool):
                partial['is_interesting'] = data['is_interesting']
            if 'is_ending' in data and isinstance(data['is_ending'], bool):
                partial['is_ending'] = data['is_ending']
            if 'probability' in data and isinstance(data['probability'], (int, float)):
                partial['probability'] = float(data['probability'])
            if 'action' in data and isinstance(data['action'], str):
                partial['action'] = data['action']
            
            return partial if partial else None
        except:
            return None
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from response text, handling various formats.
        Works with: plain JSON, markdown code blocks, text with JSON embedded
        """
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'```', '', text)
        text = text.strip()
        
        # Try to find JSON object if text contains extra content
        # Look for content between first { and last }
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            potential_json = json_match.group(0)
            # Test if it's valid JSON
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                # Try to repair truncated JSON
                repaired = self._repair_json(potential_json)
                if repaired:
                    return repaired
        
        # Try repairing the raw text as a last resort
        repaired = self._repair_json(text)
        if repaired:
            return repaired
        
        # Return as-is if no extraction worked
        return text
    
    def _repair_json(self, text: str) -> Optional[str]:
        """
        Attempt to repair truncated/malformed JSON by closing
        unterminated strings, arrays, and objects.
        """
        try:
            json.loads(text)
            return text  # Already valid
        except json.JSONDecodeError:
            pass
        
        repaired = text.rstrip()
        
        # Close unterminated string (odd number of unescaped quotes)
        quote_count = 0
        i = 0
        while i < len(repaired):
            if repaired[i] == '"' and (i == 0 or repaired[i-1] != '\\'):
                quote_count += 1
            i += 1
        if quote_count % 2 == 1:
            repaired += '"'
        
        # Close any open arrays and objects
        open_brackets = repaired.count('[') - repaired.count(']')
        open_braces = repaired.count('{') - repaired.count('}')
        
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)
        
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None


class GuardrailsManager:
    """Main interface for all guardrail operations"""
    
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.cache = ResponseCache()
        self.pre_llm = PreLLMGuardrails(self.cost_tracker, self.cache)
        self.post_llm = PostLLMGuardrails()
    
    def pre_check(self, text: str, prompt_id: str) -> Dict[str, Any]:
        """Run all pre-LLM checks"""
        is_valid, error = self.pre_llm.validate_input(text)
        if not is_valid:
            return {'can_proceed': False, 'cached_response': None, 'error': error}
        
        can_call, error = self.pre_llm.check_rate_limit()
        if not can_call:
            return {'can_proceed': False, 'cached_response': None, 'error': error}
        
        cached = self.pre_llm.check_cache(text, prompt_id)
        if cached:
            return {'can_proceed': False, 'cached_response': cached, 'error': None}
        
        return {'can_proceed': True, 'cached_response': None, 'error': None}
    
    def post_check(
        self, 
        response: str, 
        schema_model: BaseModel
    ) -> Dict[str, Any]:
        """Run all post-LLM checks"""
        is_valid, parsed, error = self.post_llm.validate_json_schema(response, schema_model)
        
        if not is_valid:
            # Try partial extraction for graceful degradation
            partial_data = self.post_llm.extract_partial_response(response)
            return {
                'is_valid': False, 
                'parsed_data': None, 
                'error': error,
                'partial_data': partial_data
            }
        
        return {'is_valid': True, 'parsed_data': parsed, 'error': None, 'partial_data': None}
    
    def record_api_call(self, tokens: int = 0):
        """Record successful API call"""
        self.cost_tracker.record_request(tokens)
    
    def cache_response(self, text: str, prompt_id: str, response: Any):
        """Cache a response for future use"""
        self.cache.set(text, prompt_id, response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            'total_calls': self.cost_tracker.total_requests,
            'successful_calls': self.cost_tracker.total_requests,  # Successful ones tracked
            'failed_calls': 0,  # Not tracked separately yet
            'total_requests': self.cost_tracker.total_requests,
            'total_tokens': self.cost_tracker.total_tokens,
            'cache_size': len(self.cache.cache),
            'rate_limit': f"{len(self.cost_tracker.requests_last_hour)}/{self.cost_tracker.MAX_REQUESTS_PER_HOUR}"
        }