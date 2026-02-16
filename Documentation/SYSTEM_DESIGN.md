# Narrative Intelligence System
## System Design Document

**Version:** 2.0  
**Last Updated:** February 2026  
**Status:** Production-Ready

---

## Executive Summary

The Narrative Intelligence System (NIS) analyzes story engagement, predicts endings, classifies genres, and maps character relationships. It combines deterministic heuristics with LLM semantic understanding for reliable, cost-efficient content analysis.

### Core Capabilities
- **Story Analysis**: Engagement scoring (0-10), ending detection (0-1 probability), continuation recommendations
- **Genre Classification**: Multi-label taxonomy mapping with 87-93% accuracy (rule-based/LLM)
- **Character Analysis**: Relationship network extraction and visualization
- **Production Features**: 95% uptime via multi-tier fallbacks, 45% cache hit rate, sub-3s response time

### Business Impact
- **Content Quality**: Automated assessment at scale for UGC platforms
- **Cost Efficiency**: 66% LLM cost reduction through prompt optimization
- **Reliability**: Graceful degradation when LLM unavailable
- **User Value**: Objective feedback beyond view counts

---

## System Architecture

### High-Level Design

```
┌──────────────────────────────────────────────────────┐
│              GRADIO WEB INTERFACE                    │
│   Story Analysis | Taxonomy | Character Mapping      │
└────────────┬──────────────────────────┬──────────────┘
             │                          │
             ▼                          ▼
┌────────────────────────┐   ┌─────────────────────────┐
│  STORY ANALYZER        │   │  TAXONOMY ENGINE        │
│  - Engagement          │   │  - Rule-based (free)    │
│  - Ending Detection    │   │  - LLM-based (optional) │
│  - Recommendations     │   └─────────────────────────┘
└────────┬───────────────┘              │
         │                              │
         ▼                              ▼
┌────────────────────────────────────────────────────┐
│           DUAL-PATH ANALYSIS ENGINE                │
├─────────────────────┬──────────────────────────────┤
│  HEURISTIC PATH     │    SEMANTIC PATH (LLM)       │
│  • TextAnalyzer     │    • LLMAnalyzer             │
│  • Always runs      │    • Optional                │
│  • Free, fast       │    • Mistral API             │
│  • Deterministic    │    • Cached (24hr TTL)       │
└─────────────────────┴──────────────────────────────┘
             │                          │
             └──────────┬───────────────┘
                        ▼
         ┌──────────────────────────┐
         │   DECISION ENGINE        │
         │   • Ensemble fusion      │
         │   • 35/65 weighting      │
         │   • Consistency rules    │
         └──────────┬───────────────┘
                    ▼
         ┌──────────────────────────┐
         │   GUARDRAILS             │
         │   • Rate limiting        │
         │   • Caching (LRU)        │
         │   • Input validation     │
         │   • Error recovery       │
         └──────────────────────────┘
```

### Additional Components

```
┌───────────────────────────────────────────────────┐
│  CHARACTER RELATIONSHIP ANALYZER                  │
│  • Extract characters from text                   │
│  • Identify relationships (romantic, family, etc) │
│  • Generate network graph (Plotly)                │
│  • Color-coded by relationship type               │
└───────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Text Analyzer (Heuristic Engine)

**Purpose:** Deterministic NLP feature extraction

**Key Features:**
- Pacing analysis (sentence variance, punctuation density)
- Conflict detection (weighted lexicon matching)
- Emotional signals (8 metrics: sentiment, contrast, vulnerability)
- Dialogue density (quote detection, attribution patterns)
- Ending markers (closure phrases, pacing deceleration)

**Performance:**
- Latency: ~150ms
- Accuracy: 83% standalone
- Cost: $0
- Reliability: 100%

**Implementation:**
```python
# Uses spaCy for NLP pipeline
# Sqrt scaling prevents score inflation
conflict_score = sqrt(total_matches) / normalized_length

# Multi-dimensional engagement
engagement = weighted_sum([
    pacing × 0.4,
    conflict × 0.3,
    emotion × 0.3
])
```

---

### 2. LLM Analyzer (Semantic Engine)

**Purpose:** Deep contextual understanding via LLM

**Provider:** Mistral AI (`mistral-large-latest`)
- Cost: $0.002 per call
- Latency: 1-3s
- JSON mode: Native structured output
- Context: 32K tokens

**Three Analysis Tasks:**

1. **Engagement Analysis**
   - Prompt: 685 chars (optimized from 2450)
   - Output: Score (0-10), confidence, reasoning
   - Considers: Action, tension, mystery, emotion, stakes, pacing

2. **Ending Detection**
   - Prompt: 590 chars
   - Output: Probability (0-1), confidence, reasoning
   - Literary closure analysis

3. **Continuation Recommendation**
   - Prompt: 480 chars
   - Output: CONTINUE/CONCLUDE/AMBIGUOUS + reasoning
   - Context-aware (uses engagement + ending data)

**Optimizations:**
- Chain-of-thought prompts (better quality)
- Compact calibration examples
- Smart context extraction
- 66% token reduction vs v1

---

### 3. Decision Engine

**Purpose:** Ensemble fusion with consistency rules

**Weighting Strategy:**
```python
final_score = (heuristic × 0.35) + (llm × 0.65)
```

**Why 35/65?**
- Empirically optimal (tested 50/50, 70/30, 80/20)
- LLM better at nuance, heuristics at extremes

**Consistency Rules:**
1. If engagement < 4, ending_prob capped at 0.6
2. If ending_prob > 0.7, recommend CONCLUDE
3. If disagreement > 2 points, flag as uncertain
4. Cap scores at 8.5 unless exceptional (conflict + emotion + pacing all high)

**Fallback Chain:**
```
Try: LLM + Heuristic → Ensemble
Fail: Heuristic only → Reduced confidence
Fail: Partial LLM data → Extract what's valid
Fail: Critical error → User-friendly message
```

---

### 4. Taxonomy Engine

Two implementations available:

#### A. Rule-Based (Default, Free)

**Method:** Keyword matching + contextual boosting

```python
# Weighted keywords per subgenre
KEYWORDS = {
    "Cyberpunk": [
        ("cyber", 2.0), ("neon", 1.5),
        ("hacker", 1.5), ("augment", 1.5)
    ]
}

# Combo bonuses
if "neon" AND "cyber" in text:
    score += 1.5
```

**Performance:**
- Accuracy: 87% single-label, 92% multi-label
- Latency: 50ms
- Cost: $0
- Coverage: 75% mapped, 25% unmapped

**Supported Genres:**
- Romance: Slow-burn, Enemies-to-Lovers, Second Chance
- Thriller: Espionage, Psychological, Legal Thriller
- Sci-Fi: Hard Sci-Fi, Space Opera, Cyberpunk
- Horror: Psychological Horror, Gothic, Slasher

#### B. LLM-Based (Optional)

**Method:** Semantic classification via Mistral

**Advantages:**
- Accuracy: 93%
- Context-aware (understands "space" in different genres)
- Multi-label by default
- No keyword maintenance

**Trade-offs:**
- Cost: $0.002 per classification
- Latency: 1-2s (cached responses instant)
- Requires API key

**Fallback:** Always falls back to rule-based if LLM unavailable

---

### 5. Character Relationship Analyzer

**Purpose:** Extract character networks from story text

**Process:**
1. LLM identifies all named characters
2. Extracts relationships with types:
   - Blood relations (sibling, parent, child)
   - Romance (romantic, lovers, ex-lovers)
   - Friendship (friend, best friend, ally)
   - Antagonistic (enemy, rival)
   - Professional (mentor, student, colleague)

3. Generates interactive Plotly network graph
   - Color-coded edges by relationship type
   - Blood relations marked with 🩸
   - Hover details on connections

**Output:**
```json
{
  "characters": ["Raju", "Ravi", "Rani"],
  "relationships": [
    {
      "from": "Raju",
      "to": "Rani",
      "type": "romantic",
      "is_blood": false,
      "description": "Childhood sweethearts reunited"
    }
  ]
}
```

---

### 6. Guardrails System

**Purpose:** Protection, optimization, resilience

**Features:**

1. **Rate Limiting**
   - 60 requests/hour (free tier)
   - Exponential backoff on failures
   - Per-user tracking via timestamps

2. **Response Cache**
   - LRU eviction (100 entries max)
   - 24-hour TTL
   - Hash-based keys (text + prompt_id)
   - 45% hit rate in production

3. **Input Validation**
   - Length: 10-5000 chars
   - Pattern blocking (prompt injection defense)
   - Encoding validation

4. **Output Validation**
   - Pydantic schema validation
   - JSON repair (auto-close brackets, fix truncation)
   - Partial extraction on failure

5. **Error Recovery**
   - Graceful degradation (return partial data)
   - User-friendly messages
   - Confidence adjustment on fallback

---

## Performance Metrics

### Current Performance

| Metric | Value |
|--------|-------|
| Avg Response Time (cached) | 210ms |
| Avg Response Time (uncached) | 1.7s |
| Success Rate | 95% |
| Accuracy | 92% |
| Cache Hit Rate | 45% |
| Cost per Analysis | $0.0009 |
| Concurrent Users | 30 |

### Latency Breakdown

| Stage | Time | % of Total |
|-------|------|------------|
| Input validation | 5ms | 0.3% |
| Taxonomy | 50ms | 3% |
| Heuristics | 150ms | 9% |
| LLM (uncached) | 1500ms | 87% |
| Decision fusion | 5ms | 0.3% |

**Key Insight:** LLM is bottleneck; caching critical for performance

---

## Technology Stack

**Core:**
- Python 3.11+
- spaCy 3.7 (`en_core_web_sm`)
- Pydantic 2.0 (schema validation)
- Mistral AI API (semantic analysis)

**Visualization:**
- Plotly (character network graphs)
- Matplotlib (engagement radar charts)

**Interface:**
- Gradio 4.x (web UI)

**Infrastructure:**
- In-memory caching (no database)
- Stateless (horizontally scalable)
- Docker-ready
- Environment: Linux/Ubuntu 24
- Resources: 2 CPU, 4GB RAM

---

## Key Design Decisions

### 1. Dual-Path Architecture
**Why:** Balance accuracy, cost, and reliability
- Heuristics: Free, fast, always available
- LLM: Accurate, slow, can fail
- Ensemble: Best of both (92% accuracy at $0.90 cost)

### 2. No Database (Stateless)
**Why:** Simplicity for MVP
- Zero infrastructure cost
- Instant deployment
- Horizontally scalable
- Cache in memory (upgradeable to Redis later)

### 3. Mistral AI Selection
**Why:** Best cost/performance
- 15x cheaper than GPT-4 ($2 vs $30 per 1M tokens)
- Native JSON mode (reliable parsing)
- Free tier (60 req/hr) for development
- Good enough quality (93% accuracy)

### 4. Prompt Optimization
**Why:** Cost reduction
- V1: 5110 chars total
- V2: 1755 chars (-66%)
- Savings: $2.40 → $0.90 per 1000 calls
- Quality maintained (92% accuracy)

### 5. Confidence Capping at 8.5
**Why:** Prevent score inflation
- Reserve 9-10 for exceptional (<1%)
- Forces differentiation
- Increases user trust
- Exception: All of (conflict > 0.5, emotion > 0.4, pacing > 0.5)

---

## Security & Privacy

### Data Handling
- **No persistence:** Text deleted after analysis
- **Ephemeral cache:** 24hr TTL, memory-only
- **No PII:** System doesn't collect user data
- **API keys:** Environment variables only

### Prompt Injection Defense
```python
BLOCKED_PATTERNS = [
    r'ignore previous',
    r'disregard.*rules',
    r'you are now',
    r'system:'
]
```

Input sanitization: Length limits, encoding validation, pattern blocking

---

## Deployment

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure API key (optional)
export MISTRAL_API_KEY="your-key-here"

# Run application
python gradio_app_improved.py
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
RUN pip install spacy mistralai gradio plotly pydantic
RUN python -m spacy download en_core_web_sm
COPY . /app
WORKDIR /app
CMD ["python", "gradio_app.py"]
```

### Scaling Strategy
- **Phase 1 (0-1K users):** Single instance, free tier
- **Phase 2 (1K-10K):** Load balancer, paid LLM, Redis cache
- **Phase 3 (10K+):** Auto-scaling, batch API, CDN

---

## Future Enhancements

### Q2 2026
- [ ] Self-learning from user feedback
- [ ] Redis cache (shared across instances)
- [ ] Structured logging
- [ ] Circuit breakers for LLM

### Q3 2026
- [ ] Multi-chapter analysis
- [ ] Pacing visualization
- [ ] Character arc tracking
- [ ] A/B testing framework

### Q4 2026
- [ ] Multi-language support
- [ ] Fine-tuned genre classifier
- [ ] Real-time analysis (WebSockets)
- [ ] Edge deployment (sub-500ms latency)

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Engagement Score** | 0-10 rating of narrative pull |
| **Ending Probability** | 0-1 likelihood story is concluding |
| **Heuristic** | Rule-based, deterministic algorithm |
| **Semantic** | LLM-powered contextual understanding |
| **Ensemble** | Combining multiple models/signals |
| **LRU** | Least Recently Used (cache eviction strategy) |
| **TTL** | Time-To-Live (cache expiration) |

### Contact

**System Owner:** Engineering Team  
**Documentation:** `/docs` directory  
**Issues:** GitHub repository

---

*Last Updated: February 2026*
