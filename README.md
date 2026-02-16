# Story Analyzer - Narrative Intelligence System (NIS)

A comprehensive story analysis toolkit that combines deterministic heuristics with optional LLM semantic understanding for reliable, cost-efficient content analysis.

## Features

The Narrative Intelligence System (NIS) provides four core analysis capabilities:

### 1. **Engagement Analysis**
Analyzes story engagement using deterministic heuristics:
- Pacing analysis (sentence length variation)
- Dialogue density measurement
- Action word frequency detection
- Emotional intensity scoring
- Opening hook quality assessment
- Readability metrics

### 2. **Ending Prediction**
Predicts story endings using pattern recognition:
- Story arc identification (rising action, climax, resolution)
- Conflict resolution pattern detection
- Character trajectory analysis
- Tension curve mapping
- Ending type classification (happy, tragic, bittersweet, open, twist, ambiguous)

### 3. **Genre Classification**
Classifies stories into genres using keyword analysis:
- Multi-genre detection and scoring
- Subgenre identification
- Tone analysis (dark, light, serious, humorous, neutral)
- Mixed-genre detection
- Supports: Fantasy, Science Fiction, Mystery, Romance, Horror, Thriller, Adventure, Historical, Comedy, Drama

### 4. **Character Relationship Mapping**
Maps character relationships and social networks:
- Character extraction and mention tracking
- Relationship inference through co-occurrence
- Main character identification
- Character importance ranking
- Network density analysis
- Character cluster detection
- Visualization-ready network data

## Installation

```bash
# Clone the repository
git clone https://github.com/AnthonyRajuKondaveeti/story-analyzer.git
cd story-analyzer

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from story_analyzer import NarrativeIntelligence

# Initialize the system
nis = NarrativeIntelligence()

# Analyze a story
story = """
Your story text here...
"""

# Full analysis
results = nis.analyze(story)

# Access individual components
engagement = results['engagement']
genre = results['genre']
ending = results['ending_prediction']
characters = results['characters']

print(f"Engagement score: {engagement['engagement_score']}")
print(f"Primary genre: {genre['primary_genre']}")
print(f"Predicted ending: {ending['predicted_ending']}")
print(f"Main characters: {characters['main_characters']}")
```

## Usage Examples

### Analyze Specific Components

```python
# Analyze only engagement
engagement = nis.analyze_engagement(story)

# Analyze only genre
genre = nis.classify_genre(story)

# Predict only ending
ending = nis.predict_ending(story)

# Map only characters
characters = nis.map_characters(story)

# Get character network for visualization
network = nis.get_character_network(story)
```

### Selective Analysis

```python
# Run only specific analyses
results = nis.analyze(story, components=['engagement', 'genre'])
```

## Architecture

The system uses a modular architecture with four independent analyzers:

- **EngagementAnalyzer**: Deterministic heuristics for engagement metrics
- **EndingPredictor**: Pattern recognition for ending prediction
- **GenreClassifier**: Keyword-based genre classification
- **CharacterMapper**: Network analysis for character relationships
- **NarrativeIntelligence**: Main orchestrator combining all components

### Deterministic-First Approach

The system prioritizes deterministic heuristics for:
- **Cost efficiency**: No API costs for basic analysis
- **Reliability**: Consistent, reproducible results
- **Speed**: Fast analysis without network calls
- **Privacy**: No data sent to external services

### Optional LLM Enhancement

LLM support can be enabled for enhanced semantic understanding:

```python
# Initialize with LLM support (requires OpenAI API key)
nis = NarrativeIntelligence(
    use_llm=True,
    llm_config={'api_key': 'your-key-here'}
)
```

## Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=story_analyzer tests/
```

## Example Output

```
=== Narrative Intelligence System Demo ===

--- Basic Info ---
Text length: 892 characters
Word count: 156 words
Analysis mode: deterministic

--- Engagement Analysis ---
Overall engagement score: 0.687
Pacing variance: 0.542
Dialogue density: 0.234
Action density: 0.156
Emotional intensity: 0.128
Hook quality: 0.850
Readability: 0.723

--- Genre Classification ---
Primary genre: mystery
Is mixed genre: True
Tone: serious
Genre scores:
  mystery: 0.456
  thriller: 0.312
  drama: 0.232

--- Ending Prediction ---
Predicted ending type: happy
Confidence: 0.783
Arc completion: 0.850
Resolution likelihood: 0.920
Emotional trajectory: ascending

--- Character Analysis ---
Total characters found: 3
Characters: Sarah, James, Detective
Main characters: Sarah, James
Network density: 0.667
```

## Project Structure

```
story-analyzer/
├── src/
│   └── story_analyzer/
│       ├── __init__.py
│       ├── engagement_analyzer.py
│       ├── ending_predictor.py
│       ├── genre_classifier.py
│       ├── character_mapper.py
│       └── narrative_intelligence.py
├── tests/
│   ├── test_engagement_analyzer.py
│   ├── test_ending_predictor.py
│   ├── test_genre_classifier.py
│   ├── test_character_mapper.py
│   └── test_narrative_intelligence.py
├── examples/
│   └── basic_usage.py
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

## Dependencies

Core dependencies:
- `numpy>=1.20.0` - Numerical computations
- `networkx>=2.5` - Graph analysis for character relationships

Development dependencies:
- `pytest>=7.0` - Testing framework
- `pytest-cov>=3.0` - Coverage reporting

Optional dependencies:
- `openai>=1.0.0` - LLM integration (install with `pip install story-analyzer[llm]`)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The Narrative Intelligence System combines insights from:
- Narrative theory and story structure analysis
- Natural language processing techniques
- Social network analysis
- Engagement metrics research