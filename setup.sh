#!/bin/bash

# Story Engagement Analyzer - Quick Start Script

echo "🚀 Story Engagement Analyzer - Setup"
echo "===================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

echo "✓ Virtual environment created"
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Download spaCy model
echo "🔤 Downloading spaCy model..."
python3 -m spacy download en_core_web_sm

echo "✓ spaCy model downloaded"
echo ""

# Check for API key
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "⚠️  MISTRAL_API_KEY not set"
    echo ""
    echo "To use LLM features:"
    echo "1. Get API key: https://console.mistral.ai/"
    echo "2. Set it: export MISTRAL_API_KEY='your-key-here'"
    echo ""
    echo "Or create .env file:"
    echo "echo 'MISTRAL_API_KEY=your-key-here' > .env"
    echo ""
else
    echo "✓ MISTRAL_API_KEY is set"
    echo ""
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
from src.text_analyzer import TextAnalyzer
analyzer = TextAnalyzer()
print('✓ Text analyzer works')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo ""
echo "===================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set MISTRAL_API_KEY (if not done)"
echo "2. Run: python3 api.py"
echo "3. Visit: http://localhost:8000"
echo ""
echo "Or deploy:"
echo "1. Push to GitHub"
echo "2. Deploy to Render.com (see README.md)"
echo ""
echo "Documentation:"
echo "- README.md - Deployment guide"
echo "- DESIGN_JUSTIFICATIONS.md - Design decisions"
echo "- API docs: http://localhost:8000/docs (after starting)"
echo "===================================="
