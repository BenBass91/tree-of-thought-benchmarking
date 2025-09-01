#!/bin/bash
# Setup script for Tree of Thought Benchmarking

echo "🌳 Setting up Tree of Thought Benchmarking Project"
echo "================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama found"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Pull base model
echo "🤖 Pulling base model..."
ollama pull phi4-mini-reasoning

# Setup custom models
echo "⚙️  Setting up benchmark models..."
python scripts/setup_models.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run a test: python scripts/test_setup.py"
echo "2. Run benchmark: python scripts/run_benchmark.py"
echo "3. Check results in: data/results/"
echo ""
echo "For more information, see README.md"
