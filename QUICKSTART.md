# Quick Start Guide

## Prerequisites

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Pull the base model**:
   ```bash
   ollama pull phi4-mini-reasoning
   ```

2. **Setup benchmark models**:
   ```bash
   python scripts/setup_models.py
   ```

## Run Benchmark

```bash
python scripts/run_benchmark.py
```

## Results

Results will be saved in `data/results/` with:
- JSON files with detailed responses
- CSV files for analysis
- PNG plots showing comparisons

## Project Overview

This benchmark compares:
- **phi4-tot**: Phi4-mini-reasoning with Tree of Thought prompting
- **phi4-normal**: Phi4-mini-reasoning with standard prompting

Test categories:
- Math problems (arithmetic, algebra, geometry)
- Logic puzzles (deduction, reasoning validation)
- General reasoning tasks (analysis, comparison)

Evaluation metrics:
- Accuracy (40% weight)
- Reasoning quality (25% weight)  
- Completeness (20% weight)
- Clarity (15% weight)
