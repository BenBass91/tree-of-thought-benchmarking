# 🌳 Tree of Thought Benchmarking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Latest-green.svg)](https://ollama.ai/)

A comprehensive benchmarking suite comparing Microsoft's Phi4-mini-reasoning model with Tree of Thought (ToT) prompting versus standard prompting approaches.

## 📊 Key Results

Initial benchmarking shows **measurable improvements** with Tree of Thought methodology:

- **Overall Performance**: +4.5% improvement
- **Reasoning Tasks**: +17.2% improvement 🚀
- **Math Problems**: +4.3% improvement
- **Logic Puzzles**: +0.5% improvement

> *Tree of Thought shows the biggest gains in complex reasoning tasks, validating the structured thinking approach.*

## 📁 Project Structure

```text
ToT_Tuning/
├── README.md                 # Project documentation
├── QUICKSTART.md            # Quick setup guide  
├── requirements.txt         # Python dependencies
├── config/                  # Configuration files
│   ├── model_config.yaml    # Model settings
│   └── benchmark_config.yaml # Benchmark parameters
├── modelfiles/              # Ollama model definitions
│   ├── phi4_tot.modelfile   # ToT-enhanced model
│   └── phi4_normal.modelfile # Standard model
├── src/                     # Source code
│   ├── benchmark.py         # Main benchmarking engine
│   ├── model_manager.py     # Ollama model management
│   ├── evaluator.py         # Response evaluation metrics
│   └── prompts/             # Prompt templates
├── data/                    # Test data and results
│   ├── test_datasets/       # Curated test problems
│   └── results/             # Benchmark outputs
└── scripts/                 # Utility scripts
    ├── setup_models.py      # Model setup automation
    ├── run_benchmark.py     # Main execution
    └── test_setup.py        # Project verification
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and setup Ollama:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull phi4-mini-reasoning
   ```

3. **Setup benchmark models:**
   ```bash
   python scripts/setup_models.py
   ```

4. **Run benchmark:**
   ```bash
   python scripts/run_benchmark.py
   ```

## 🎯 What This Tests

### Problem Categories
- **Math Problems**: Arithmetic, algebra, geometry
- **Logic Puzzles**: Deductive reasoning, constraint satisfaction
- **General Reasoning**: Analysis, comparison, argumentation

### Evaluation Metrics
- **Accuracy** (40%): Correctness of final answers
- **Reasoning Quality** (25%): Logical coherence and depth
- **Completeness** (20%): Thoroughness of explanation
- **Clarity** (15%): Organization and readability

### Models Compared
- **phi4-tot**: Enhanced with Tree of Thought methodology
- **phi4-normal**: Standard prompting baseline

## 🌳 Tree of Thought Approach

The ToT model uses a structured 6-step reasoning process:

1. **Problem Analysis**: Break down components
2. **Branch Generation**: Explore multiple solution paths
3. **Path Evaluation**: Assess approach viability
4. **Path Selection**: Choose optimal strategy
5. **Step-by-Step Execution**: Implement solution
6. **Verification**: Validate final answer

## 📊 Expected Outcomes

This benchmark will help determine:
- Whether ToT prompting improves reasoning accuracy
- If structured prompting affects response times
- Which problem types benefit most from ToT
- Trade-offs between quality and efficiency

## 🔧 Customization

- **Add problems**: Edit JSON files in `data/test_datasets/`
- **Modify prompts**: Update templates in `src/prompts/`
- **Adjust evaluation**: Configure weights in `config/benchmark_config.yaml`
- **Change models**: Update model definitions in `modelfiles/`

---

Ready to explore the power of structured reasoning! 🧠✨

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for ways to help:

- 🧠 Add new test cases and datasets
- 🔧 Improve evaluation metrics
- 🎯 Experiment with prompt engineering
- 📊 Enhance visualizations
- 🚀 Add support for new models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft for the Phi4-mini-reasoning model
- The Ollama team for the excellent local LLM framework
- The Tree of Thought methodology researchers
- Open source community for inspiration and tools

## 📚 Citation

If you use this benchmarking suite in your research, please cite:

```
Tree of Thought Benchmarking Suite
https://github.com/bsoldate/tree-of-thought-benchmarking
```
