# Contributing to Tree of Thought Benchmarking

Thank you for your interest in contributing! This project aims to explore and benchmark the effectiveness of Tree of Thought prompting compared to standard approaches.

## Ways to Contribute

### 🧠 Add New Test Cases
- Add problems to `data/test_datasets/`
- Include math, logic, or reasoning challenges
- Provide expected answers for evaluation

### 🔧 Improve Evaluation Metrics
- Enhance scoring algorithms in `src/evaluator.py`
- Add new evaluation criteria
- Improve accuracy detection

### 🎯 Experiment with Prompting
- Create new prompt templates in `src/prompts/`
- Modify Tree of Thought methodology
- Test different reasoning structures

### 📊 Add Visualizations
- Enhance plots in `src/benchmark.py`
- Create new analysis charts
- Improve result presentation

### 🚀 Support New Models
- Add support for other LLMs
- Create new modelfiles
- Improve model management

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes (`python scripts/test_setup.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Style

- Follow PEP 8 for Python code
- Add docstrings to new functions
- Include type hints where appropriate
- Write clear commit messages

## Testing

Before submitting:
- Run `python scripts/test_setup.py` to verify setup
- Test with a small benchmark run
- Ensure all new test cases have expected answers

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the methodology
- Suggestions for improvements

We welcome all skill levels and perspectives! 🎉
