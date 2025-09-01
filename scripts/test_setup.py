#!/usr/bin/env python3
"""
Simple test to verify the project setup
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import yaml

def test_config_files():
    """Test that configuration files are valid"""
    try:
        # Test model config
        with open('config/model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        print("✅ Model config loaded successfully")
        
        # Test benchmark config
        with open('config/benchmark_config.yaml', 'r') as f:
            benchmark_config = yaml.safe_load(f)
        print("✅ Benchmark config loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_datasets():
    """Test that dataset files are valid"""
    try:
        datasets = [
            'data/test_datasets/math_problems.json',
            'data/test_datasets/logic_puzzles.json',
            'data/test_datasets/reasoning_tasks.json'
        ]
        
        for dataset_path in datasets:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            print(f"✅ {dataset_path} loaded successfully ({len(data)} problems)")
        
        return True
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False

def test_imports():
    """Test that main modules can be imported"""
    try:
        from model_manager import ModelManager
        print("✅ ModelManager imported successfully")
        
        from evaluator import ResponseEvaluator
        print("✅ ResponseEvaluator imported successfully")
        
        from prompts.tot_prompts import TOT_MATH_TEMPLATE
        from prompts.normal_prompts import NORMAL_MATH_TEMPLATE
        print("✅ Prompt templates imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Tree of Thought project setup...")
    print("=" * 45)
    
    tests = [
        ("Configuration files", test_config_files),
        ("Dataset files", test_datasets),
        ("Module imports", test_imports)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 45)
    if all_passed:
        print("🎉 All tests passed! Project setup is correct.")
        print("\nNext steps:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Setup models: python scripts/setup_models.py")
        print("4. Run benchmark: python scripts/run_benchmark.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
