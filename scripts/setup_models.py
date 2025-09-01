#!/usr/bin/env python3
"""
Setup script for Ollama models
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
from model_manager import ModelManager

def setup_models():
    """Setup all required models for the benchmark"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up models for ToT benchmarking...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        
        # Setup models
        success = model_manager.setup_models()
        
        if success:
            logger.info("✅ All models setup successfully!")
            logger.info("Models created:")
            logger.info("  - phi4-tot (Tree of Thought)")
            logger.info("  - phi4-normal (Standard)")
            return True
        else:
            logger.error("❌ Failed to setup models")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during model setup: {e}")
        return False

def verify_ollama():
    """Verify Ollama is installed and running"""
    
    import subprocess
    
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama command not found")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not installed or not in PATH")
        print("Please install Ollama from: https://ollama.ai/")
        return False

def main():
    """Main setup function"""
    
    print("Setting up Tree of Thought Benchmarking Environment")
    print("=" * 50)
    
    # Verify Ollama installation
    if not verify_ollama():
        sys.exit(1)
    
    # Setup models
    if not setup_models():
        sys.exit(1)
    
    print("\n🎉 Setup complete! Ready to run benchmarks.")
    print("\nNext steps:")
    print("1. Run: python scripts/run_benchmark.py")
    print("2. Check results in: data/results/")

if __name__ == "__main__":
    main()
