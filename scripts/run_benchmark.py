#!/usr/bin/env python3
"""
Main script to run the Tree of Thought benchmark
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmark import main as run_benchmark

if __name__ == "__main__":
    print("Starting Tree of Thought Benchmark")
    print("=" * 40)
    
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        sys.exit(1)
