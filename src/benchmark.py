"""
Main benchmarking script for comparing ToT vs normal prompting
"""
import json
import time
import yaml
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from model_manager import ModelManager
from evaluator import ResponseEvaluator
from prompts.tot_prompts import TOT_MATH_TEMPLATE, TOT_LOGIC_TEMPLATE, TOT_GENERAL_TEMPLATE
from prompts.normal_prompts import NORMAL_MATH_TEMPLATE, NORMAL_LOGIC_TEMPLATE, NORMAL_GENERAL_TEMPLATE

class BenchmarkRunner:
    def __init__(self, config_path: str = "config/benchmark_config.yaml"):
        """Initialize the benchmark runner"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.model_manager = ModelManager()
        self.evaluator = ResponseEvaluator()
        
        # Setup result directory
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Template mappings
        self.tot_templates = {
            "math": TOT_MATH_TEMPLATE,
            "logic": TOT_LOGIC_TEMPLATE,
            "reasoning": TOT_GENERAL_TEMPLATE
        }
        
        self.normal_templates = {
            "math": NORMAL_MATH_TEMPLATE,
            "logic": NORMAL_LOGIC_TEMPLATE,
            "reasoning": NORMAL_GENERAL_TEMPLATE
        }
    
    def load_test_datasets(self) -> Dict[str, List[Dict]]:
        """Load all test datasets"""
        datasets = {}
        
        dataset_files = {
            "math": "data/test_datasets/math_problems.json",
            "logic": "data/test_datasets/logic_puzzles.json", 
            "reasoning": "data/test_datasets/reasoning_tasks.json"
        }
        
        for dataset_name, file_path in dataset_files.items():
            dataset_key = f'{dataset_name}_problems' if dataset_name == 'math' else f'{dataset_name}_puzzles' if dataset_name == 'logic' else f'{dataset_name}_tasks'
            if self.config['datasets'][dataset_key]['enabled']:
                with open(file_path, 'r') as f:
                    datasets[dataset_name] = json.load(f)
                    
                # Limit dataset size if specified
                count = self.config['datasets'][dataset_key]['count']
                if count and len(datasets[dataset_name]) > count:
                    datasets[dataset_name] = datasets[dataset_name][:count]
        
        return datasets
    
    def run_single_test(self, problem: Dict, model_name: str, template: str) -> Optional[Dict]:
        """Run a single test case"""
        
        # Format the prompt
        prompt = template.format(problem=problem['problem'])
        
        # Query the model
        start_time = time.time()
        result = self.model_manager.query_model(model_name, prompt)
        end_time = time.time()
        
        if "error" in result:
            self.logger.error(f"Error querying {model_name}: {result['error']}")
            return None
        
        # Evaluate the response
        scores = self.evaluator.evaluate_response(
            result['response'], 
            problem['expected_answer'], 
            problem['type']
        )
        
        # Calculate overall score
        overall_score = self.evaluator.calculate_overall_score(scores)
        
        return {
            "problem_id": problem['id'],
            "problem_type": problem['type'],
            "difficulty": problem['difficulty'],
            "model": model_name,
            "prompt_type": "tot" if "tot" in model_name else "normal",
            "response": result['response'],
            "response_time": end_time - start_time,
            "scores": scores,
            "overall_score": overall_score,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_benchmark(self) -> List[Dict]:
        """Run the complete benchmark"""
        
        self.logger.info("Starting benchmark run...")
        
        # Load datasets
        datasets = self.load_test_datasets()
        
        # Model names from config
        tot_model = self.model_manager.config['tot_model']['name']
        normal_model = self.model_manager.config['normal_model']['name']
        
        results = []
        
        for dataset_name, problems in datasets.items():
            self.logger.info(f"Testing {dataset_name} problems...")
            
            for problem in problems:
                self.logger.info(f"Testing problem {problem['id']}: {problem['problem'][:50]}...")
                
                # Test with ToT model
                tot_result = self.run_single_test(
                    problem, 
                    tot_model,
                    self.tot_templates[dataset_name]
                )
                if tot_result:
                    results.append(tot_result)
                
                # Test with normal model
                normal_result = self.run_single_test(
                    problem,
                    normal_model, 
                    self.normal_templates[dataset_name]
                )
                if normal_result:
                    results.append(normal_result)
                
                # Small delay to avoid overwhelming the model
                time.sleep(1)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save benchmark results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        if "json" in self.config['output']['save_format']:
            json_path = self.results_dir / f"benchmark_results_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to {json_path}")
        
        # Save as CSV
        if "csv" in self.config['output']['save_format']:
            # Flatten the results for CSV
            flattened_results = []
            for result in results:
                flat_result = {
                    "problem_id": result["problem_id"],
                    "problem_type": result["problem_type"],
                    "difficulty": result["difficulty"],
                    "model": result["model"],
                    "prompt_type": result["prompt_type"],
                    "response_time": result["response_time"],
                    "overall_score": result["overall_score"],
                    "timestamp": result["timestamp"]
                }
                # Add individual scores
                for score_name, score_value in result["scores"].items():
                    flat_result[f"score_{score_name}"] = score_value
                
                flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            csv_path = self.results_dir / f"benchmark_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Results saved to {csv_path}")
    
    def generate_analysis_plots(self, results: List[Dict]):
        """Generate analysis plots from results"""
        
        if not self.config['output']['generate_plots']:
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Overall Score Comparison
        sns.boxplot(data=df, x='prompt_type', y='overall_score', ax=axes[0,0])
        axes[0,0].set_title('Overall Score: ToT vs Normal')
        axes[0,0].set_ylabel('Overall Score')
        
        # Plot 2: Response Time Comparison
        sns.boxplot(data=df, x='prompt_type', y='response_time', ax=axes[0,1])
        axes[0,1].set_title('Response Time: ToT vs Normal')
        axes[0,1].set_ylabel('Response Time (seconds)')
        
        # Plot 3: Score by Problem Type
        score_cols = [col for col in df.columns if col.startswith('score_')]
        if score_cols:
            score_data = df.melt(
                id_vars=['prompt_type', 'problem_type'], 
                value_vars=score_cols,
                var_name='metric', 
                value_name='score'
            )
            sns.barplot(data=score_data, x='metric', y='score', hue='prompt_type', ax=axes[1,0])
            axes[1,0].set_title('Detailed Score Comparison')
            axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # Plot 4: Performance by Difficulty
        sns.boxplot(data=df, x='difficulty', y='overall_score', hue='prompt_type', ax=axes[1,1])
        axes[1,1].set_title('Performance by Difficulty Level')
        
        plt.tight_layout()
        
        # Save the plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"analysis_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Analysis plots saved to {plot_path}")
    
    def print_summary(self, results: List[Dict]):
        """Print benchmark summary"""
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Overall statistics
        tot_results = df[df['prompt_type'] == 'tot']
        normal_results = df[df['prompt_type'] == 'normal']
        
        print(f"\nTotal problems tested: {len(df) // 2}")
        print(f"Problems per model: {len(tot_results)}")
        
        print(f"\nToT Model Average Score: {tot_results['overall_score'].mean():.3f}")
        print(f"Normal Model Average Score: {normal_results['overall_score'].mean():.3f}")
        
        print(f"\nToT Model Average Response Time: {tot_results['response_time'].mean():.2f}s")
        print(f"Normal Model Average Response Time: {normal_results['response_time'].mean():.2f}s")
        
        # Performance by problem type
        print(f"\nPerformance by Problem Type:")
        for problem_type in df['problem_type'].unique():
            type_tot = tot_results[tot_results['problem_type'] == problem_type]['overall_score'].mean()
            type_normal = normal_results[normal_results['problem_type'] == problem_type]['overall_score'].mean()
            
            print(f"  {problem_type.capitalize()}:")
            print(f"    ToT: {type_tot:.3f}")
            print(f"    Normal: {type_normal:.3f}")
            print(f"    Improvement: {((type_tot - type_normal) / type_normal * 100):+.1f}%")
        
        print("="*60)

def main():
    """Main execution function"""
    
    benchmark = BenchmarkRunner()
    
    # Setup models first
    print("Setting up models...")
    if not benchmark.model_manager.setup_models():
        print("Failed to setup models. Please check your Ollama installation.")
        return
    
    # Run benchmark
    print("Running benchmark...")
    results = benchmark.run_benchmark()
    
    if not results:
        print("No results generated. Check logs for errors.")
        return
    
    # Save results
    benchmark.save_results(results)
    
    # Generate analysis
    benchmark.generate_analysis_plots(results)
    
    # Print summary
    benchmark.print_summary(results)
    
    print(f"\nBenchmark complete! Results saved to {benchmark.results_dir}")

if __name__ == "__main__":
    main()
