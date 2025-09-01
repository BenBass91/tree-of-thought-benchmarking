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
from scipy import stats
import numpy as np

from model_manager import ModelManager
from evaluator import ResponseEvaluator
from prompts.tot_prompts import TOT_MATH_TEMPLATE, TOT_LOGIC_TEMPLATE, TOT_GENERAL_TEMPLATE, TOT_CODING_TEMPLATE
from prompts.normal_prompts import NORMAL_MATH_TEMPLATE, NORMAL_LOGIC_TEMPLATE, NORMAL_GENERAL_TEMPLATE, NORMAL_CODING_TEMPLATE

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
            "reasoning": TOT_GENERAL_TEMPLATE,
            "coding": TOT_CODING_TEMPLATE
        }
        
        self.normal_templates = {
            "math": NORMAL_MATH_TEMPLATE,
            "logic": NORMAL_LOGIC_TEMPLATE,
            "reasoning": NORMAL_GENERAL_TEMPLATE,
            "coding": NORMAL_CODING_TEMPLATE
        }
    
    def load_test_datasets(self) -> Dict[str, List[Dict]]:
        """Load all test datasets"""
        datasets = {}
        
        dataset_files = {
            "math": "data/test_datasets/math_problems.json",
            "logic": "data/test_datasets/logic_puzzles.json", 
            "reasoning": "data/test_datasets/reasoning_tasks.json",
            "coding": "data/test_datasets/coding_problems.json"
        }
        
        for dataset_name, file_path in dataset_files.items():
            dataset_key = f'{dataset_name}_problems' if dataset_name in ['math', 'coding'] else f'{dataset_name}_puzzles' if dataset_name == 'logic' else f'{dataset_name}_tasks'
            if self.config['datasets'][dataset_key]['enabled']:
                with open(file_path, 'r') as f:
                    datasets[dataset_name] = json.load(f)
                    
                # Limit dataset size if specified
                count = self.config['datasets'][dataset_key]['count']
                if count and len(datasets[dataset_name]) > count:
                    datasets[dataset_name] = datasets[dataset_name][:count]
        
        return datasets
    
    def run_single_test(self, problem: Dict, model_name: str, template: str, max_retries: int = 2) -> Optional[Dict]:
        """Run a single test case with retry logic"""
        
        # Format the prompt
        prompt = template.format(problem=problem['problem'])
        
        for attempt in range(max_retries + 1):
            try:
                # Query the model
                start_time = time.time()
                result = self.model_manager.query_model(model_name, prompt)
                end_time = time.time()
                
                if "error" in result:
                    if attempt < max_retries:
                        self.logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {result['error']}. Retrying...")
                        time.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        self.logger.error(f"All attempts failed for {model_name}: {result['error']}")
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
                    "timestamp": datetime.now().isoformat(),
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed with exception: {e}. Retrying...")
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(f"All attempts failed with exception: {e}")
                    return None
        
        return None
    
    def run_benchmark(self) -> List[Dict]:
        """Run the complete benchmark"""
        
        self.logger.info("Starting benchmark run...")
        
        # Load datasets
        datasets = self.load_test_datasets()
        
        # Calculate total problems for progress tracking
        total_problems = sum(len(problems) for problems in datasets.values())
        completed_problems = 0
        
        # Model names from config
        tot_model = self.model_manager.config['tot_model']['name']
        normal_model = self.model_manager.config['normal_model']['name']
        
        results = []
        
        for dataset_name, problems in datasets.items():
            self.logger.info(f"Testing {dataset_name} problems ({len(problems)} problems)...")
            
            for i, problem in enumerate(problems, 1):
                completed_problems += 1
                progress = (completed_problems / total_problems) * 100
                
                self.logger.info(f"Progress: {progress:.1f}% - Testing problem {problem['id']}: {problem['problem'][:50]}...")
                print(f"[{completed_problems}/{total_problems}] ({progress:.1f}%) - {dataset_name} problem {i}/{len(problems)}")
                
                # Test with ToT model
                tot_result = self.run_single_test(
                    problem, 
                    tot_model,
                    self.tot_templates[dataset_name]
                )
                if tot_result:
                    results.append(tot_result)
                    print(f"  ✓ ToT completed in {tot_result['response_time']:.1f}s (attempts: {tot_result.get('attempts', 1)})")
                else:
                    print(f"  ✗ ToT failed")
                
                # Test with normal model
                normal_result = self.run_single_test(
                    problem,
                    normal_model, 
                    self.normal_templates[dataset_name]
                )
                if normal_result:
                    results.append(normal_result)
                    print(f"  ✓ Normal completed in {normal_result['response_time']:.1f}s (attempts: {normal_result.get('attempts', 1)})")
                else:
                    print(f"  ✗ Normal failed")
                
                # Small delay to avoid overwhelming the model
                time.sleep(1)
        
        print(f"\nBenchmark completed! Successfully processed {len(results)}/{total_problems * 2} test cases.")
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
    
    def perform_statistical_analysis(self, results: List[Dict]) -> Dict:
        """Perform statistical significance testing"""
        
        df = pd.DataFrame(results)
        
        # Group by problem and compare ToT vs Normal
        analysis = {}
        
        tot_scores = np.array(df[df['prompt_type'] == 'tot']['overall_score'].astype(float))
        normal_scores = np.array(df[df['prompt_type'] == 'normal']['overall_score'].astype(float))
        
        # Paired t-test (since we test the same problems with both methods)
        if len(tot_scores) == len(normal_scores) and len(tot_scores) > 1:
            statistic, p_value = stats.ttest_rel(tot_scores, normal_scores)
            
            analysis['overall'] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size': float(np.mean(tot_scores) - np.mean(normal_scores)),
                'tot_mean': float(np.mean(tot_scores)),
                'normal_mean': float(np.mean(normal_scores)),
                'sample_size': len(tot_scores)
            }
        
        # Analysis by problem type
        for problem_type in df['problem_type'].unique():
            type_df = df[df['problem_type'] == problem_type]
            tot_type = np.array(type_df[type_df['prompt_type'] == 'tot']['overall_score'].astype(float))
            normal_type = np.array(type_df[type_df['prompt_type'] == 'normal']['overall_score'].astype(float))
            
            if len(tot_type) == len(normal_type) and len(tot_type) > 1:
                statistic, p_value = stats.ttest_rel(tot_type, normal_type)
                
                analysis[problem_type] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size': float(np.mean(tot_type) - np.mean(normal_type)),
                    'tot_mean': float(np.mean(tot_type)),
                    'normal_mean': float(np.mean(normal_type)),
                    'sample_size': len(tot_type)
                }
        
        return analysis
    
    def print_summary(self, results: List[Dict], statistical_analysis: Optional[Dict] = None):
        """Print benchmark summary with statistical analysis"""
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("TREE OF THOUGHT BENCHMARK RESULTS")
        print("="*60)
        
        # Overall comparison
        tot_results = df[df['prompt_type'] == 'tot']
        normal_results = df[df['prompt_type'] == 'normal']
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Tree of Thought Average: {tot_results['overall_score'].mean():.3f}")
        print(f"Normal Prompting Average: {normal_results['overall_score'].mean():.3f}")
        
        improvement = ((tot_results['overall_score'].mean() - normal_results['overall_score'].mean()) / 
                      normal_results['overall_score'].mean()) * 100
        print(f"Improvement: {improvement:+.1f}%")
        
        # Statistical significance
        if statistical_analysis and 'overall' in statistical_analysis:
            overall_stats = statistical_analysis['overall']
            print(f"\nSTATISTICAL ANALYSIS:")
            print(f"Sample size: {overall_stats['sample_size']} paired comparisons")
            print(f"Effect size: {overall_stats['effect_size']:+.3f}")
            print(f"P-value: {overall_stats['p_value']:.4f}")
            if overall_stats['significant']:
                print("✓ Statistically significant (p < 0.05)")
            else:
                print("✗ Not statistically significant (p ≥ 0.05)")
        
        # Performance by problem type
        print(f"\nPERFORMANCE BY PROBLEM TYPE:")
        for problem_type in df['problem_type'].unique():
            tot_avg = tot_results[tot_results['problem_type'] == problem_type]['overall_score'].mean()
            normal_avg = normal_results[normal_results['problem_type'] == problem_type]['overall_score'].mean()
            type_improvement = ((tot_avg - normal_avg) / normal_avg) * 100
            
            significance_indicator = ""
            if statistical_analysis and problem_type in statistical_analysis:
                if statistical_analysis[problem_type]['significant']:
                    significance_indicator = " ✓"
                else:
                    significance_indicator = " ○"
            
            print(f"{problem_type:15} | ToT: {tot_avg:.3f} | Normal: {normal_avg:.3f} | "
                  f"Improvement: {type_improvement:+.1f}%{significance_indicator}")
        
        # Timing analysis
        print(f"\nTIMING ANALYSIS:")
        tot_time = tot_results['response_time'].mean()
        normal_time = normal_results['response_time'].mean()
        time_overhead = ((tot_time - normal_time) / normal_time) * 100
        
        print(f"Tree of Thought Avg Time: {tot_time:.2f}s")
        print(f"Normal Prompting Avg Time: {normal_time:.2f}s")
        print(f"Time Overhead: {time_overhead:+.1f}%")
        
        # Difficulty analysis
        print(f"\nPERFORMANCE BY DIFFICULTY:")
        for difficulty in sorted(df['difficulty'].unique()):
            diff_tot = tot_results[tot_results['difficulty'] == difficulty]['overall_score'].mean()
            diff_normal = normal_results[normal_results['difficulty'] == difficulty]['overall_score'].mean()
            diff_improvement = ((diff_tot - diff_normal) / diff_normal) * 100
            
            print(f"{difficulty:10} | ToT: {diff_tot:.3f} | Normal: {diff_normal:.3f} | "
                  f"Improvement: {diff_improvement:+.1f}%")
        
        print("\n" + "="*60)
        
        if statistical_analysis:
            print("\nLegend: ✓ = Statistically significant, ○ = Not significant")
        
        print(f"\nTotal problems tested: {len(df) // 2}")
        print(f"Results saved to: {self.results_dir}")
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
    
    # Perform statistical analysis
    statistical_analysis = benchmark.perform_statistical_analysis(results)
    
    # Save statistical analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = benchmark.results_dir / f"statistical_analysis_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(statistical_analysis, f, indent=2)
    print(f"Statistical analysis saved to {stats_path}")
    
    # Generate analysis
    benchmark.generate_analysis_plots(results)
    
    # Perform statistical analysis
    statistical_analysis = benchmark.perform_statistical_analysis(results)
    
    # Print summary with statistics
    benchmark.print_summary(results, statistical_analysis)
    
    print(f"\nBenchmark complete! Results saved to {benchmark.results_dir}")

if __name__ == "__main__":
    main()
