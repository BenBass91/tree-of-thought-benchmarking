"""
Evaluation module for scoring model responses
"""
import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

class ResponseEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def evaluate_response(self, response: str, expected_answer: Any, problem_type: str) -> Dict[str, float]:
        """Evaluate a model response across multiple metrics"""
        
        scores = {
            "accuracy": self._score_accuracy(response, expected_answer, problem_type),
            "reasoning_quality": self._score_reasoning_quality(response),
            "completeness": self._score_completeness(response, problem_type),
            "clarity": self._score_clarity(response),
            "step_count": self._count_reasoning_steps(response)
        }
        
        return scores
    
    def _score_accuracy(self, response: str, expected_answer: Any, problem_type: str) -> float:
        """Score accuracy based on problem type"""
        
        if problem_type == "math":
            return self._score_math_accuracy(response, expected_answer)
        elif problem_type == "logic":
            return self._score_logic_accuracy(response, expected_answer)
        else:
            return self._score_general_accuracy(response, expected_answer)
    
    def _score_math_accuracy(self, response: str, expected_answer: float) -> float:
        """Score mathematical accuracy"""
        try:
            # Extract numbers from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if not numbers:
                return 0.0
            
            # Check if any extracted number matches expected answer
            for num_str in numbers:
                try:
                    num = float(num_str)
                    if abs(num - expected_answer) < 0.001:  # Allow small floating point errors
                        return 1.0
                except ValueError:
                    continue
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error scoring math accuracy: {e}")
            return 0.0
    
    def _score_logic_accuracy(self, response: str, expected_answer: str) -> float:
        """Score logical reasoning accuracy"""
        try:
            response_lower = response.lower().strip()
            expected_lower = str(expected_answer).lower().strip()
            
            # Check for exact match
            if expected_lower in response_lower:
                return 1.0
            
            # Check for common logical answer patterns
            if expected_lower == "true" and any(word in response_lower for word in ["yes", "correct", "true"]):
                return 1.0
            elif expected_lower == "false" and any(word in response_lower for word in ["no", "incorrect", "false"]):
                return 1.0
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error scoring logic accuracy: {e}")
            return 0.0
    
    def _score_general_accuracy(self, response: str, expected_answer: str) -> float:
        """Score general question accuracy using keyword matching"""
        try:
            response_lower = response.lower()
            expected_lower = str(expected_answer).lower()
            
            # Simple keyword overlap scoring
            expected_words = set(expected_lower.split())
            response_words = set(response_lower.split())
            
            if len(expected_words) == 0:
                return 0.5  # Neutral score for empty expected answer
            
            overlap = len(expected_words.intersection(response_words))
            return overlap / len(expected_words)
            
        except Exception as e:
            self.logger.error(f"Error scoring general accuracy: {e}")
            return 0.0
    
    def _score_reasoning_quality(self, response: str) -> float:
        """Score the quality of reasoning shown"""
        score = 0.0
        
        # Check for reasoning indicators
        reasoning_words = [
            "because", "therefore", "since", "thus", "hence", 
            "first", "second", "next", "then", "finally",
            "step", "approach", "method", "solution", "analysis"
        ]
        
        word_count = len(response.split())
        reasoning_count = sum(1 for word in reasoning_words if word in response.lower())
        
        # Score based on reasoning word density
        if word_count > 0:
            score += min(reasoning_count / word_count * 10, 0.5)  # Max 0.5 for word density
        
        # Check for structured reasoning
        if any(pattern in response.lower() for pattern in ["1.", "2.", "3.", "step 1", "step 2"]):
            score += 0.3
        
        # Check for explanation depth
        if word_count > 100:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_completeness(self, response: str, problem_type: str) -> float:
        """Score response completeness"""
        score = 0.0
        
        # Check for final answer
        if any(phrase in response.lower() for phrase in ["answer:", "final answer", "result:", "solution:"]):
            score += 0.4
        
        # Check for explanation
        if len(response.split()) > 20:
            score += 0.3
        
        # Check for problem-specific completeness
        if problem_type == "math":
            if any(word in response.lower() for word in ["calculate", "solve", "equation"]):
                score += 0.3
        elif problem_type == "logic":
            if any(word in response.lower() for word in ["logic", "reasoning", "conclusion"]):
                score += 0.3
        
        return min(score, 1.0)
    
    def _score_clarity(self, response: str) -> float:
        """Score response clarity and structure"""
        score = 0.0
        
        # Check for proper sentence structure
        sentences = response.split('.')
        if len(sentences) > 1:
            score += 0.3
        
        # Check for organization
        if any(marker in response for marker in ['\n', '1.', '2.', '-', '*']):
            score += 0.4
        
        # Penalize very short responses
        if len(response.split()) < 10:
            score -= 0.2
        
        # Penalize very long responses without structure
        if len(response.split()) > 500 and score < 0.5:
            score -= 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _count_reasoning_steps(self, response: str) -> int:
        """Count the number of reasoning steps in the response"""
        
        # Count numbered steps
        numbered_steps = len(re.findall(r'\d+\.', response))
        
        # Count bullet points
        bullet_steps = len(re.findall(r'[•\-\*]\s', response))
        
        # Count explicit step mentions
        step_mentions = len(re.findall(r'step \d+', response.lower()))
        
        return max(numbered_steps, bullet_steps, step_mentions)
    
    def calculate_overall_score(self, scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall score"""
        
        if weights is None:
            weights = {
                "accuracy": 0.4,
                "reasoning_quality": 0.25,
                "completeness": 0.2,
                "clarity": 0.15
            }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            if metric in weights:
                overall_score += score * weights[metric]
                total_weight += weights[metric]
        
        return overall_score / total_weight if total_weight > 0 else 0.0
