"""
Consistency checking utilities for the ATCOT framework.

This module implements various consistency checking mechanisms including
contradiction detection, semantic similarity computation, and logical
coherence verification.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..core.state_representation import ReasoningTrace, ReasoningStep

logger = logging.getLogger(__name__)


class SemanticSimilarityComputer(ABC):
    """Abstract base class for semantic similarity computation."""
    
    @abstractmethod
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        pass


class SimpleSemanticSimilarity(SemanticSimilarityComputer):
    """Simple semantic similarity based on keyword overlap."""
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using keyword overlap."""
        try:
            # Normalize texts
            words1 = set(self._tokenize(text1.lower()))
            words2 = set(self._tokenize(text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            # Compute Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to compute similarity: {e}")
            return 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word for word in text.split() if len(word) > 2]


class ContradictionDetector(ABC):
    """Abstract base class for contradiction detection."""
    
    @abstractmethod
    def detect_contradiction(self, text1: str, text2: str) -> float:
        """Detect contradiction between two texts, return score [0, 1]."""
        pass


class SimpleContradictionDetector(ContradictionDetector):
    """Simple contradiction detector using keyword-based heuristics."""
    
    def __init__(self):
        # Define contradiction patterns
        self.negation_patterns = [
            r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bfalse\b',
            r'\bincorrect\b', r'\bwrong\b', r'\binvalid\b'
        ]
        
        self.affirmation_patterns = [
            r'\byes\b', r'\btrue\b', r'\bcorrect\b', r'\bvalid\b',
            r'\baccurate\b', r'\bright\b'
        ]
        
        self.numeric_pattern = r'[-+]?\d*\.?\d+'

    def detect_contradiction(self, text1: str, text2: str) -> float:
        """Detect contradiction using pattern matching."""
        try:
            text1_lower = text1.lower()
            text2_lower = text2.lower()
            
            contradiction_score = 0.0
            
            # Check for direct negation patterns
            negation_score = self._check_negation_contradiction(text1_lower, text2_lower)
            contradiction_score = max(contradiction_score, negation_score)
            
            # Check for numeric contradictions
            numeric_score = self._check_numeric_contradiction(text1, text2)
            contradiction_score = max(contradiction_score, numeric_score)
            
            # Check for logical contradictions
            logical_score = self._check_logical_contradiction(text1_lower, text2_lower)
            contradiction_score = max(contradiction_score, logical_score)
            
            return min(1.0, contradiction_score)
            
        except Exception as e:
            logger.warning(f"Failed to detect contradiction: {e}")
            return 0.0

    def _check_negation_contradiction(self, text1: str, text2: str) -> float:
        """Check for negation-based contradictions."""
        # Count negations in each text
        negations1 = sum(1 for pattern in self.negation_patterns if re.search(pattern, text1))
        negations2 = sum(1 for pattern in self.negation_patterns if re.search(pattern, text2))
        
        affirmations1 = sum(1 for pattern in self.affirmation_patterns if re.search(pattern, text1))
        affirmations2 = sum(1 for pattern in self.affirmation_patterns if re.search(pattern, text2))
        
        # High contradiction if one text has negations and the other has affirmations
        if (negations1 > 0 and affirmations2 > 0) or (affirmations1 > 0 and negations2 > 0):
            return 0.7
        
        return 0.0

    def _check_numeric_contradiction(self, text1: str, text2: str) -> float:
        """Check for numeric contradictions."""
        numbers1 = re.findall(self.numeric_pattern, text1)
        numbers2 = re.findall(self.numeric_pattern, text2)
        
        if not numbers1 or not numbers2:
            return 0.0
        
        try:
            # Convert to floats
            nums1 = [float(n) for n in numbers1]
            nums2 = [float(n) for n in numbers2]
            
            # Check for significant differences
            for n1 in nums1:
                for n2 in nums2:
                    if abs(n1 - n2) > max(abs(n1), abs(n2)) * 0.1:  # >10% difference
                        return 0.6
            
        except ValueError:
            pass
        
        return 0.0

    def _check_logical_contradiction(self, text1: str, text2: str) -> float:
        """Check for logical contradictions."""
        # Simple heuristic: if texts share keywords but have opposite conclusions
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        
        if overlap > 0.3:  # Significant overlap
            # Check if conclusions are opposite
            if self._has_opposite_conclusions(text1, text2):
                return 0.5
        
        return 0.0

    def _has_opposite_conclusions(self, text1: str, text2: str) -> bool:
        """Check if texts have opposite conclusions."""
        conclusion_words1 = self._extract_conclusion_words(text1)
        conclusion_words2 = self._extract_conclusion_words(text2)
        
        # Check for contradictory pairs
        contradictory_pairs = [
            ('true', 'false'), ('correct', 'incorrect'), ('yes', 'no'),
            ('valid', 'invalid'), ('right', 'wrong'), ('possible', 'impossible')
        ]
        
        for word1 in conclusion_words1:
            for word2 in conclusion_words2:
                for pair in contradictory_pairs:
                    if (word1 in pair and word2 in pair and word1 != word2):
                        return True
        
        return False

    def _extract_conclusion_words(self, text: str) -> List[str]:
        """Extract words that typically indicate conclusions."""
        conclusion_indicators = [
            'therefore', 'thus', 'hence', 'consequently', 'so',
            'conclusion', 'result', 'answer', 'final'
        ]
        
        words = []
        sentences = text.split('.')
        
        for sentence in sentences:
            for indicator in conclusion_indicators:
                if indicator in sentence.lower():
                    words.extend(sentence.lower().split())
                    break
        
        return words


class ConsistencyChecker:
    """
    Main consistency checker implementing various verification mechanisms.
    
    Provides methods for local contradiction detection, global consistency
    verification, and semantic alignment checking.
    """
    
    def __init__(self,
                 similarity_computer: Optional[SemanticSimilarityComputer] = None,
                 contradiction_detector: Optional[ContradictionDetector] = None,
                 contradiction_threshold: float = 0.7):
        self.similarity_computer = similarity_computer or SimpleSemanticSimilarity()
        self.contradiction_detector = contradiction_detector or SimpleContradictionDetector()
        self.contradiction_threshold = contradiction_threshold

    def check_local_contradiction(self, observation: Any, reasoning_trace: ReasoningTrace) -> bool:
        """
        Check for local contradictions between observation and reasoning trace.
        
        Returns True if contradiction is detected, False otherwise.
        """
        if not reasoning_trace.steps:
            return False
        
        observation_text = str(observation)
        
        for step in reasoning_trace.steps:
            contradiction_score = self.contradiction_detector.detect_contradiction(
                observation_text, step.content
            )
            
            if contradiction_score > self.contradiction_threshold:
                logger.debug(f"Local contradiction detected: score={contradiction_score:.3f}")
                return True
        
        return False

    def check_logical_contradiction(self, text1: str, text2: str) -> bool:
        """Check for logical contradiction between two texts."""
        contradiction_score = self.contradiction_detector.detect_contradiction(text1, text2)
        return contradiction_score > self.contradiction_threshold

    def compute_contradiction_score(self, observation: Any, reasoning_content: str) -> float:
        """Compute contradiction score between observation and reasoning content."""
        observation_text = str(observation)
        return self.contradiction_detector.detect_contradiction(observation_text, reasoning_content)

    def compute_semantic_similarity(self, observation: Any, reasoning_trace: ReasoningTrace) -> float:
        """Compute semantic similarity between observation and reasoning trace."""
        if not reasoning_trace.steps:
            return 0.0
        
        observation_text = str(observation)
        
        # Compute similarity with recent reasoning steps
        recent_steps = reasoning_trace.get_latest_steps(3)
        total_similarity = 0.0
        
        for step in recent_steps:
            similarity = self.similarity_computer.compute_similarity(
                observation_text, step.content
            )
            total_similarity += similarity
        
        return total_similarity / len(recent_steps) if recent_steps else 0.0

    def check_plan_reasoning_alignment(self, plan_goals: List[str], reasoning_content: List[str]) -> bool:
        """Check if plan goals align with reasoning content."""
        if not plan_goals or not reasoning_content:
            return True  # Vacuously true
        
        # Compute overall similarity between plan and reasoning
        plan_text = " ".join(plan_goals)
        reasoning_text = " ".join(reasoning_content)
        
        similarity = self.similarity_computer.compute_similarity(plan_text, reasoning_text)
        
        # Require at least 30% similarity for alignment
        return similarity >= 0.3

    def check_tool_reasoning_consistency(self, tool_result: Any, reasoning_content: str) -> bool:
        """Check consistency between tool result and reasoning content."""
        tool_text = str(tool_result)
        
        # Check for contradiction
        contradiction_score = self.contradiction_detector.detect_contradiction(
            tool_text, reasoning_content
        )
        
        # Also check for reasonable similarity
        similarity = self.similarity_computer.compute_similarity(tool_text, reasoning_content)
        
        # Consistent if no strong contradiction and some similarity
        return contradiction_score < self.contradiction_threshold and similarity > 0.1

    def evaluate_overall_consistency(self, reasoning_trace: ReasoningTrace) -> float:
        """Evaluate the overall consistency of a reasoning trace."""
        if len(reasoning_trace.steps) < 2:
            return 1.0
        
        total_consistency = 0.0
        comparisons = 0
        
        # Check pairwise consistency between steps
        for i in range(len(reasoning_trace.steps)):
            for j in range(i + 1, len(reasoning_trace.steps)):
                step1 = reasoning_trace.steps[i]
                step2 = reasoning_trace.steps[j]
                
                # Consistency = 1 - contradiction_score
                contradiction = self.contradiction_detector.detect_contradiction(
                    step1.content, step2.content
                )
                consistency = 1.0 - contradiction
                
                total_consistency += consistency
                comparisons += 1
        
        return total_consistency / comparisons if comparisons > 0 else 1.0

    def get_consistency_report(self, reasoning_trace: ReasoningTrace) -> Dict[str, Any]:
        """Generate a detailed consistency report."""
        if not reasoning_trace.steps:
            return {"overall_consistency": 1.0, "step_count": 0, "contradictions": []}
        
        contradictions = []
        step_consistencies = []
        
        # Check each step against all others
        for i, step1 in enumerate(reasoning_trace.steps):
            step_score = 0.0
            step_comparisons = 0
            
            for j, step2 in enumerate(reasoning_trace.steps):
                if i != j:
                    contradiction_score = self.contradiction_detector.detect_contradiction(
                        step1.content, step2.content
                    )
                    
                    if contradiction_score > self.contradiction_threshold:
                        contradictions.append({
                            "step1_index": i,
                            "step2_index": j,
                            "contradiction_score": contradiction_score,
                            "step1_content": step1.content[:100] + "...",
                            "step2_content": step2.content[:100] + "..."
                        })
                    
                    step_score += (1.0 - contradiction_score)
                    step_comparisons += 1
            
            if step_comparisons > 0:
                step_consistencies.append(step_score / step_comparisons)
            else:
                step_consistencies.append(1.0)
        
        overall_consistency = sum(step_consistencies) / len(step_consistencies)
        
        return {
            "overall_consistency": overall_consistency,
            "step_count": len(reasoning_trace.steps),
            "contradictions": contradictions,
            "step_consistencies": step_consistencies,
            "contradiction_threshold": self.contradiction_threshold
        }
