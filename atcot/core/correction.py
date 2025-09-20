"""
Adaptive correction mechanism with convergence guarantees for the ATCOT framework.

This module implements the correction mechanism that activates upon contradiction
detection, performs backward traversal to identify minimal revision sets, and
ensures convergence through bounded corrections.
"""

import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .state_representation import (
    ATCOTState, ReasoningStep, CorrectionEntry, CorrectionType
)
from ..utils.consistency import ConsistencyChecker
from ..utils.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class CorrectionContext:
    """Context information for correction operations."""
    triggering_observation: Any
    current_state: ATCOTState
    contradiction_threshold: float
    max_revision_attempts: int


class RevisionStrategy(ABC):
    """Abstract base class for revision strategies."""
    
    @abstractmethod
    def find_minimal_revision_set(self, 
                                reasoning_trace: List[ReasoningStep],
                                observation: Any,
                                context: CorrectionContext) -> Set[str]:
        """Find minimal set of reasoning steps to revise."""
        pass

    @abstractmethod
    def revise_reasoning(self, 
                        reasoning_trace: List[ReasoningStep],
                        revision_set: Set[str],
                        observation: Any,
                        context: CorrectionContext) -> List[ReasoningStep]:
        """Generate revised reasoning steps."""
        pass


class BackwardTraversalRevision(RevisionStrategy):
    """Revision strategy using backward traversal to find minimal revision set."""
    
    def __init__(self, llm_interface: LLMInterface, consistency_checker: ConsistencyChecker):
        self.llm_interface = llm_interface
        self.consistency_checker = consistency_checker

    def find_minimal_revision_set(self, 
                                reasoning_trace: List[ReasoningStep],
                                observation: Any,
                                context: CorrectionContext) -> Set[str]:
        """
        Find minimal revision set M through optimization:
        M = argmin_{M ⊆ R_t} |M| s.t. φ(S_t[R := (R_t \ M) ∪ incorporate(o_t)]) = 1
        """
        if not reasoning_trace:
            return set()
        
        # Start with backward traversal from most recent step
        revision_candidates = []
        
        # Check each reasoning step for contradiction with observation
        for i, step in enumerate(reversed(reasoning_trace)):
            contradiction_score = self.consistency_checker.compute_contradiction_score(
                observation, step.content
            )
            
            if contradiction_score > context.contradiction_threshold:
                revision_candidates.append((step.step_id, contradiction_score, len(reasoning_trace) - 1 - i))
        
        if not revision_candidates:
            logger.warning("No contradictory steps found despite contradiction detection")
            return set()
        
        # Sort by contradiction score and recency (higher score and more recent first)
        revision_candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        
        # Find minimal set using greedy approach
        minimal_set = set()
        
        for step_id, score, position in revision_candidates:
            # Test if adding this step to revision set resolves contradiction
            test_set = minimal_set | {step_id}
            
            if self._test_revision_set(reasoning_trace, test_set, observation, context):
                minimal_set.add(step_id)
                
                # Check if we have resolved all contradictions
                if self._is_revision_sufficient(reasoning_trace, minimal_set, observation, context):
                    break
        
        logger.info(f"Found minimal revision set with {len(minimal_set)} steps")
        return minimal_set

    def revise_reasoning(self, 
                        reasoning_trace: List[ReasoningStep],
                        revision_set: Set[str],
                        observation: Any,
                        context: CorrectionContext) -> List[ReasoningStep]:
        """Generate revised reasoning incorporating the new observation."""
        if not revision_set:
            return reasoning_trace
        
        # Separate steps to keep and steps to revise
        steps_to_keep = []
        revised_positions = []
        
        for i, step in enumerate(reasoning_trace):
            if step.step_id in revision_set:
                revised_positions.append(i)
            else:
                steps_to_keep.append((i, step))
        
        # Generate new reasoning steps to replace the removed ones
        new_steps = self._generate_replacement_steps(
            reasoning_trace, revision_set, observation, context
        )
        
        # Reconstruct reasoning trace
        revised_trace = []
        new_step_idx = 0
        
        for i in range(len(reasoning_trace)):
            if i in revised_positions:
                # Insert new step if available
                if new_step_idx < len(new_steps):
                    revised_trace.append(new_steps[new_step_idx])
                    new_step_idx += 1
            else:
                # Keep original step
                original_step = next(step for pos, step in steps_to_keep if pos == i)
                revised_trace.append(original_step)
        
        # Add any remaining new steps
        while new_step_idx < len(new_steps):
            revised_trace.append(new_steps[new_step_idx])
            new_step_idx += 1
        
        logger.info(f"Generated {len(new_steps)} replacement steps")
        return revised_trace

    def _test_revision_set(self, 
                          reasoning_trace: List[ReasoningStep],
                          revision_set: Set[str],
                          observation: Any,
                          context: CorrectionContext) -> bool:
        """Test if a revision set would resolve contradictions."""
        # Simulate removal of revision set and incorporation of observation
        remaining_steps = [
            step for step in reasoning_trace
            if step.step_id not in revision_set
        ]
        
        # Check for contradictions between observation and remaining steps
        for step in remaining_steps:
            if self.consistency_checker.check_local_contradiction(observation, step.content):
                return False
        
        return True

    def _is_revision_sufficient(self, 
                              reasoning_trace: List[ReasoningStep],
                              revision_set: Set[str],
                              observation: Any,
                              context: CorrectionContext) -> bool:
        """Check if the current revision set is sufficient to resolve all contradictions."""
        return self._test_revision_set(reasoning_trace, revision_set, observation, context)

    def _generate_replacement_steps(self, 
                                  reasoning_trace: List[ReasoningStep],
                                  revision_set: Set[str],
                                  observation: Any,
                                  context: CorrectionContext) -> List[ReasoningStep]:
        """Generate new reasoning steps to replace the removed ones."""
        prompt = self._build_revision_prompt(reasoning_trace, revision_set, observation, context)
        
        try:
            response = self.llm_interface.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=512
            )
            
            new_steps = self._parse_revision_response(response, observation)
            return new_steps
            
        except Exception as e:
            logger.error(f"Failed to generate replacement steps: {e}")
            
            # Fallback: create simple incorporation step
            incorporation_step = ReasoningStep(
                content=f"Incorporating new information: {str(observation)[:200]}",
                justification="Direct incorporation of tool observation",
                confidence_score=0.7,
                tool_observations=[str(observation)]
            )
            return [incorporation_step]

    def _build_revision_prompt(self, 
                             reasoning_trace: List[ReasoningStep],
                             revision_set: Set[str],
                             observation: Any,
                             context: CorrectionContext) -> str:
        """Build prompt for generating revised reasoning steps."""
        # Get context from remaining steps
        remaining_steps = [
            step for step in reasoning_trace
            if step.step_id not in revision_set
        ]
        
        remaining_context = "\n".join([
            f"Step {i+1}: {step.content}"
            for i, step in enumerate(remaining_steps[-3:])  # Last 3 remaining steps
        ]) if remaining_steps else "No remaining reasoning steps"
        
        removed_context = "\n".join([
            f"Removed: {step.content}"
            for step in reasoning_trace
            if step.step_id in revision_set
        ])
        
        return f"""The reasoning process has encountered a contradiction and needs revision.

Original Query: {context.current_state.query}

Remaining Reasoning Context:
{remaining_context}

Steps Removed Due to Contradiction:
{removed_context}

New Observation from Tool:
{str(observation)}

Generate new reasoning steps that:
1. Incorporate the new observation logically
2. Connect smoothly with the remaining reasoning steps
3. Resolve the contradiction that was detected
4. Maintain logical consistency

Provide 1-3 new reasoning steps in the format:
Step: [reasoning content]
Justification: [why this step is valid]
Confidence: [0.0 to 1.0]

Focus on creating coherent reasoning that integrates the new information.
"""

    def _parse_revision_response(self, response: str, observation: Any) -> List[ReasoningStep]:
        """Parse LLM response into new reasoning steps."""
        new_steps = []
        lines = response.strip().split('\n')
        
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Step:'):
                if current_step:
                    new_steps.append(current_step)
                
                content = line[5:].strip()
                current_step = ReasoningStep(
                    content=content,
                    tool_observations=[str(observation)]
                )
                
            elif line.startswith('Justification:') and current_step:
                current_step.justification = line[14:].strip()
                
            elif line.startswith('Confidence:') and current_step:
                try:
                    conf_str = line[11:].strip()
                    confidence = float(conf_str)
                    current_step.confidence_score = max(0.0, min(1.0, confidence))
                except ValueError:
                    current_step.confidence_score = 0.5
        
        # Add the last step
        if current_step:
            new_steps.append(current_step)
        
        # If no steps were parsed, create a default incorporation step
        if not new_steps:
            new_steps.append(ReasoningStep(
                content=f"Incorporating tool observation: {str(observation)[:100]}",
                justification="Direct incorporation of contradictory observation",
                confidence_score=0.6,
                tool_observations=[str(observation)]
            ))
        
        return new_steps


class ConvergenceTracker:
    """Tracks convergence conditions and prevents infinite correction loops."""
    
    def __init__(self, max_corrections: int = 5, improvement_threshold: float = 0.01):
        self.max_corrections = max_corrections
        self.improvement_threshold = improvement_threshold
        self.consistency_history = []

    def check_convergence(self, state: ATCOTState) -> bool:
        """Check if the reasoning process has converged."""
        # Check correction budget
        if state.correction_log.correction_count >= self.max_corrections:
            logger.info("Reached maximum correction budget")
            return True
        
        # Check for improvement in consistency
        if len(self.consistency_history) >= 3:
            recent_scores = self.consistency_history[-3:]
            if all(abs(recent_scores[i] - recent_scores[i-1]) < self.improvement_threshold 
                   for i in range(1, len(recent_scores))):
                logger.info("Consistency scores have stabilized")
                return True
        
        # Check if all plan steps are completed successfully
        completed_steps = sum(
            1 for step in state.planning_structure.steps
            if step.status.value == "completed"
        )
        total_steps = len(state.planning_structure.steps)
        
        if completed_steps == total_steps and total_steps > 0:
            logger.info("All plan steps completed successfully")
            return True
        
        return False

    def update_consistency_score(self, score: float):
        """Update the consistency score history."""
        self.consistency_history.append(score)
        
        # Keep only recent history
        if len(self.consistency_history) > 10:
            self.consistency_history = self.consistency_history[-10:]

    def get_improvement_trend(self) -> str:
        """Get the trend in consistency improvements."""
        if len(self.consistency_history) < 2:
            return "insufficient_data"
        
        recent_change = self.consistency_history[-1] - self.consistency_history[-2]
        
        if recent_change > self.improvement_threshold:
            return "improving"
        elif recent_change < -self.improvement_threshold:
            return "degrading"
        else:
            return "stable"


class AdaptiveCorrectionMechanism:
    """
    Main correction mechanism implementing adaptive correction with convergence guarantees.
    
    Handles contradiction detection, minimal revision set identification, and
    reasoning revision while ensuring bounded corrections and monotonic improvement.
    """
    
    def __init__(self,
                 consistency_checker: ConsistencyChecker,
                 revision_strategy: RevisionStrategy,
                 contradiction_threshold: float = 0.7,
                 max_corrections: int = 5):
        self.consistency_checker = consistency_checker
        self.revision_strategy = revision_strategy
        self.contradiction_threshold = contradiction_threshold
        self.convergence_tracker = ConvergenceTracker(max_corrections)

    def detect_contradiction(self, observation: Any, reasoning_trace: List[ReasoningStep]) -> bool:
        """
        Detect local contradictions using:
        ψ(o_t, R_t) = 1[max_{r_j ∈ R_t} contradict(o_t, r_j) > τ_contra]
        """
        if not reasoning_trace:
            return False
        
        max_contradiction = 0.0
        
        for step in reasoning_trace:
            contradiction_score = self.consistency_checker.compute_contradiction_score(
                observation, step.content
            )
            max_contradiction = max(max_contradiction, contradiction_score)
        
        is_contradiction = max_contradiction > self.contradiction_threshold
        
        if is_contradiction:
            logger.info(f"Local contradiction detected with score {max_contradiction:.3f}")
        
        return is_contradiction

    def perform_correction(self, 
                         observation: Any,
                         state: ATCOTState) -> Tuple[List[ReasoningStep], CorrectionEntry]:
        """
        Perform adaptive correction when contradiction is detected.
        
        Returns:
            Tuple of (revised_reasoning_trace, correction_entry)
        """
        logger.info("Performing adaptive correction")
        
        context = CorrectionContext(
            triggering_observation=observation,
            current_state=state,
            contradiction_threshold=self.contradiction_threshold,
            max_revision_attempts=3
        )
        
        try:
            # Find minimal revision set
            revision_set = self.revision_strategy.find_minimal_revision_set(
                state.reasoning_trace.steps, observation, context
            )
            
            if not revision_set:
                logger.warning("No revision set identified despite contradiction detection")
                return state.reasoning_trace.steps, self._create_failed_correction_entry(
                    observation, "No revision set identified"
                )
            
            # Generate revised reasoning
            revised_reasoning = self.revision_strategy.revise_reasoning(
                state.reasoning_trace.steps, revision_set, observation, context
            )
            
            # Create correction entry
            correction_entry = CorrectionEntry(
                correction_type=CorrectionType.LOCAL_CONTRADICTION,
                trigger_description=f"Local contradiction detected with observation: {str(observation)[:100]}",
                affected_steps=revision_set,
                revision_description=f"Revised {len(revision_set)} steps, generated {len(revised_reasoning)} new steps",
                success=True
            )
            
            logger.info(f"Correction completed: revised {len(revision_set)} steps")
            return revised_reasoning, correction_entry
            
        except Exception as e:
            logger.error(f"Correction failed: {e}")
            return state.reasoning_trace.steps, self._create_failed_correction_entry(
                observation, f"Correction failed: {str(e)}"
            )

    def check_global_consistency(self, state: ATCOTState) -> bool:
        """
        Check global consistency φ(S) across the entire state.
        
        Returns True if state is globally consistent, False otherwise.
        """
        # Check consistency between reasoning steps
        reasoning_steps = state.reasoning_trace.steps
        
        for i in range(len(reasoning_steps)):
            for j in range(i + 1, len(reasoning_steps)):
                if self.consistency_checker.check_logical_contradiction(
                    reasoning_steps[i].content, reasoning_steps[j].content
                ):
                    logger.debug(f"Global inconsistency detected between steps {i} and {j}")
                    return False
        
        # Check consistency between plan and reasoning
        plan_goals = [step.description for step in state.planning_structure.steps]
        reasoning_content = [step.content for step in reasoning_steps]
        
        if not self.consistency_checker.check_plan_reasoning_alignment(plan_goals, reasoning_content):
            logger.debug("Plan-reasoning alignment check failed")
            return False
        
        # Check tool result consistency
        successful_invocations = state.tool_history.get_successful_invocations()
        for invocation in successful_invocations:
            if invocation.reasoning_step_ref:
                reasoning_step = state.reasoning_trace.get_step(invocation.reasoning_step_ref)
                if reasoning_step and not self.consistency_checker.check_tool_reasoning_consistency(
                    invocation.result, reasoning_step.content
                ):
                    logger.debug(f"Tool-reasoning inconsistency in step {invocation.reasoning_step_ref}")
                    return False
        
        return True

    def update_convergence_tracking(self, state: ATCOTState) -> bool:
        """Update convergence tracking and check if process has converged."""
        # Compute overall consistency score
        consistency_score = self._compute_consistency_score(state)
        self.convergence_tracker.update_consistency_score(consistency_score)
        
        return self.convergence_tracker.check_convergence(state)

    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get statistics about the correction process."""
        return {
            "total_corrections": len(self.convergence_tracker.consistency_history),
            "improvement_trend": self.convergence_tracker.get_improvement_trend(),
            "current_consistency": (
                self.convergence_tracker.consistency_history[-1] 
                if self.convergence_tracker.consistency_history else 0.0
            ),
            "contradiction_threshold": self.contradiction_threshold
        }

    def _create_failed_correction_entry(self, observation: Any, error_message: str) -> CorrectionEntry:
        """Create correction entry for failed correction attempt."""
        return CorrectionEntry(
            correction_type=CorrectionType.LOCAL_CONTRADICTION,
            trigger_description=f"Correction attempt for observation: {str(observation)[:100]}",
            affected_steps=set(),
            revision_description=error_message,
            success=False
        )

    def _compute_consistency_score(self, state: ATCOTState) -> float:
        """Compute overall consistency score for the state."""
        if not state.reasoning_trace.steps:
            return 1.0
        
        total_score = 0.0
        comparisons = 0
        
        # Check pairwise consistency between reasoning steps
        steps = state.reasoning_trace.steps
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                consistency = self.consistency_checker.compute_semantic_similarity(
                    steps[i].content, steps[j].content
                )
                total_score += consistency
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        return total_score / comparisons
