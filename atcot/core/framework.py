"""
Main ATCOT framework implementing the complete adaptive reasoning algorithm.

This module implements the core ATCOT execution algorithm as described in the
methodology, integrating planning, execution, and correction with convergence guarantees.
"""

import asyncio
import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass

from .state_representation import (
    ATCOTState, PlanStep, ReasoningStep, ToolInvocation, CorrectionEntry,
    StepStatus, CorrectionType
)
from .planning import AdaptivePlanner
from .execution import ToolAugmentedExecutor, CandidateObservation
from .correction import AdaptiveCorrectionMechanism
from ..tools.base import ToolRegistry
from ..utils.consistency import ConsistencyChecker
from ..utils.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class ATCOTResult:
    """Result of ATCOT execution."""
    final_answer: str
    state: ATCOTState
    converged: bool
    total_corrections: int
    execution_time_ms: float
    statistics: Dict[str, Any]


class ATCOTFramework:
    """
    Main ATCOT framework implementing adaptive tool-augmented chain of thought reasoning.
    
    This class orchestrates the complete reasoning process including planning,
    execution, and correction as described in Algorithm 1 of the methodology.
    """
    
    def __init__(self,
                 llm_interface: LLMInterface,
                 tool_registry: ToolRegistry,
                 consistency_checker: ConsistencyChecker,
                 adaptive_planner: AdaptivePlanner,
                 executor: ToolAugmentedExecutor,
                 correction_mechanism: AdaptiveCorrectionMechanism,
                 max_correction_budget: int = 5):
        self.llm_interface = llm_interface
        self.tool_registry = tool_registry
        self.consistency_checker = consistency_checker
        self.adaptive_planner = adaptive_planner
        self.executor = executor
        self.correction_mechanism = correction_mechanism
        self.max_correction_budget = max_correction_budget

    async def execute(self, query: str) -> ATCOTResult:
        """
        Execute the complete ATCOT reasoning process.
        
        Implements Algorithm 1: ATCOT Execution with Adaptive Correction
        
        Args:
            query: The input query to solve
            
        Returns:
            ATCOTResult containing the final answer and execution details
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting ATCOT execution for query: {query[:100]}...")
        
        # Initialize state S ← InitializeState(q)
        state = self._initialize_state(query)
        
        # Generate initial plan P ← GeneratePlan(q, S)
        available_tools = self.tool_registry.get_tool_names()
        initial_plan = self.adaptive_planner.generate_initial_plan(state, available_tools)
        state.planning_structure = initial_plan
        
        corrections = 0
        
        # Main execution loop
        while not self._converged(state) and corrections < self.max_correction_budget:
            logger.debug(f"Execution iteration {corrections + 1}")
            
            # Execute pending steps
            await self._execute_pending_steps(state)
            
            # Global consistency check
            if not self.correction_mechanism.check_global_consistency(state):
                logger.info("Global consistency check failed, performing replanning")
                
                # Perform replanning
                new_plan, correction_entry = self.adaptive_planner.perform_replanning(
                    state.planning_structure, state
                )
                state.planning_structure = new_plan
                state.correction_log.add_correction(correction_entry)
                corrections += 1
            
            # Update convergence tracking
            converged = self.correction_mechanism.update_convergence_tracking(state)
            if converged:
                state.converged = True
                break
        
        # Generate final answer
        final_answer = await self._generate_final_answer(state)
        state.final_answer = final_answer
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Compile statistics
        statistics = self._compile_execution_statistics(state, corrections, execution_time)
        
        logger.info(f"ATCOT execution completed in {execution_time:.2f}ms with {corrections} corrections")
        
        return ATCOTResult(
            final_answer=final_answer,
            state=state,
            converged=state.converged,
            total_corrections=corrections,
            execution_time_ms=execution_time,
            statistics=statistics
        )

    async def _execute_pending_steps(self, state: ATCOTState) -> None:
        """
        Execute all pending steps in the current plan.
        
        For each step p_i ∈ P that is pending:
        1. Execute candidate tools and select best observation
        2. Check for local contradictions
        3. Perform correction if needed
        4. Update state
        """
        ready_steps = state.planning_structure.get_ready_steps()
        
        for step in ready_steps:
            logger.debug(f"Executing step: {step.description}")
            
            # Execute step and get candidate observations
            best_observation, requires_correction = await self.executor.execute_step(step, state)
            
            if best_observation is None:
                logger.warning(f"Step execution failed: {step.description}")
                continue
            
            # Default: no change to reasoning trace
            reasoning_trace_updated = state.reasoning_trace.steps.copy()
            
            # Check for local contradiction
            if requires_correction and best_observation.success:
                logger.info("Local contradiction detected, performing correction")
                
                # Find minimal revision set and revise reasoning
                revised_reasoning, correction_entry = self.correction_mechanism.perform_correction(
                    best_observation.content, state
                )
                
                # Update reasoning trace
                state.reasoning_trace.steps = revised_reasoning
                state.correction_log.add_correction(correction_entry)
                
                # Adapt plan for removed reasoning steps
                if correction_entry.affected_steps:
                    adapted_plan = self.adaptive_planner.adapt_plan_for_correction(
                        state.planning_structure, correction_entry.affected_steps, state
                    )
                    state.planning_structure = adapted_plan
                    
                    # Update dependency graph
                    self.executor.prune_dependency_graph(correction_entry.affected_steps)
            else:
                # No contradiction, append observation to reasoning trace
                if best_observation.success and best_observation.content:
                    reasoning_step = ReasoningStep(
                        content=str(best_observation.content),
                        justification=f"Result from {best_observation.tool_name} tool execution",
                        confidence_score=0.8,
                        plan_step_ref=step.step_id,
                        tool_observations=[str(best_observation.content)]
                    )
                    state.reasoning_trace.append_step(reasoning_step)
            
            # Update dependency graph for completed step
            dependency_ids = [
                dep_step.step_id for dep_step in state.planning_structure.steps
                if dep_step.step_id in step.dependencies
            ]
            self.executor.update_dependency_graph(step.step_id, dependency_ids)
            
            # Record tool invocation in history (already done in executor)
            # Update state timestamp
            state.update_state()

    def _initialize_state(self, query: str) -> ATCOTState:
        """Initialize the ATCOT state S for the given query."""
        state = ATCOTState(query=query)
        
        logger.debug("Initialized ATCOT state")
        return state

    def _converged(self, state: ATCOTState) -> bool:
        """Check if the reasoning process has converged."""
        # Check if all plan steps are completed
        all_completed = all(
            step.status in [StepStatus.COMPLETED, StepStatus.CANCELLED]
            for step in state.planning_structure.steps
        )
        
        if not all_completed:
            return False
        
        # Check convergence through correction mechanism
        return self.correction_mechanism.update_convergence_tracking(state)

    async def _generate_final_answer(self, state: ATCOTState) -> str:
        """Generate the final answer based on the current state."""
        prompt = self._build_answer_generation_prompt(state)
        
        try:
            response = await self.llm_interface.generate_async(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512
            )
            
            # Extract and clean the final answer
            answer = self._extract_final_answer(response)
            logger.info(f"Generated final answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate final answer: {e}")
            
            # Fallback: use the last reasoning step as answer
            if state.reasoning_trace.steps:
                return state.reasoning_trace.steps[-1].content
            else:
                return "Unable to generate answer due to execution failure."

    def _build_answer_generation_prompt(self, state: ATCOTState) -> str:
        """Build prompt for final answer generation."""
        # Get key reasoning steps
        reasoning_summary = []
        for i, step in enumerate(state.reasoning_trace.steps):
            reasoning_summary.append(f"{i+1}. {step.content}")
        
        reasoning_text = "\n".join(reasoning_summary) if reasoning_summary else "No reasoning steps completed"
        
        # Get tool results summary
        successful_tools = state.tool_history.get_successful_invocations()
        tool_summary = []
        for inv in successful_tools[-5:]:  # Last 5 successful tool calls
            tool_summary.append(f"- {inv.tool_name}: {str(inv.result)[:100]}")
        
        tool_text = "\n".join(tool_summary) if tool_summary else "No successful tool executions"
        
        return f"""Based on the following reasoning process and tool results, provide a final answer to the original query.

Original Query: {state.query}

Reasoning Steps:
{reasoning_text}

Tool Results:
{tool_text}

Corrections Applied: {state.correction_log.correction_count}

Please provide a clear, concise final answer that directly addresses the original query.
If the reasoning is incomplete or contradictory, acknowledge the limitations.

Final Answer:"""

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the LLM response."""
        # Look for "Final Answer:" or similar markers
        lines = response.strip().split('\n')
        
        # Find the line containing the answer
        answer_line = None
        for line in lines:
            line = line.strip()
            if line.lower().startswith('final answer:'):
                answer_line = line[13:].strip()
                break
            elif line.lower().startswith('answer:'):
                answer_line = line[7:].strip()
                break
        
        if answer_line:
            return answer_line
        
        # Fallback: return the last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line and not line.endswith(':'):
                return line
        
        return response.strip()

    def _compile_execution_statistics(self, 
                                    state: ATCOTState, 
                                    corrections: int, 
                                    execution_time: float) -> Dict[str, Any]:
        """Compile comprehensive execution statistics."""
        return {
            # Basic metrics
            "execution_time_ms": execution_time,
            "total_corrections": corrections,
            "converged": state.converged,
            
            # Planning metrics
            "total_plan_steps": len(state.planning_structure.steps),
            "completed_steps": sum(
                1 for step in state.planning_structure.steps
                if step.status == StepStatus.COMPLETED
            ),
            "failed_steps": sum(
                1 for step in state.planning_structure.steps
                if step.status == StepStatus.FAILED
            ),
            
            # Reasoning metrics
            "reasoning_steps": len(state.reasoning_trace.steps),
            "average_confidence": (
                sum(step.confidence_score for step in state.reasoning_trace.steps) / 
                len(state.reasoning_trace.steps)
                if state.reasoning_trace.steps else 0.0
            ),
            
            # Tool usage metrics
            "total_tool_invocations": len(state.tool_history.invocations),
            "successful_tool_calls": len(state.tool_history.get_successful_invocations()),
            "tool_usage_distribution": state.tool_history.tool_usage_stats,
            "average_tool_execution_time": (
                sum(inv.execution_time_ms for inv in state.tool_history.invocations) /
                len(state.tool_history.invocations)
                if state.tool_history.invocations else 0.0
            ),
            
            # Correction metrics
            "correction_types": {
                correction_type.value: len(state.correction_log.get_corrections_by_type(correction_type))
                for correction_type in CorrectionType
            },
            "successful_corrections": len(state.correction_log.get_successful_corrections()),
            
            # Dependency metrics
            "dependency_graph_stats": self.executor.get_execution_statistics(),
            
            # Consistency metrics
            "final_global_consistency": self.correction_mechanism.check_global_consistency(state),
            "correction_statistics": self.correction_mechanism.get_correction_statistics()
        }

    async def debug_execution(self, query: str, debug_callback=None) -> ATCOTResult:
        """
        Execute ATCOT with detailed debugging information.
        
        Args:
            query: The input query
            debug_callback: Optional callback function for debugging info
            
        Returns:
            ATCOTResult with detailed execution trace
        """
        def default_debug_callback(event_type: str, data: Dict[str, Any]):
            logger.debug(f"DEBUG [{event_type}]: {data}")
        
        callback = debug_callback or default_debug_callback
        
        # Execute with debugging
        callback("START", {"query": query})
        
        result = await self.execute(query)
        
        callback("COMPLETE", {
            "final_answer": result.final_answer,
            "converged": result.converged,
            "corrections": result.total_corrections,
            "execution_time": result.execution_time_ms
        })
        
        return result

    def get_framework_info(self) -> Dict[str, Any]:
        """Get information about the framework configuration."""
        return {
            "framework_version": "1.0.0",
            "max_correction_budget": self.max_correction_budget,
            "available_tools": self.tool_registry.get_tool_names(),
            "llm_interface": type(self.llm_interface).__name__,
            "consistency_checker": type(self.consistency_checker).__name__,
            "planning_strategy": type(self.adaptive_planner).__name__,
            "execution_strategy": type(self.executor).__name__,
            "correction_strategy": type(self.correction_mechanism).__name__
        }
