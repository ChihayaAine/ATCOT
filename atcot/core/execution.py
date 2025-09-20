"""
Tool-augmented execution with state tracking for the ATCOT framework.

This module implements the execution engine that manages tool selection, invocation,
candidate observation generation, and semantic consistency scoring as described
in the methodology.
"""

import asyncio
import logging
import time
from typing import List, Dict, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

from .state_representation import (
    ATCOTState, PlanStep, ReasoningStep, ToolInvocation, 
    StepStatus, ToolHistory
)
from ..tools.base import BaseTool, ToolRegistry
from ..utils.consistency import ConsistencyChecker
from ..utils.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class CandidateObservation:
    """A candidate observation from tool execution."""
    observation_id: str
    content: Any
    tool_name: str
    tool_args: Dict[str, Any]
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    reliability_score: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionContext:
    """Context for tool execution operations."""
    current_step: PlanStep
    state: ATCOTState
    available_tools: List[BaseTool]
    max_candidates: int = 3
    temperature: float = 1.0


class ToolSelector(ABC):
    """Abstract base class for tool selection strategies."""
    
    @abstractmethod
    def select_tool(self, step: PlanStep, state: ATCOTState, available_tools: List[BaseTool]) -> Optional[BaseTool]:
        """Select the most appropriate tool for the given step."""
        pass

    @abstractmethod
    def compute_selection_probability(self, tool: BaseTool, step: PlanStep, state: ATCOTState) -> float:
        """Compute P(τ|p_i, S_t, R_t) for tool selection."""
        pass


class LLMToolSelector(ToolSelector):
    """LLM-based tool selector using learned policy."""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

    def select_tool(self, step: PlanStep, state: ATCOTState, available_tools: List[BaseTool]) -> Optional[BaseTool]:
        """
        Select tool using: τ* = argmax_τ P(τ|p_i, S_t, R_t)
        """
        if not available_tools:
            return None
        
        # Compute probabilities for all tools
        tool_probabilities = []
        for tool in available_tools:
            prob = self.compute_selection_probability(tool, step, state)
            tool_probabilities.append((tool, prob))
        
        # Select tool with highest probability
        best_tool, best_prob = max(tool_probabilities, key=lambda x: x[1])
        
        logger.debug(f"Selected tool {best_tool.name} with probability {best_prob:.3f}")
        return best_tool

    def compute_selection_probability(self, tool: BaseTool, step: PlanStep, state: ATCOTState) -> float:
        """Compute tool selection probability using LLM."""
        prompt = self._build_selection_prompt(tool, step, state)
        
        try:
            response = self.llm_interface.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=100
            )
            
            # Parse probability from response
            probability = self._parse_probability_response(response)
            return probability
            
        except Exception as e:
            logger.warning(f"Failed to compute tool probability: {e}")
            # Fallback: use simple heuristic based on tool requirements
            if tool.name in step.tool_requirements:
                return 0.8
            return 0.1

    def _build_selection_prompt(self, tool: BaseTool, step: PlanStep, state: ATCOTState) -> str:
        """Build prompt for tool selection probability estimation."""
        recent_reasoning = state.reasoning_trace.get_latest_steps(3)
        reasoning_context = "\n".join([
            f"- {rs.content}" for rs in recent_reasoning
        ]) if recent_reasoning else "No previous reasoning steps"
        
        return f"""Estimate the probability that the following tool is appropriate for the given step.

Query: {state.query}
Current Step: {step.description}
Tool: {tool.name}
Tool Description: {tool.description}
Tool Capabilities: {', '.join(tool.capabilities)}

Recent Reasoning Context:
{reasoning_context}

Consider:
- How well the tool's capabilities match the step requirements
- Whether the tool has been useful in similar contexts
- The specificity and relevance of the tool for this step

Provide a probability score between 0.0 (completely inappropriate) and 1.0 (perfect match).
Respond with only the numerical score.
"""

    def _parse_probability_response(self, response: str) -> float:
        """Parse probability score from LLM response."""
        try:
            import re
            numbers = re.findall(r'(\d+\.?\d*)', response.strip())
            if numbers:
                prob = float(numbers[0])
                if prob > 1.0:
                    prob = prob / 100.0 if prob <= 100 else 1.0
                return max(0.0, min(1.0, prob))
            return 0.1
        except Exception:
            return 0.1


class ObservationGenerator:
    """Generates candidate observations through tool execution."""
    
    def __init__(self, tool_selector: ToolSelector, consistency_checker: ConsistencyChecker):
        self.tool_selector = tool_selector
        self.consistency_checker = consistency_checker

    async def generate_candidates(self, context: ExecutionContext) -> List[CandidateObservation]:
        """
        Generate candidate observations O_i = {ô_{i,1}, ..., ô_{i,K}}
        through parallel or sequential tool invocations.
        """
        candidates = []
        step = context.current_step
        state = context.state
        
        # Select primary tool
        primary_tool = self.tool_selector.select_tool(step, state, context.available_tools)
        if not primary_tool:
            logger.warning("No appropriate tool found for step")
            return candidates
        
        # Generate primary candidate
        primary_candidate = await self._execute_tool_candidate(
            primary_tool, step, state, is_primary=True
        )
        if primary_candidate:
            candidates.append(primary_candidate)
        
        # Generate additional candidates if requested
        if context.max_candidates > 1 and len(context.available_tools) > 1:
            # Select alternative tools
            alternative_tools = [
                tool for tool in context.available_tools
                if tool != primary_tool
            ]
            
            # Sort by selection probability and take top alternatives
            tool_probs = [
                (tool, self.tool_selector.compute_selection_probability(tool, step, state))
                for tool in alternative_tools
            ]
            tool_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Execute top alternatives in parallel
            alternative_tasks = []
            for tool, _ in tool_probs[:context.max_candidates - 1]:
                task = self._execute_tool_candidate(tool, step, state, is_primary=False)
                alternative_tasks.append(task)
            
            if alternative_tasks:
                alternative_results = await asyncio.gather(*alternative_tasks, return_exceptions=True)
                for result in alternative_results:
                    if isinstance(result, CandidateObservation):
                        candidates.append(result)
        
        # Compute reliability scores for all candidates
        self._compute_reliability_scores(candidates, state)
        
        logger.info(f"Generated {len(candidates)} candidate observations")
        return candidates

    async def _execute_tool_candidate(self, 
                                    tool: BaseTool, 
                                    step: PlanStep, 
                                    state: ATCOTState,
                                    is_primary: bool = True) -> Optional[CandidateObservation]:
        """Execute a single tool to generate a candidate observation."""
        try:
            # Extract arguments for tool execution
            args = await self._extract_tool_arguments(tool, step, state)
            
            # Execute tool with timing
            start_time = time.time()
            result = await tool.execute(args)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create candidate observation
            candidate = CandidateObservation(
                observation_id=f"{tool.name}_{step.step_id}_{int(time.time() * 1000)}",
                content=result.content if hasattr(result, 'content') else result,
                tool_name=tool.name,
                tool_args=args,
                execution_time_ms=execution_time,
                success=True,
                metadata={
                    "is_primary": is_primary,
                    "tool_confidence": getattr(result, 'confidence', 1.0)
                }
            )
            
            logger.debug(f"Successfully executed {tool.name} in {execution_time:.2f}ms")
            return candidate
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool.name}: {e}")
            return CandidateObservation(
                observation_id=f"{tool.name}_{step.step_id}_error",
                content=None,
                tool_name=tool.name,
                tool_args={},
                execution_time_ms=0.0,
                success=False,
                error_message=str(e)
            )

    async def _extract_tool_arguments(self, 
                                    tool: BaseTool, 
                                    step: PlanStep, 
                                    state: ATCOTState) -> Dict[str, Any]:
        """Extract appropriate arguments for tool execution."""
        # Use LLM to determine tool arguments based on step and context
        prompt = f"""Extract the appropriate arguments to call the {tool.name} tool for this step.

Query: {state.query}
Step: {step.description}
Tool: {tool.name}
Tool Schema: {tool.get_schema()}

Recent reasoning context:
{self._get_reasoning_context(state)}

Provide the arguments in JSON format. Only include the required parameters.
"""
        
        try:
            # For now, use a simple heuristic approach
            # In a full implementation, this would use the LLM interface
            args = {}
            
            if tool.name == "calculator":
                # Extract mathematical expressions from step description
                args["expression"] = self._extract_math_expression(step.description)
            elif tool.name == "web_search":
                # Extract search query from step description
                args["query"] = self._extract_search_query(step.description, state.query)
            elif tool.name == "python_interpreter":
                # Extract code from step description
                args["code"] = self._extract_code(step.description)
            
            return args
            
        except Exception as e:
            logger.warning(f"Failed to extract tool arguments: {e}")
            return {}

    def _compute_reliability_scores(self, candidates: List[CandidateObservation], state: ATCOTState):
        """
        Compute reliability scores using normalized semantic consistency:
        r_{i,k} = exp(sim(ô_{i,k}, R_t) / κ) / Σ exp(sim(ô_{i,k'}, R_t) / κ)
        """
        if not candidates:
            return
        
        temperature = 1.0  # κ parameter
        
        # Compute similarity scores for each candidate
        similarity_scores = []
        for candidate in candidates:
            if candidate.success and candidate.content:
                similarity = self.consistency_checker.compute_semantic_similarity(
                    candidate.content, state.reasoning_trace
                )
            else:
                similarity = 0.0
            similarity_scores.append(similarity)
        
        # Apply softmax normalization
        if max(similarity_scores) > 0:
            exp_scores = [math.exp(score / temperature) for score in similarity_scores]
            sum_exp = sum(exp_scores)
            
            for i, candidate in enumerate(candidates):
                candidate.reliability_score = exp_scores[i] / sum_exp if sum_exp > 0 else 0.0
        else:
            # All candidates have zero similarity, assign equal probabilities
            uniform_score = 1.0 / len(candidates)
            for candidate in candidates:
                candidate.reliability_score = uniform_score

    def _get_reasoning_context(self, state: ATCOTState) -> str:
        """Get recent reasoning context for tool argument extraction."""
        recent_steps = state.reasoning_trace.get_latest_steps(3)
        return "\n".join([f"- {step.content}" for step in recent_steps])

    def _extract_math_expression(self, description: str) -> str:
        """Extract mathematical expression from step description."""
        # Simple heuristic - look for mathematical patterns
        import re
        math_patterns = [
            r'calculate\s+([^.]+)',
            r'compute\s+([^.]+)', 
            r'solve\s+([^.]+)',
            r'(\d+[\+\-\*/\(\)\s\d\.]+\d+)'
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return description

    def _extract_search_query(self, description: str, original_query: str) -> str:
        """Extract search query from step description."""
        # Simple heuristic - use key terms from description and original query
        import re
        
        # Look for explicit search terms
        search_patterns = [
            r'search\s+for\s+([^.]+)',
            r'find\s+([^.]+)',
            r'look\s+up\s+([^.]+)'
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use original query
        return original_query

    def _extract_code(self, description: str) -> str:
        """Extract code from step description."""
        # Simple heuristic - look for code-related keywords
        if any(keyword in description.lower() for keyword in ['code', 'script', 'program', 'calculate']):
            return f"# {description}\nprint('Please implement the required calculation')"
        return description


class DependencyGraphManager:
    """Manages the directed acyclic graph G = (V, E) for execution dependencies."""
    
    def __init__(self):
        self.vertices = set()  # V: completed reasoning steps
        self.edges = set()     # E: information dependencies
        self.adjacency_list = {}

    def add_vertex(self, step_id: str):
        """Add a completed reasoning step as vertex."""
        self.vertices.add(step_id)
        if step_id not in self.adjacency_list:
            self.adjacency_list[step_id] = set()

    def add_edge(self, from_step: str, to_step: str):
        """Add dependency edge from_step -> to_step."""
        if from_step in self.vertices and to_step in self.vertices:
            self.edges.add((from_step, to_step))
            self.adjacency_list[from_step].add(to_step)

    def prune_and_reconnect(self, removed_steps: Set[str]) -> None:
        """
        Prune removed steps and reconnect graph maintaining dependencies.
        
        When reasoning steps in M are removed, this function ensures
        the dependency graph remains valid and connected.
        """
        # Remove vertices and associated edges
        for step_id in removed_steps:
            self.vertices.discard(step_id)
            self.adjacency_list.pop(step_id, None)
            
            # Remove edges involving this step
            edges_to_remove = [
                edge for edge in self.edges
                if edge[0] == step_id or edge[1] == step_id
            ]
            for edge in edges_to_remove:
                self.edges.remove(edge)
        
        # Update adjacency list
        for step_id in self.adjacency_list:
            self.adjacency_list[step_id] = {
                neighbor for neighbor in self.adjacency_list[step_id]
                if neighbor not in removed_steps
            }
        
        # Reconnect dependencies - if A -> B and B -> C, and B is removed,
        # we need to create A -> C if there's a logical dependency
        self._reconnect_dependencies(removed_steps)

    def _reconnect_dependencies(self, removed_steps: Set[str]):
        """Reconnect dependencies after removing steps."""
        # For each removed step, connect its predecessors to its successors
        # This is a simplified approach - a full implementation would need
        # semantic analysis to determine if the connection is valid
        
        for removed_step in removed_steps:
            # Find predecessors (steps that pointed to removed step)
            predecessors = [
                edge[0] for edge in self.edges
                if edge[1] == removed_step
            ]
            
            # Find successors (steps that removed step pointed to)  
            successors = [
                edge[1] for edge in self.edges
                if edge[0] == removed_step
            ]
            
            # Connect each predecessor to each successor
            for pred in predecessors:
                for succ in successors:
                    if pred != succ and pred in self.vertices and succ in self.vertices:
                        self.add_edge(pred, succ)

    def get_dependency_chain(self, step_id: str) -> List[str]:
        """Get the full dependency chain for a step."""
        if step_id not in self.vertices:
            return []
        
        visited = set()
        chain = []
        
        def dfs(current_step):
            if current_step in visited:
                return
            visited.add(current_step)
            chain.append(current_step)
            
            for neighbor in self.adjacency_list.get(current_step, set()):
                dfs(neighbor)
        
        dfs(step_id)
        return chain


class ToolAugmentedExecutor:
    """
    Main execution engine for tool-augmented reasoning with state tracking.
    
    Implements the core execution loop with candidate generation, selection,
    and state updates as described in the methodology.
    """
    
    def __init__(self,
                 tool_registry: ToolRegistry,
                 tool_selector: ToolSelector,
                 consistency_checker: ConsistencyChecker,
                 max_candidates_per_step: int = 3):
        self.tool_registry = tool_registry
        self.tool_selector = tool_selector
        self.consistency_checker = consistency_checker
        self.max_candidates_per_step = max_candidates_per_step
        self.observation_generator = ObservationGenerator(tool_selector, consistency_checker)
        self.dependency_graph = DependencyGraphManager()

    async def execute_step(self, 
                          step: PlanStep, 
                          state: ATCOTState) -> Tuple[Optional[CandidateObservation], bool]:
        """
        Execute a single plan step and return the best observation.
        
        Returns:
            Tuple of (selected_observation, requires_correction)
        """
        logger.info(f"Executing step: {step.description}")
        
        # Mark step as in progress
        step.update_status(StepStatus.IN_PROGRESS)
        
        try:
            # Get available tools for this step
            available_tools = self._get_available_tools(step)
            if not available_tools:
                logger.warning("No available tools for step")
                step.update_status(StepStatus.FAILED)
                return None, False
            
            # Generate candidate observations
            context = ExecutionContext(
                current_step=step,
                state=state,
                available_tools=available_tools,
                max_candidates=self.max_candidates_per_step
            )
            
            candidates = await self.observation_generator.generate_candidates(context)
            
            if not candidates:
                logger.warning("No candidate observations generated")
                step.update_status(StepStatus.FAILED)
                return None, False
            
            # Select best candidate by reliability score
            best_candidate = max(candidates, key=lambda c: c.reliability_score)
            
            # Check for local contradictions
            requires_correction = False
            if best_candidate.success and best_candidate.content:
                requires_correction = self.consistency_checker.check_local_contradiction(
                    best_candidate.content, state.reasoning_trace
                )
            
            # Record tool invocation in history
            tool_invocation = ToolInvocation(
                tool_name=best_candidate.tool_name,
                args=best_candidate.tool_args,
                result=best_candidate.content,
                success=best_candidate.success,
                error_message=best_candidate.error_message,
                execution_time_ms=best_candidate.execution_time_ms,
                plan_step_ref=step.step_id,
                metadata=best_candidate.metadata
            )
            state.tool_history.add_invocation(tool_invocation)
            
            # Update step status
            if best_candidate.success:
                step.update_status(StepStatus.COMPLETED)
                self.dependency_graph.add_vertex(step.step_id)
            else:
                step.update_status(StepStatus.FAILED)
            
            logger.info(f"Step execution completed, requires_correction: {requires_correction}")
            return best_candidate, requires_correction
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            step.update_status(StepStatus.FAILED)
            return None, False

    def update_dependency_graph(self, 
                              completed_step_id: str, 
                              dependency_step_ids: List[str]):
        """Update the dependency graph with new dependencies."""
        self.dependency_graph.add_vertex(completed_step_id)
        
        for dep_id in dependency_step_ids:
            if dep_id in self.dependency_graph.vertices:
                self.dependency_graph.add_edge(dep_id, completed_step_id)

    def prune_dependency_graph(self, removed_step_ids: Set[str]):
        """Prune the dependency graph when steps are removed."""
        self.dependency_graph.prune_and_reconnect(removed_step_ids)

    def _get_available_tools(self, step: PlanStep) -> List[BaseTool]:
        """Get available tools for the given step."""
        if step.tool_requirements:
            # Use specifically required tools
            tools = []
            for tool_name in step.tool_requirements:
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    tools.append(tool)
            return tools
        else:
            # Return all available tools
            return self.tool_registry.get_all_tools()

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for analysis."""
        return {
            "dependency_graph_size": len(self.dependency_graph.vertices),
            "dependency_edges": len(self.dependency_graph.edges),
            "tools_used": len(set(
                inv.tool_name for inv in self.dependency_graph.adjacency_list.keys()
            ))
        }
