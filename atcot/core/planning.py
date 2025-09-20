"""
Adaptive planning generation and dynamic replanning for the ATCOT framework.

This module implements the planning module that decomposes input queries into
structured plans through maximum a posteriori estimation and performs dynamic
replanning when inconsistencies are detected.
"""

import math
import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .state_representation import (
    ATCOTState, PlanStep, PlanningStructure, ReasoningStep, 
    StepStatus, CorrectionType, CorrectionEntry
)
from ..utils.llm_interface import LLMInterface
from ..utils.consistency import ConsistencyChecker

logger = logging.getLogger(__name__)


@dataclass
class PlanningContext:
    """Context information for planning operations."""
    query: str
    available_tools: List[str]
    previous_attempts: List[PlanningStructure]
    reasoning_history: List[ReasoningStep]
    constraints: Dict[str, Any]


class PlanGenerator(ABC):
    """Abstract base class for plan generation strategies."""
    
    @abstractmethod
    def generate_plan(self, context: PlanningContext) -> PlanningStructure:
        """Generate a plan for the given context."""
        pass

    @abstractmethod
    def estimate_confidence(self, step: PlanStep, context: PlanningContext) -> float:
        """Estimate confidence score for a plan step."""
        pass


class LLMPlanGenerator(PlanGenerator):
    """LLM-based plan generator using maximum a posteriori estimation."""
    
    def __init__(self, llm_interface: LLMInterface, temperature: float = 0.1):
        self.llm_interface = llm_interface
        self.temperature = temperature

    def generate_plan(self, context: PlanningContext) -> PlanningStructure:
        """
        Generate plan P_0 through maximum a posteriori estimation:
        P_0 = argmax_P P(P|q, S_0, θ)
        """
        planning_prompt = self._build_planning_prompt(context)
        
        try:
            response = self.llm_interface.generate(
                prompt=planning_prompt,
                temperature=self.temperature,
                max_tokens=1024
            )
            
            plan_structure = self._parse_plan_response(response, context)
            
            # Validate DAG structure
            if not plan_structure.is_valid_dag():
                logger.warning("Generated plan contains cycles, attempting repair")
                plan_structure = self._repair_dag_structure(plan_structure)
            
            return plan_structure
            
        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            return self._create_fallback_plan(context)

    def estimate_confidence(self, step: PlanStep, context: PlanningContext) -> float:
        """
        Compute confidence score α_i = σ(W_c · h_i + b_c).
        
        Uses LLM to estimate step confidence based on context and step description.
        """
        confidence_prompt = self._build_confidence_prompt(step, context)
        
        try:
            response = self.llm_interface.generate(
                prompt=confidence_prompt,
                temperature=0.0,
                max_tokens=50
            )
            
            # Extract confidence score from response
            confidence = self._parse_confidence_response(response)
            return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Failed to estimate confidence: {e}")
            return 0.5  # Default moderate confidence

    def _build_planning_prompt(self, context: PlanningContext) -> str:
        """Build prompt for plan generation."""
        prompt = f"""Given the following query, create a detailed step-by-step plan to solve it.

Query: {context.query}

Available tools: {', '.join(context.available_tools)}

Requirements:
1. Break down the query into logical, sequential steps
2. Specify dependencies between steps where one step requires output from another
3. Indicate which tools might be needed for each step
4. Keep steps specific and actionable

Please provide the plan in the following format:
Step [ID]: [Description]
Dependencies: [List of step IDs this depends on, or "None"]
Tools: [List of tools that might be needed]
Confidence: [Your confidence in this step from 0.0 to 1.0]

Example:
Step 1: Extract key information from the query
Dependencies: None
Tools: None
Confidence: 0.9

Step 2: Search for relevant information online
Dependencies: 1
Tools: web_search
Confidence: 0.8
"""
        
        if context.reasoning_history:
            recent_reasoning = context.reasoning_history[-3:]  # Last 3 steps
            prompt += f"\nPrevious reasoning context:\n"
            for step in recent_reasoning:
                prompt += f"- {step.content}\n"
        
        return prompt

    def _build_confidence_prompt(self, step: PlanStep, context: PlanningContext) -> str:
        """Build prompt for confidence estimation."""
        return f"""Estimate the confidence level for executing this plan step successfully.

Query: {context.query}
Step: {step.description}
Dependencies: {list(step.dependencies) if step.dependencies else "None"}
Required tools: {step.tool_requirements}

Consider:
- Clarity and specificity of the step
- Availability of required tools
- Complexity of the task
- Likelihood of success

Provide a confidence score between 0.0 (very uncertain) and 1.0 (very confident).
Respond with only the numerical score.
"""

    def _parse_plan_response(self, response: str, context: PlanningContext) -> PlanningStructure:
        """Parse LLM response into a PlanningStructure."""
        plan = PlanningStructure()
        lines = response.strip().split('\n')
        
        current_step = None
        step_mapping = {}  # Map step numbers to step IDs
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Step '):
                # Parse step definition
                try:
                    step_num_end = line.find(':')
                    if step_num_end == -1:
                        continue
                    
                    step_num_str = line[5:step_num_end].strip()
                    description = line[step_num_end + 1:].strip()
                    
                    current_step = PlanStep(description=description)
                    step_mapping[step_num_str] = current_step.step_id
                    plan.add_step(current_step)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse step: {line}, error: {e}")
                    continue
                    
            elif line.startswith('Dependencies:') and current_step:
                # Parse dependencies
                deps_str = line[13:].strip()
                if deps_str.lower() != 'none':
                    dep_nums = [d.strip() for d in deps_str.split(',')]
                    for dep_num in dep_nums:
                        if dep_num in step_mapping:
                            current_step.add_dependency(step_mapping[dep_num])
                            
            elif line.startswith('Tools:') and current_step:
                # Parse tool requirements
                tools_str = line[6:].strip()
                if tools_str.lower() != 'none':
                    tools = [t.strip() for t in tools_str.split(',')]
                    current_step.tool_requirements = tools
                    
            elif line.startswith('Confidence:') and current_step:
                # Parse confidence score
                try:
                    conf_str = line[11:].strip()
                    confidence = float(conf_str)
                    current_step.confidence_score = max(0.0, min(1.0, confidence))
                except ValueError:
                    current_step.confidence_score = 0.5
        
        return plan

    def _parse_confidence_response(self, response: str) -> float:
        """Parse confidence score from LLM response."""
        try:
            # Extract first number found in response
            import re
            numbers = re.findall(r'(\d+\.?\d*)', response.strip())
            if numbers:
                confidence = float(numbers[0])
                # If the number is greater than 1, assume it's on a scale of 1-10 or 1-100
                if confidence > 1.0:
                    if confidence <= 10:
                        confidence = confidence / 10.0
                    elif confidence <= 100:
                        confidence = confidence / 100.0
                    else:
                        confidence = 1.0
                return confidence
            return 0.5
        except Exception:
            return 0.5

    def _repair_dag_structure(self, plan: PlanningStructure) -> PlanningStructure:
        """Repair cyclic dependencies in plan structure."""
        # Simple cycle removal by breaking edges in reverse topological order
        repaired_plan = PlanningStructure()
        
        for step in plan.steps:
            # Create new step without problematic dependencies
            new_step = PlanStep(
                description=step.description,
                confidence_score=step.confidence_score,
                tool_requirements=step.tool_requirements.copy(),
                metadata=step.metadata.copy()
            )
            
            # Only add dependencies that don't create cycles
            for dep_id in step.dependencies:
                new_step.add_dependency(dep_id)
                if not repaired_plan.is_valid_dag():
                    new_step.remove_dependency(dep_id)
            
            repaired_plan.add_step(new_step)
        
        return repaired_plan

    def _create_fallback_plan(self, context: PlanningContext) -> PlanningStructure:
        """Create a simple fallback plan when generation fails."""
        plan = PlanningStructure()
        
        # Simple 3-step plan
        step1 = PlanStep(
            description="Analyze the query and identify key information needed",
            confidence_score=0.7,
            tool_requirements=[]
        )
        
        step2 = PlanStep(
            description="Gather necessary information using available tools",
            confidence_score=0.6,
            tool_requirements=context.available_tools
        )
        step2.add_dependency(step1.step_id)
        
        step3 = PlanStep(
            description="Synthesize information and formulate final answer",
            confidence_score=0.5
        )
        step3.add_dependency(step2.step_id)
        
        plan.add_step(step1)
        plan.add_step(step2)
        plan.add_step(step3)
        
        return plan


class AdaptivePlanner:
    """
    Adaptive planner implementing dynamic replanning with consistency checks.
    
    Handles both initial plan generation and adaptive replanning when conflicts
    are detected through global consistency verification.
    """
    
    def __init__(self, 
                 plan_generator: PlanGenerator,
                 consistency_checker: ConsistencyChecker,
                 max_planning_attempts: int = 3):
        self.plan_generator = plan_generator
        self.consistency_checker = consistency_checker
        self.max_planning_attempts = max_planning_attempts

    def generate_initial_plan(self, state: ATCOTState, available_tools: List[str]) -> PlanningStructure:
        """Generate initial plan P_0 for the given query."""
        context = PlanningContext(
            query=state.query,
            available_tools=available_tools,
            previous_attempts=[],
            reasoning_history=state.reasoning_trace.steps,
            constraints={}
        )
        
        for attempt in range(self.max_planning_attempts):
            try:
                plan = self.plan_generator.generate_plan(context)
                
                # Validate plan quality
                if self._validate_plan_quality(plan, context):
                    logger.info(f"Generated initial plan with {len(plan.steps)} steps")
                    return plan
                
                # Add failed attempt to context for next iteration
                context.previous_attempts.append(plan)
                
            except Exception as e:
                logger.error(f"Planning attempt {attempt + 1} failed: {e}")
        
        logger.warning("All planning attempts failed, using fallback")
        return self.plan_generator._create_fallback_plan(context)

    def perform_replanning(self, 
                          current_plan: PlanningStructure,
                          state: ATCOTState,
                          trigger_observation: Any = None) -> Tuple[PlanningStructure, CorrectionEntry]:
        """
        Perform dynamic replanning when inconsistencies are detected.
        
        Implements: P_{t+1} = f_replan(P_t, o_t, R_t) if φ(S_t) = 0
        """
        logger.info("Performing replanning due to detected inconsistencies")
        
        context = PlanningContext(
            query=state.query,
            available_tools=self._extract_available_tools(state),
            previous_attempts=[current_plan],
            reasoning_history=state.reasoning_trace.steps,
            constraints={}
        )
        
        # Analyze conflicts between observations and existing plan
        conflict_analysis = self._analyze_conflicts(current_plan, state, trigger_observation)
        
        # Generate modified plan that incorporates new information
        new_plan = self._generate_adapted_plan(context, conflict_analysis)
        
        # Create correction entry
        correction = CorrectionEntry(
            correction_type=CorrectionType.GLOBAL_INCONSISTENCY,
            trigger_description=f"Global consistency check failed, replanning triggered",
            affected_steps=set(step.step_id for step in current_plan.steps),
            revision_description=f"Generated new plan with {len(new_plan.steps)} steps",
            success=True
        )
        
        return new_plan, correction

    def adapt_plan_for_correction(self,
                                current_plan: PlanningStructure,
                                removed_reasoning_steps: Set[str],
                                state: ATCOTState) -> PlanningStructure:
        """
        Adapt plan when reasoning steps are removed due to local corrections.
        
        Rebuilds dependency closure ensuring reachability and DAG properties.
        """
        adapted_plan = PlanningStructure()
        
        # Copy all steps but update dependencies
        for step in current_plan.steps:
            adapted_step = PlanStep(
                step_id=step.step_id,
                description=step.description,
                confidence_score=step.confidence_score,
                status=step.status,
                tool_requirements=step.tool_requirements.copy(),
                metadata=step.metadata.copy()
            )
            
            # Update dependencies - remove references to removed reasoning steps
            updated_dependencies = set()
            for dep_id in step.dependencies:
                if dep_id not in removed_reasoning_steps:
                    updated_dependencies.add(dep_id)
            
            adapted_step.dependencies = updated_dependencies
            adapted_plan.add_step(adapted_step)
        
        # Ensure DAG properties are maintained
        if not adapted_plan.is_valid_dag():
            logger.warning("Plan adaptation resulted in invalid DAG, repairing")
            adapted_plan = self.plan_generator._repair_dag_structure(adapted_plan)
        
        return adapted_plan

    def _validate_plan_quality(self, plan: PlanningStructure, context: PlanningContext) -> bool:
        """Validate that the generated plan meets quality criteria."""
        if len(plan.steps) == 0:
            return False
        
        # Check for reasonable confidence scores
        avg_confidence = sum(step.confidence_score for step in plan.steps) / len(plan.steps)
        if avg_confidence < 0.3:
            logger.warning(f"Plan has low average confidence: {avg_confidence}")
            return False
        
        # Check for valid DAG structure
        if not plan.is_valid_dag():
            logger.warning("Plan contains cycles")
            return False
        
        # Check that plan addresses the query
        query_keywords = set(context.query.lower().split())
        plan_keywords = set()
        for step in plan.steps:
            plan_keywords.update(step.description.lower().split())
        
        keyword_overlap = len(query_keywords.intersection(plan_keywords)) / len(query_keywords)
        if keyword_overlap < 0.2:
            logger.warning(f"Plan has low keyword overlap with query: {keyword_overlap}")
            return False
        
        return True

    def _analyze_conflicts(self, 
                          plan: PlanningStructure, 
                          state: ATCOTState, 
                          observation: Any) -> Dict[str, Any]:
        """Analyze conflicts between observations and existing plan."""
        conflicts = {
            "inconsistent_steps": [],
            "failed_dependencies": [],
            "tool_failures": [],
            "confidence_degradation": []
        }
        
        # Check for steps that contradict new observations
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                # Check if completed step conflicts with new observation
                step_reasoning = [
                    rs for rs in state.reasoning_trace.steps
                    if rs.plan_step_ref == step.step_id
                ]
                
                for reasoning_step in step_reasoning:
                    if self.consistency_checker.check_local_contradiction(observation, reasoning_step.content):
                        conflicts["inconsistent_steps"].append(step.step_id)
                        break
        
        return conflicts

    def _generate_adapted_plan(self, 
                             context: PlanningContext, 
                             conflict_analysis: Dict[str, Any]) -> PlanningStructure:
        """Generate an adapted plan that addresses identified conflicts."""
        # For now, regenerate the entire plan with conflict context
        # In a more sophisticated implementation, we could modify only affected parts
        
        # Add conflict information to context
        context.constraints["conflicts"] = conflict_analysis
        
        return self.plan_generator.generate_plan(context)

    def _extract_available_tools(self, state: ATCOTState) -> List[str]:
        """Extract list of available tools from state."""
        available_tools = set()
        
        # Extract from tool history
        for invocation in state.tool_history.invocations:
            available_tools.add(invocation.tool_name)
        
        # Extract from plan steps
        for step in state.planning_structure.steps:
            available_tools.update(step.tool_requirements)
        
        return list(available_tools)
