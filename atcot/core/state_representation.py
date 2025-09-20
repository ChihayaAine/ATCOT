"""
State representation for the ATCOT framework.

This module implements the comprehensive state representation S = {P, R, H, C} where:
- P: Planning structure with ordered sequence of steps and dependencies
- R: Reasoning trace with intermediate conclusions and justifications  
- H: Tool invocation history with temporal annotations and results
- C: Correction log tracking all revisions and triggering conditions
"""

from typing import List, Dict, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CorrectionType(Enum):
    """Type of correction performed."""
    LOCAL_CONTRADICTION = "local_contradiction"
    GLOBAL_INCONSISTENCY = "global_inconsistency"
    TOOL_FAILURE = "tool_failure"
    DEPENDENCY_VIOLATION = "dependency_violation"


@dataclass
class PlanStep:
    """Individual step in the planning structure."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    dependencies: Set[str] = field(default_factory=set)
    confidence_score: float = 0.0
    status: StepStatus = StepStatus.PENDING
    tool_requirements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_dependency(self, step_id: str) -> None:
        """Add a dependency to this step."""
        self.dependencies.add(step_id)
        self.updated_at = datetime.now()

    def remove_dependency(self, step_id: str) -> None:
        """Remove a dependency from this step."""
        self.dependencies.discard(step_id)
        self.updated_at = datetime.now()

    def update_status(self, status: StepStatus) -> None:
        """Update the status of this step."""
        self.status = status
        self.updated_at = datetime.now()


@dataclass
class PlanningStructure:
    """
    Planning structure P denoting ordered sequence of steps with explicit dependencies.
    
    Maintains a DAG structure where each step has dependencies on previous steps.
    """
    steps: List[PlanStep] = field(default_factory=list)
    step_index: Dict[str, int] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_step(self, step: PlanStep) -> None:
        """Add a new step to the planning structure."""
        if step.step_id not in self.step_index:
            self.steps.append(step)
            self.step_index[step.step_id] = len(self.steps) - 1
            self.dependency_graph[step.step_id] = step.dependencies.copy()
            self.updated_at = datetime.now()

    def remove_step(self, step_id: str) -> bool:
        """Remove a step and update dependencies."""
        if step_id not in self.step_index:
            return False
        
        index = self.step_index[step_id]
        self.steps.pop(index)
        
        # Update indices
        self.step_index = {
            sid: idx if idx < index else idx - 1 
            for sid, idx in self.step_index.items() 
            if sid != step_id
        }
        
        # Remove from dependency graph and update other steps
        self.dependency_graph.pop(step_id, None)
        for other_step in self.steps:
            other_step.remove_dependency(step_id)
            self.dependency_graph[other_step.step_id] = other_step.dependencies.copy()
        
        self.updated_at = datetime.now()
        return True

    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute (all dependencies completed)."""
        ready_steps = []
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                dependencies_met = all(
                    self.get_step(dep_id).status == StepStatus.COMPLETED
                    for dep_id in step.dependencies
                    if self.get_step(dep_id) is not None
                )
                if dependencies_met:
                    ready_steps.append(step)
        return ready_steps

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by its ID."""
        if step_id in self.step_index:
            return self.steps[self.step_index[step_id]]
        return None

    def is_valid_dag(self) -> bool:
        """Check if the dependency structure forms a valid DAG."""
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False

        for step_id in self.dependency_graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return False
        return True


@dataclass
class ReasoningStep:
    """Individual step in the reasoning trace."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    justification: str = ""
    confidence_score: float = 0.0
    plan_step_ref: Optional[str] = None
    tool_observations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ReasoningTrace:
    """
    Reasoning trace R comprising intermediate conclusions and their justifications.
    
    Maintains sequential ordering of reasoning steps with references to plan steps.
    """
    steps: List[ReasoningStep] = field(default_factory=list)
    step_index: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def append_step(self, step: ReasoningStep) -> None:
        """Append a new reasoning step."""
        self.steps.append(step)
        self.step_index[step.step_id] = len(self.steps) - 1
        self.updated_at = datetime.now()

    def remove_steps(self, step_ids: Set[str]) -> List[ReasoningStep]:
        """Remove specified reasoning steps and return them."""
        removed_steps = []
        indices_to_remove = []
        
        for step_id in step_ids:
            if step_id in self.step_index:
                index = self.step_index[step_id]
                indices_to_remove.append(index)
                removed_steps.append(self.steps[index])
        
        # Remove in reverse order to maintain indices
        for index in sorted(indices_to_remove, reverse=True):
            self.steps.pop(index)
        
        # Rebuild index
        self.step_index = {
            step.step_id: idx 
            for idx, step in enumerate(self.steps)
        }
        
        self.updated_at = datetime.now()
        return removed_steps

    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """Get a reasoning step by ID."""
        if step_id in self.step_index:
            return self.steps[self.step_index[step_id]]
        return None

    def get_latest_steps(self, n: int = 5) -> List[ReasoningStep]:
        """Get the latest n reasoning steps."""
        return self.steps[-n:] if len(self.steps) >= n else self.steps


@dataclass
class ToolInvocation:
    """Record of a single tool invocation."""
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = False
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    plan_step_ref: Optional[str] = None
    reasoning_step_ref: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolHistory:
    """
    Tool invocation history H with temporal annotations and results.
    
    Maintains complete record of all tool invocations for dependency tracking.
    """
    invocations: List[ToolInvocation] = field(default_factory=list)
    invocation_index: Dict[str, int] = field(default_factory=dict)
    tool_usage_stats: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_invocation(self, invocation: ToolInvocation) -> None:
        """Add a new tool invocation record."""
        self.invocations.append(invocation)
        self.invocation_index[invocation.invocation_id] = len(self.invocations) - 1
        
        # Update usage stats
        tool_name = invocation.tool_name
        self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + 1
        
        self.updated_at = datetime.now()

    def get_invocations_for_step(self, step_id: str) -> List[ToolInvocation]:
        """Get all tool invocations for a specific plan or reasoning step."""
        return [
            inv for inv in self.invocations
            if inv.plan_step_ref == step_id or inv.reasoning_step_ref == step_id
        ]

    def get_successful_invocations(self) -> List[ToolInvocation]:
        """Get all successful tool invocations."""
        return [inv for inv in self.invocations if inv.success]

    def get_failed_invocations(self) -> List[ToolInvocation]:
        """Get all failed tool invocations."""
        return [inv for inv in self.invocations if not inv.success]


@dataclass
class CorrectionEntry:
    """Single correction entry in the correction log."""
    correction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correction_type: CorrectionType = CorrectionType.LOCAL_CONTRADICTION
    trigger_description: str = ""
    affected_steps: Set[str] = field(default_factory=set)
    revision_description: str = ""
    success: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionLog:
    """
    Correction log C tracking all revisions and their triggering conditions.
    
    Maintains history of all corrections for analysis and convergence tracking.
    """
    entries: List[CorrectionEntry] = field(default_factory=list)
    entry_index: Dict[str, int] = field(default_factory=dict)
    correction_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_correction(self, entry: CorrectionEntry) -> None:
        """Add a new correction entry."""
        self.entries.append(entry)
        self.entry_index[entry.correction_id] = len(self.entries) - 1
        self.correction_count += 1
        self.updated_at = datetime.now()

    def get_recent_corrections(self, n: int = 5) -> List[CorrectionEntry]:
        """Get the most recent n corrections."""
        return self.entries[-n:] if len(self.entries) >= n else self.entries

    def get_corrections_by_type(self, correction_type: CorrectionType) -> List[CorrectionEntry]:
        """Get all corrections of a specific type."""
        return [entry for entry in self.entries if entry.correction_type == correction_type]

    def get_successful_corrections(self) -> List[CorrectionEntry]:
        """Get all successful corrections."""
        return [entry for entry in self.entries if entry.success]


@dataclass
class ATCOTState:
    """
    Comprehensive state representation S = {P, R, H, C} for the ATCOT framework.
    
    This class implements the core state representation that enables bidirectional
    state transitions and adaptive reasoning with tool feedback.
    """
    planning_structure: PlanningStructure = field(default_factory=PlanningStructure)
    reasoning_trace: ReasoningTrace = field(default_factory=ReasoningTrace)
    tool_history: ToolHistory = field(default_factory=ToolHistory)
    correction_log: CorrectionLog = field(default_factory=CorrectionLog)
    query: str = ""
    final_answer: Optional[str] = None
    converged: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update_state(self, 
                    observation: Any = None,
                    reasoning_step: Optional[ReasoningStep] = None,
                    tool_invocation: Optional[ToolInvocation] = None,
                    correction: Optional[CorrectionEntry] = None) -> None:
        """Update the state with new information."""
        if reasoning_step:
            self.reasoning_trace.append_step(reasoning_step)
        
        if tool_invocation:
            self.tool_history.add_invocation(tool_invocation)
        
        if correction:
            self.correction_log.add_correction(correction)
        
        self.updated_at = datetime.now()

    def get_current_context(self) -> Dict[str, Any]:
        """Get current context for LLM processing."""
        return {
            "query": self.query,
            "planning_structure": self.planning_structure,
            "reasoning_trace": self.reasoning_trace,
            "tool_history": self.tool_history,
            "correction_log": self.correction_log,
            "converged": self.converged,
            "correction_count": self.correction_log.correction_count
        }

    def clone(self) -> 'ATCOTState':
        """Create a deep copy of the current state."""
        import copy
        return copy.deepcopy(self)

    def apply_delta(self, delta: Dict[str, Any]) -> 'ATCOTState':
        """
        Apply a state delta using the composition operator ⊕.
        
        Implements: S ⊕ δ = {append(P, δ_P), append(R, δ_R), H ∪ δ_H, C ∪ δ_C}
        """
        new_state = self.clone()
        
        # Apply planning structure changes
        if "planning_structure" in delta:
            for step in delta["planning_structure"].get("new_steps", []):
                new_state.planning_structure.add_step(step)
        
        # Apply reasoning trace changes  
        if "reasoning_trace" in delta:
            for step in delta["reasoning_trace"].get("new_steps", []):
                new_state.reasoning_trace.append_step(step)
        
        # Apply tool history changes (union operation)
        if "tool_history" in delta:
            for invocation in delta["tool_history"].get("new_invocations", []):
                new_state.tool_history.add_invocation(invocation)
        
        # Apply correction log changes (union operation)
        if "correction_log" in delta:
            for correction in delta["correction_log"].get("new_corrections", []):
                new_state.correction_log.add_correction(correction)
        
        new_state.updated_at = datetime.now()
        return new_state
