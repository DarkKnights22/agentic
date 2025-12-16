"""
Phase orchestration for Documentation Agent.

Manages phased exploration with strict budget enforcement.
Each phase has hard limits on file reads, grep calls, and memory queries.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from enum import Enum
import json


class Phase(str, Enum):
    NOT_STARTED = "not_started"
    ARCHITECTURE_DISCOVERY = "architecture_discovery"
    COMPONENT_DEEP_DIVE = "component_deep_dive"
    CROSS_CUTTING = "cross_cutting"
    DOCUMENTATION_GENERATION = "documentation_generation"
    COMPLETED = "completed"


class BudgetExhaustedError(Exception):
    """Raised when exploration budget is exhausted."""
    pass


# =============================================================================
# Budget Configuration
# =============================================================================

@dataclass
class PhaseBudget:
    """Budget limits for a single phase."""
    max_files_read: int
    max_grep_calls: int
    max_symbols_calls: int = 0
    max_memory_queries: int = 0
    max_cross_component_queries: int = 0


# Default budgets from the plan
PHASE_BUDGETS = {
    Phase.ARCHITECTURE_DISCOVERY: PhaseBudget(
        max_files_read=10,
        max_grep_calls=15,
        max_symbols_calls=5,
    ),
    Phase.COMPONENT_DEEP_DIVE: PhaseBudget(
        max_files_read=20,  # Per component
        max_grep_calls=10,  # Per component
        max_symbols_calls=10,
    ),
    Phase.CROSS_CUTTING: PhaseBudget(
        max_files_read=15,
        max_grep_calls=10,
        max_cross_component_queries=5,
    ),
    Phase.DOCUMENTATION_GENERATION: PhaseBudget(
        max_files_read=0,  # NO file reads allowed
        max_grep_calls=0,
        max_memory_queries=20,
    ),
}


# =============================================================================
# Budget Tracking
# =============================================================================

@dataclass
class BudgetUsage:
    """Current usage against budget."""
    files_read: int = 0
    grep_calls: int = 0
    symbols_calls: int = 0
    memory_queries: int = 0
    cross_component_queries: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhaseState:
    """State for a single phase."""
    phase: str
    status: str = "pending"  # pending, in_progress, completed, budget_exhausted
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    budget_usage: BudgetUsage = field(default_factory=BudgetUsage)
    current_component: Optional[str] = None  # For Phase B
    findings_summary: str = ""
    error_message: Optional[str] = None


@dataclass
class OrchestratorState:
    """Full orchestrator state for persistence."""
    session_id: str
    current_phase: str = Phase.NOT_STARTED.value
    phases: dict = field(default_factory=dict)  # phase -> PhaseState as dict
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# =============================================================================
# Phase Orchestrator
# =============================================================================

class PhaseOrchestrator:
    """
    Orchestrates phased exploration with strict budget enforcement.

    Key responsibilities:
    - Track current phase and progress
    - Enforce exploration budgets (file reads, greps, queries)
    - Force early summarization when budget exhausted
    - Persist phase state for resume capability
    """

    def __init__(self, repo_path: Path, session_id: str):
        self.repo_path = Path(repo_path)
        self.session_id = session_id
        self.state_path = self.repo_path / ".agentic" / "doc_sessions" / f"{session_id}.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self._state = self._load_state()
        self._current_budget: Optional[PhaseBudget] = None
        self._current_usage: Optional[BudgetUsage] = None

        # Restore budget tracking if in a phase
        if self._state.current_phase != Phase.NOT_STARTED.value:
            self._restore_budget_tracking()

    def _load_state(self) -> OrchestratorState:
        """Load orchestrator state from disk."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding='utf-8'))
                return OrchestratorState(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        return OrchestratorState(session_id=self.session_id)

    def _save_state(self):
        """Persist orchestrator state to disk."""
        self._state.updated_at = datetime.utcnow().isoformat() + "Z"

        # Update current phase state with usage
        if self._current_usage and self._state.current_phase in self._state.phases:
            self._state.phases[self._state.current_phase]["budget_usage"] = self._current_usage.to_dict()

        self.state_path.write_text(
            json.dumps(asdict(self._state), indent=2),
            encoding='utf-8'
        )

    def _restore_budget_tracking(self):
        """Restore budget tracking from persisted state."""
        try:
            phase = Phase(self._state.current_phase)
            self._current_budget = PHASE_BUDGETS.get(phase)
            if self._state.current_phase in self._state.phases:
                usage_data = self._state.phases[self._state.current_phase].get("budget_usage", {})
                self._current_usage = BudgetUsage(**usage_data)
            else:
                self._current_usage = BudgetUsage()
        except (ValueError, TypeError):
            self._current_budget = None
            self._current_usage = None

    # -------------------------------------------------------------------------
    # Phase Lifecycle
    # -------------------------------------------------------------------------

    def get_current_phase(self) -> Phase:
        """Get the current phase."""
        try:
            return Phase(self._state.current_phase)
        except ValueError:
            return Phase.NOT_STARTED

    def start_phase(self, phase: Phase) -> dict:
        """
        Start a new phase.

        Returns context for the phase including budget limits.
        """
        # Create phase state
        phase_state = PhaseState(
            phase=phase.value,
            status="in_progress",
            started_at=datetime.utcnow().isoformat() + "Z",
        )
        self._state.phases[phase.value] = asdict(phase_state)
        self._state.current_phase = phase.value

        # Initialize budget tracking
        self._current_budget = PHASE_BUDGETS.get(phase)
        self._current_usage = BudgetUsage()

        self._save_state()

        return self.get_phase_context()

    def complete_phase(self, findings_summary: str = "") -> Phase:
        """
        Mark current phase as completed.

        Returns the next phase to start.
        """
        current_phase = self.get_current_phase()

        if current_phase.value in self._state.phases:
            self._state.phases[current_phase.value]["status"] = "completed"
            self._state.phases[current_phase.value]["completed_at"] = datetime.utcnow().isoformat() + "Z"
            self._state.phases[current_phase.value]["findings_summary"] = findings_summary

        # Determine next phase
        next_phase = self._get_next_phase(current_phase)
        self._state.current_phase = next_phase.value
        self._save_state()

        return next_phase

    def _get_next_phase(self, current: Phase) -> Phase:
        """Get the next phase in sequence."""
        sequence = [
            Phase.NOT_STARTED,
            Phase.ARCHITECTURE_DISCOVERY,
            Phase.COMPONENT_DEEP_DIVE,
            Phase.CROSS_CUTTING,
            Phase.DOCUMENTATION_GENERATION,
            Phase.COMPLETED,
        ]
        try:
            idx = sequence.index(current)
            if idx < len(sequence) - 1:
                return sequence[idx + 1]
        except ValueError:
            pass
        return Phase.COMPLETED

    def get_phase_context(self) -> dict:
        """
        Get context for the current phase.

        Includes phase info, budget limits, and current usage.
        """
        current_phase = self.get_current_phase()
        budget = PHASE_BUDGETS.get(current_phase)

        context = {
            "phase": current_phase.value,
            "session_id": self.session_id,
        }

        if budget and self._current_usage:
            context["budget"] = {
                "max_files_read": budget.max_files_read,
                "max_grep_calls": budget.max_grep_calls,
                "max_symbols_calls": budget.max_symbols_calls,
                "max_memory_queries": budget.max_memory_queries,
            }
            context["usage"] = self._current_usage.to_dict()
            context["remaining"] = {
                "files_read": budget.max_files_read - self._current_usage.files_read,
                "grep_calls": budget.max_grep_calls - self._current_usage.grep_calls,
                "symbols_calls": budget.max_symbols_calls - self._current_usage.symbols_calls,
                "memory_queries": budget.max_memory_queries - self._current_usage.memory_queries,
            }

        # Add component info for Phase B
        if current_phase == Phase.COMPONENT_DEEP_DIVE:
            current_comp = self._state.phases.get(current_phase.value, {}).get("current_component")
            context["current_component"] = current_comp

        return context

    # -------------------------------------------------------------------------
    # Budget Enforcement
    # -------------------------------------------------------------------------

    def check_budget(self, operation: str) -> tuple[bool, str]:
        """
        Check if an operation is allowed within budget.

        Returns (allowed, message).
        If not allowed, message explains why and suggests summarization.
        """
        if not self._current_budget or not self._current_usage:
            return True, "No budget constraints active"

        budget = self._current_budget
        usage = self._current_usage

        if operation == "file_read":
            if usage.files_read >= budget.max_files_read:
                return False, f"BUDGET_EXHAUSTED: File read limit ({budget.max_files_read}) reached. Summarize findings and store to memory."
        elif operation == "grep":
            if usage.grep_calls >= budget.max_grep_calls:
                return False, f"BUDGET_EXHAUSTED: Grep limit ({budget.max_grep_calls}) reached. Summarize findings and store to memory."
        elif operation == "symbols":
            if usage.symbols_calls >= budget.max_symbols_calls:
                return False, f"BUDGET_EXHAUSTED: Symbols limit ({budget.max_symbols_calls}) reached. Summarize findings and store to memory."
        elif operation == "memory_query":
            if budget.max_memory_queries > 0 and usage.memory_queries >= budget.max_memory_queries:
                return False, f"BUDGET_EXHAUSTED: Memory query limit ({budget.max_memory_queries}) reached."

        return True, "OK"

    def record_operation(self, operation: str) -> None:
        """Record that an operation was performed."""
        if not self._current_usage:
            return

        if operation == "file_read":
            self._current_usage.files_read += 1
        elif operation == "grep":
            self._current_usage.grep_calls += 1
        elif operation == "symbols":
            self._current_usage.symbols_calls += 1
        elif operation == "memory_query":
            self._current_usage.memory_queries += 1
        elif operation == "cross_component_query":
            self._current_usage.cross_component_queries += 1

        self._save_state()

    def get_budget_status(self) -> dict:
        """Get current budget status for display."""
        if not self._current_budget or not self._current_usage:
            return {"status": "no_budget"}

        budget = self._current_budget
        usage = self._current_usage

        return {
            "files_read": f"{usage.files_read}/{budget.max_files_read}",
            "grep_calls": f"{usage.grep_calls}/{budget.max_grep_calls}",
            "symbols_calls": f"{usage.symbols_calls}/{budget.max_symbols_calls}",
            "memory_queries": f"{usage.memory_queries}/{budget.max_memory_queries}" if budget.max_memory_queries else "N/A",
        }

    # -------------------------------------------------------------------------
    # Component Tracking (Phase B)
    # -------------------------------------------------------------------------

    def start_component(self, component_id: str):
        """Start exploring a specific component (Phase B)."""
        current_phase = self.get_current_phase()
        if current_phase != Phase.COMPONENT_DEEP_DIVE:
            return

        # Reset budget for new component
        self._current_usage = BudgetUsage()

        if current_phase.value in self._state.phases:
            self._state.phases[current_phase.value]["current_component"] = component_id
            self._state.phases[current_phase.value]["budget_usage"] = self._current_usage.to_dict()

        self._save_state()

    def complete_component(self, component_id: str):
        """Mark a component as fully explored."""
        current_phase = self.get_current_phase()
        if current_phase.value in self._state.phases:
            self._state.phases[current_phase.value]["current_component"] = None
        self._save_state()

    # -------------------------------------------------------------------------
    # State Queries
    # -------------------------------------------------------------------------

    def is_phase_completed(self, phase: Phase) -> bool:
        """Check if a phase has been completed."""
        phase_data = self._state.phases.get(phase.value, {})
        return phase_data.get("status") == "completed"

    def get_phase_findings(self, phase: Phase) -> str:
        """Get the findings summary for a completed phase."""
        phase_data = self._state.phases.get(phase.value, {})
        return phase_data.get("findings_summary", "")

    def get_documentation_status(self) -> dict:
        """Get overall documentation progress status."""
        phases_status = {}
        for phase in Phase:
            if phase in (Phase.NOT_STARTED, Phase.COMPLETED):
                continue
            phase_data = self._state.phases.get(phase.value, {})
            phases_status[phase.value] = phase_data.get("status", "pending")

        return {
            "current_phase": self._state.current_phase,
            "phases": phases_status,
            "session_id": self.session_id,
            "started_at": self._state.created_at,
            "updated_at": self._state.updated_at,
        }

    def can_resume(self) -> bool:
        """Check if there's a session that can be resumed."""
        return (
            self._state.current_phase != Phase.NOT_STARTED.value
            and self._state.current_phase != Phase.COMPLETED.value
        )


# =============================================================================
# Context Builders for Each Phase
# =============================================================================

def build_architecture_context(memory_manager, orchestrator: PhaseOrchestrator) -> str:
    """
    Build context string for Architecture Discovery phase.

    Minimal context - agent should discover through exploration.
    """
    from .doc_memory import MemoryQuery

    phase_ctx = orchestrator.get_phase_context()

    context_parts = [
        "## Phase: Architecture Discovery",
        "",
        f"Budget: {phase_ctx.get('remaining', {}).get('files_read', '?')} file reads, "
        f"{phase_ctx.get('remaining', {}).get('grep_calls', '?')} grep calls remaining",
        "",
        "Goal: Discover high-level system architecture without deep details.",
        "Find: Entry points, major components, tech stack, architectural style.",
        "",
        "Remember: Store findings to memory before budget exhaustion.",
    ]

    return "\n".join(context_parts)


def build_component_context(
    memory_manager,
    orchestrator: PhaseOrchestrator,
    component_id: str
) -> str:
    """
    Build context string for Component Deep Dive phase.

    Loads only the specific component's basic info from memory.
    """
    from .doc_memory import MemoryQuery

    phase_ctx = orchestrator.get_phase_context()

    # Get component info (bounded query)
    component_info = memory_manager.query_memory(MemoryQuery(
        query_type="component",
        filter_by={"component_id": component_id},
        max_entries=1,
        required_fields=["component_name", "root_path", "responsibility"],
        min_confidence=0.0,
    ))

    context_parts = [
        "## Phase: Component Deep Dive",
        f"## Component: {component_id}",
        "",
        f"Budget: {phase_ctx.get('remaining', {}).get('files_read', '?')} file reads, "
        f"{phase_ctx.get('remaining', {}).get('grep_calls', '?')} grep calls remaining",
        "",
    ]

    if component_info:
        info = component_info[0]
        context_parts.extend([
            f"Name: {info.get('component_name', 'Unknown')}",
            f"Path: {info.get('root_path', 'Unknown')}",
            f"Responsibility: {info.get('responsibility', 'To be discovered')}",
        ])

    context_parts.extend([
        "",
        "Goal: Fully understand this component in isolation.",
        "Find: All files, interfaces, dependencies, data models, flows.",
        "",
        "Remember: Store ComponentSummary before moving to next component.",
    ])

    return "\n".join(context_parts)


def build_cross_cutting_context(memory_manager, orchestrator: PhaseOrchestrator) -> str:
    """
    Build context string for Cross-Cutting Concerns phase.

    Loads component names from memory for reference.
    """
    from .doc_memory import MemoryQuery

    phase_ctx = orchestrator.get_phase_context()

    # Get component names (bounded query)
    components = memory_manager.query_memory(MemoryQuery(
        query_type="component",
        max_entries=20,
        required_fields=["component_id", "component_name"],
        min_confidence=0.5,
    ))

    context_parts = [
        "## Phase: Cross-Cutting Concerns",
        "",
        f"Budget: {phase_ctx.get('remaining', {}).get('files_read', '?')} file reads remaining",
        "",
        "Components explored:",
    ]

    for comp in components:
        context_parts.append(f"  - {comp.get('component_name', comp.get('component_id', 'Unknown'))}")

    context_parts.extend([
        "",
        "Goal: Document concerns spanning multiple components.",
        "Look for: Authentication, Configuration, Logging, Error handling, Data access.",
        "",
        "Remember: Work from memory first, only read files to verify.",
    ])

    return "\n".join(context_parts)


def build_docgen_context(memory_manager, orchestrator: PhaseOrchestrator) -> str:
    """
    Build context string for Documentation Generation phase.

    NO file reads allowed - memory only.
    """
    from .doc_memory import MemoryQuery

    phase_ctx = orchestrator.get_phase_context()

    # Get architecture overview
    arch = memory_manager.query_memory(MemoryQuery(
        query_type="architecture",
        max_entries=1,
        required_fields=["system_name", "purpose", "major_components", "architectural_style"],
        min_confidence=0.0,
    ))

    context_parts = [
        "## Phase: Documentation Generation",
        "",
        f"Memory queries remaining: {phase_ctx.get('remaining', {}).get('memory_queries', '?')}",
        "",
        "IMPORTANT: NO file reads allowed in this phase. Generate docs from memory only.",
        "",
    ]

    if arch:
        info = arch[0]
        context_parts.extend([
            f"System: {info.get('system_name', 'Unknown')}",
            f"Purpose: {info.get('purpose', 'Unknown')}",
            f"Style: {info.get('architectural_style', 'Unknown')}",
            f"Components: {', '.join(info.get('major_components', []))}",
        ])

    context_parts.extend([
        "",
        "Goal: Generate comprehensive documentation from memory.",
        "Output: ARCHITECTURE.md, component docs, DATA_MODELS.md, Mermaid diagrams.",
    ])

    return "\n".join(context_parts)
