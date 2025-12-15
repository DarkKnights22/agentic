"""
State Manager

Manages the .agentic/ directory, artifacts, and enforces:
- Plan Lock protocol (immutability after approval)
- Context Contract (sub-agent isolation)
- Escalation Rules (failure tracking)
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import shutil


AGENTIC_DIR = ".agentic"
PLAN_FILE = "plan.md"
STATUS_FILE = "status.md"
META_FILE = "meta.json"
ATTEMPTS_DIR = "attempts"
CONTEXT_DIR = "contexts"
LOGS_DIR = "logs"
TASKS_DIR = "tasks"


class PlanStatus(Enum):
    """Status of the planning document."""
    DRAFT = "draft"
    APPROVED = "approved"
    LOCKED = "locked"
    SUPERSEDED = "superseded"


class TaskStatus(Enum):
    """Status of an individual task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass
class TaskAttempt:
    """Record of a single task execution attempt."""
    attempt_number: int
    timestamp: str
    sub_agent_id: str
    success: bool
    error_message: Optional[str] = None
    changes_made: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class TaskState:
    """State tracking for a single task."""
    task_id: str
    title: str
    status: str = TaskStatus.PENDING.value
    attempts: list[TaskAttempt] = field(default_factory=list)
    max_attempts: int = 3
    files_allowed: list[str] = field(default_factory=list)
    
    @property
    def attempt_count(self) -> int:
        return len(self.attempts)
    
    @property
    def should_escalate(self) -> bool:
        """Check if task has failed enough times to require escalation."""
        failed_attempts = sum(1 for a in self.attempts if not a.success)
        return failed_attempts >= self.max_attempts


@dataclass
class PlanMeta:
    """Metadata for a planning session."""
    session_id: str
    created_at: str
    updated_at: str
    model: str
    plan_version: int = 1
    plan_status: str = PlanStatus.DRAFT.value
    plan_hash: Optional[str] = None  # Hash of approved plan for lock verification
    user_prompt: str = ""
    escalation_threshold: int = 3


class PlanLockError(Exception):
    """Raised when attempting to modify a locked plan."""
    pass


class ContextViolationError(Exception):
    """Raised when a sub-agent attempts to access unauthorized context."""
    pass


class StateManager:
    """
    Manages all persistent state for the agentic system.
    
    Enforces:
    - Plan Lock: Once approved, plans cannot be modified
    - Context Contract: Sub-agents only access allowed files
    - Escalation Rules: Failed tasks trigger escalation
    
    Usage:
        state = StateManager("/path/to/repo")
        state.initialize_session("user prompt here")
        state.save_plan(plan_content)
        state.lock_plan()  # After user approval
    """
    
    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path).resolve()
        self.agentic_path = self.repo_path / AGENTIC_DIR
        self.plan_path = self.agentic_path / PLAN_FILE
        self.status_path = self.agentic_path / STATUS_FILE
        self.meta_path = self.agentic_path / META_FILE
        self.attempts_path = self.agentic_path / ATTEMPTS_DIR
        self.contexts_path = self.agentic_path / CONTEXT_DIR
        self.logs_path = self.agentic_path / LOGS_DIR
        self.tasks_path = self.agentic_path / TASKS_DIR
        
        self._meta: Optional[PlanMeta] = None
        self._tasks: dict[str, TaskState] = {}
    
    def initialize_session(
        self,
        user_prompt: str,
        model: str = "anthropic/claude-sonnet-4.5",
    ) -> str:
        """
        Initialize a new planning session.
        
        Creates .agentic/ directory and initial metadata.
        Returns the session ID.
        """
        # Create directories
        self.agentic_path.mkdir(exist_ok=True)
        self.attempts_path.mkdir(exist_ok=True)
        self.contexts_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)
        self.tasks_path.mkdir(exist_ok=True)
        
        # Generate session ID
        session_id = self._generate_session_id(user_prompt)
        now = datetime.utcnow().isoformat() + "Z"
        
        self._meta = PlanMeta(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            model=model,
            user_prompt=user_prompt,
        )
        
        self._save_meta()
        
        # Create empty status file
        self.status_path.write_text("# Task Status\n\nNo tasks executed yet.\n", encoding='utf-8')
        
        return session_id
    
    def load_session(self) -> Optional[PlanMeta]:
        """Load existing session metadata."""
        if not self.meta_path.exists():
            return None
        
        try:
            data = json.loads(self.meta_path.read_text(encoding='utf-8'))
            self._meta = PlanMeta(**data)
            return self._meta
        except (json.JSONDecodeError, TypeError):
            return None
    
    def _generate_session_id(self, prompt: str) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        content = f"{timestamp}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _save_meta(self):
        """Save metadata to disk."""
        if self._meta:
            self._meta.updated_at = datetime.utcnow().isoformat() + "Z"
            self.meta_path.write_text(
                json.dumps(asdict(self._meta), indent=2),
                encoding='utf-8'
            )
    
    # -------------------------------------------------------------------------
    # Plan Lock Protocol
    # -------------------------------------------------------------------------
    
    def save_plan(self, content: str) -> None:
        """
        Save the planning document.
        
        Raises PlanLockError if plan is already locked.
        """
        if self._meta and self._meta.plan_status == PlanStatus.LOCKED.value:
            raise PlanLockError(
                "Plan is locked. Create a new plan version to make changes."
            )
        
        self.plan_path.write_text(content, encoding='utf-8')
        
        if self._meta:
            self._meta.plan_status = PlanStatus.DRAFT.value
            self._save_meta()
    
    def approve_plan(self) -> None:
        """Mark the plan as approved (but not yet locked)."""
        if not self._meta:
            raise ValueError("No active session")
        
        self._meta.plan_status = PlanStatus.APPROVED.value
        self._save_meta()
    
    def lock_plan(self) -> str:
        """
        Lock the plan, making it immutable.
        
        Returns the plan hash for verification.
        """
        if not self._meta:
            raise ValueError("No active session")
        
        if not self.plan_path.exists():
            raise ValueError("No plan exists to lock")
        
        # Calculate hash of plan content
        plan_content = self.plan_path.read_text(encoding='utf-8')
        plan_hash = hashlib.sha256(plan_content.encode('utf-8')).hexdigest()
        
        self._meta.plan_status = PlanStatus.LOCKED.value
        self._meta.plan_hash = plan_hash
        self._save_meta()
        
        return plan_hash
    
    def verify_plan_integrity(self) -> bool:
        """Verify the locked plan hasn't been tampered with."""
        if not self._meta or not self._meta.plan_hash:
            return False
        
        if not self.plan_path.exists():
            return False
        
        current_hash = hashlib.sha256(
            self.plan_path.read_text(encoding='utf-8').encode('utf-8')
        ).hexdigest()
        
        return current_hash == self._meta.plan_hash
    
    def create_new_plan_version(self) -> int:
        """
        Create a new plan version, superseding the current one.
        
        Archives the old plan and unlocks for new planning.
        Returns the new version number.
        """
        if not self._meta:
            raise ValueError("No active session")
        
        # Archive old plan if it exists
        if self.plan_path.exists():
            old_version = self._meta.plan_version
            archive_name = f"plan_v{old_version}.md"
            archive_path = self.agentic_path / archive_name
            shutil.copy(self.plan_path, archive_path)
        
        # Increment version and reset status
        self._meta.plan_version += 1
        self._meta.plan_status = PlanStatus.DRAFT.value
        self._meta.plan_hash = None
        self._save_meta()
        
        return self._meta.plan_version
    
    @property
    def is_plan_locked(self) -> bool:
        """Check if the plan is locked."""
        return (
            self._meta is not None and 
            self._meta.plan_status == PlanStatus.LOCKED.value
        )
    
    # -------------------------------------------------------------------------
    # Context Contract
    # -------------------------------------------------------------------------
    
    def register_task(
        self,
        task_id: str,
        title: str,
        allowed_files: list[str],
        max_attempts: int = 3,
    ) -> TaskState:
        """
        Register a task with its context contract.
        
        Args:
            task_id: Unique task identifier
            title: Human-readable task title
            allowed_files: Files this task's sub-agent may access
            max_attempts: Maximum attempts before escalation
        """
        task = TaskState(
            task_id=task_id,
            title=title,
            files_allowed=allowed_files,
            max_attempts=max_attempts,
        )
        self._tasks[task_id] = task
        self._save_task_state(task)
        return task
    
    def get_task_context(self, task_id: str) -> dict[str, Any]:
        """
        Get the context contract for a sub-agent.
        
        Returns only what the sub-agent is allowed to see.
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        
        # Build minimal context
        context = {
            "task_id": task.task_id,
            "task_title": task.title,
            "allowed_files": task.files_allowed,
            "attempt_number": task.attempt_count + 1,
            "previous_attempts": [
                {
                    "attempt": a.attempt_number,
                    "success": a.success,
                    "error": a.error_message,
                }
                for a in task.attempts
            ],
        }
        
        # Save context to disk for audit
        context_file = self.contexts_path / f"{task_id}_context.json"
        context_file.write_text(json.dumps(context, indent=2), encoding='utf-8')
        
        return context
    
    def validate_file_access(self, task_id: str, file_path: str) -> bool:
        """
        Validate that a sub-agent can access a file.
        
        Raises ContextViolationError if access is not allowed.
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        
        # Normalize path for comparison
        normalized = str(Path(file_path))
        
        # Check if file is in allowed list
        for allowed in task.files_allowed:
            allowed_normalized = str(Path(allowed))
            if normalized == allowed_normalized:
                return True
            # Also allow access to files in allowed directories
            if normalized.startswith(allowed_normalized + "/"):
                return True
        
        raise ContextViolationError(
            f"Task '{task_id}' is not allowed to access '{file_path}'. "
            f"Allowed files: {task.files_allowed}"
        )
    
    # -------------------------------------------------------------------------
    # Escalation Rules
    # -------------------------------------------------------------------------
    
    def record_attempt(
        self,
        task_id: str,
        sub_agent_id: str,
        success: bool,
        error_message: Optional[str] = None,
        changes_made: Optional[list[str]] = None,
        duration_seconds: float = 0.0,
    ) -> tuple[TaskAttempt, bool]:
        """
        Record a task execution attempt.
        
        Returns:
            Tuple of (attempt record, should_escalate)
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        
        attempt = TaskAttempt(
            attempt_number=task.attempt_count + 1,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sub_agent_id=sub_agent_id,
            success=success,
            error_message=error_message,
            changes_made=changes_made or [],
            duration_seconds=duration_seconds,
        )
        
        task.attempts.append(attempt)
        
        if success:
            task.status = TaskStatus.COMPLETED.value
        elif task.should_escalate:
            task.status = TaskStatus.ESCALATED.value
        else:
            task.status = TaskStatus.FAILED.value
        
        self._save_task_state(task)
        self._save_attempt(task_id, attempt)
        
        return attempt, task.should_escalate
    
    def get_escalated_tasks(self) -> list[TaskState]:
        """Get all tasks that require escalation."""
        return [t for t in self._tasks.values() if t.should_escalate]
    
    def _save_task_state(self, task: TaskState):
        """Save task state to disk."""
        task_file = self.agentic_path / f"task_{task.task_id}.json"
        task_file.write_text(json.dumps(asdict(task), indent=2), encoding='utf-8')
    
    def _save_attempt(self, task_id: str, attempt: TaskAttempt):
        """Save attempt record to disk."""
        attempt_file = (
            self.attempts_path / 
            f"{task_id}_attempt_{attempt.attempt_number}.json"
        )
        attempt_file.write_text(json.dumps(asdict(attempt), indent=2), encoding='utf-8')
    
    # -------------------------------------------------------------------------
    # Status Updates
    # -------------------------------------------------------------------------
    
    def update_status(self, content: str):
        """Update the status file."""
        self.status_path.write_text(content, encoding='utf-8')
    
    def get_status_summary(self) -> str:
        """Generate a status summary."""
        lines = ["# Task Status", ""]
        
        if not self._tasks:
            lines.append("No tasks registered yet.")
            return "\n".join(lines)
        
        for task in self._tasks.values():
            status_emoji = {
                TaskStatus.PENDING.value: "⏳",
                TaskStatus.IN_PROGRESS.value: "🔄",
                TaskStatus.COMPLETED.value: "✅",
                TaskStatus.FAILED.value: "❌",
                TaskStatus.ESCALATED.value: "🚨",
            }.get(task.status, "❓")
            
            lines.append(f"## {status_emoji} Task: {task.title}")
            lines.append(f"- Status: {task.status}")
            lines.append(f"- Attempts: {task.attempt_count}/{task.max_attempts}")
            
            if task.attempts:
                last = task.attempts[-1]
                lines.append(f"- Last attempt: {'Success' if last.success else 'Failed'}")
                if last.error_message:
                    lines.append(f"- Error: {last.error_message}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Task Loading from Plan
    # -------------------------------------------------------------------------
    
    def load_tasks_from_plan(self) -> list[TaskState]:
        """
        Load task definitions from the locked plan.
        
        Parses plan.md and registers all tasks.
        Returns list of registered TaskState objects.
        """
        from .plan_parser import load_plan, PlanParseError
        
        if not self.plan_path.exists():
            return []
        
        try:
            parsed_plan = load_plan(self.plan_path)
        except PlanParseError:
            return []
        
        registered_tasks = []
        for task in parsed_plan.tasks:
            task_state = self.register_task(
                task_id=task.id,
                title=task.title,
                allowed_files=task.files_touched,
                max_attempts=parsed_plan.escalation_threshold,
            )
            registered_tasks.append(task_state)
        
        return registered_tasks
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> list[TaskState]:
        """Get all registered tasks."""
        return list(self._tasks.values())
    
    def is_execution_complete(self) -> bool:
        """Check if all tasks are completed."""
        if not self._tasks:
            return False
        return all(
            t.status == TaskStatus.COMPLETED.value
            for t in self._tasks.values()
        )
    
    def has_escalated_tasks(self) -> bool:
        """Check if any tasks have been escalated."""
        return any(
            t.status == TaskStatus.ESCALATED.value
            for t in self._tasks.values()
        )


if __name__ == "__main__":
    import sys
    
    # Quick test
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    state = StateManager(path)
    
    print(f"Initializing session in {path}...")
    session_id = state.initialize_session("Test prompt")
    print(f"Session ID: {session_id}")
    
    print("\nSaving draft plan...")
    state.save_plan("# Test Plan\n\nThis is a test.")
    
    print("Locking plan...")
    plan_hash = state.lock_plan()
    print(f"Plan hash: {plan_hash}")
    
    print(f"Plan integrity valid: {state.verify_plan_integrity()}")

