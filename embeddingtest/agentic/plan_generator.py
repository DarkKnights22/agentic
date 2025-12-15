"""
Plan Generator

Generates structured plan.md documents following the required schema.
Includes Plan Lock, Context Contract, and Escalation Rules sections.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Task:
    """A single task in the plan."""
    id: str
    title: str
    description: str
    files_touched: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"  # low, medium, high


@dataclass
class TestStrategy:
    """Test strategy for the plan."""
    existing_tests: list[str] = field(default_factory=list)
    new_tests_required: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)


@dataclass
class PlanDocument:
    """
    Complete planning document structure.
    
    Follows the schema from phase-1.md with additions for:
    - Plan Lock protocol
    - Context Contract
    - Escalation Rules
    """
    # Core sections
    project_goal: str
    non_goals: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    
    # Tasks
    tasks: list[Task] = field(default_factory=list)
    
    # Testing
    test_strategy: Optional[TestStrategy] = None
    
    # Success criteria
    success_criteria: list[str] = field(default_factory=list)
    
    # Governance (new sections per user request)
    escalation_threshold: int = 3  # Failed attempts before escalation
    
    # Metadata
    version: int = 1
    created_at: Optional[str] = None
    plan_hash: Optional[str] = None


class PlanGenerator:
    """
    Generates plan.md content from structured data.
    
    Usage:
        doc = PlanDocument(
            project_goal="Build a feature X",
            tasks=[Task(id="1", title="Create API", description="...")],
        )
        generator = PlanGenerator(doc)
        markdown = generator.generate()
    """
    
    def __init__(self, document: PlanDocument):
        self.doc = document
    
    def generate(self) -> str:
        """Generate the complete plan.md content."""
        sections = [
            self._header(),
            self._project_goal(),
            self._non_goals(),
            self._assumptions(),
            self._constraints(),
            self._task_breakdown(),
            self._context_contracts(),
            self._test_strategy(),
            self._success_criteria(),
            self._plan_lock(),
            self._escalation_rules(),
            self._footer(),
        ]
        
        return "\n".join(filter(None, sections))
    
    def _header(self) -> str:
        """Generate document header."""
        timestamp = self.doc.created_at or datetime.utcnow().isoformat() + "Z"
        return f"""# Agentic Plan Document

> **Version:** {self.doc.version}  
> **Created:** {timestamp}  
> **Status:** Draft (pending approval)

---
"""
    
    def _project_goal(self) -> str:
        """Generate project goal section."""
        return f"""# Project Goal

{self.doc.project_goal}
"""
    
    def _non_goals(self) -> str:
        """Generate non-goals section."""
        if not self.doc.non_goals:
            return """# Non-Goals

*No explicit non-goals defined.*
"""
        
        items = "\n".join(f"- {ng}" for ng in self.doc.non_goals)
        return f"""# Non-Goals

{items}
"""
    
    def _assumptions(self) -> str:
        """Generate assumptions section."""
        if not self.doc.assumptions:
            return """# Assumptions

*No assumptions documented.*
"""
        
        items = "\n".join(f"- {a}" for a in self.doc.assumptions)
        return f"""# Assumptions

{items}
"""
    
    def _constraints(self) -> str:
        """Generate constraints section."""
        if not self.doc.constraints:
            return """# Constraints

*No constraints documented.*
"""
        
        items = "\n".join(f"- {c}" for c in self.doc.constraints)
        return f"""# Constraints

{items}
"""
    
    def _task_breakdown(self) -> str:
        """Generate task breakdown section."""
        if not self.doc.tasks:
            return """# Task Breakdown

*No tasks defined yet.*
"""
        
        lines = ["# Task Breakdown", ""]
        
        for i, task in enumerate(self.doc.tasks, 1):
            lines.append(f"## Task {i}: {task.title}")
            lines.append(f"**ID:** `{task.id}`")
            lines.append(f"**Complexity:** {task.estimated_complexity}")
            lines.append("")
            lines.append("### Description")
            lines.append(task.description)
            lines.append("")
            
            if task.files_touched:
                lines.append("### Files Touched")
                for f in task.files_touched:
                    lines.append(f"- `{f}`")
                lines.append("")
            
            if task.dependencies:
                lines.append("### Dependencies")
                for d in task.dependencies:
                    lines.append(f"- Task `{d}`")
                lines.append("")
            
            if task.risks:
                lines.append("### Risks / Unknowns")
                for r in task.risks:
                    lines.append(f"- {r}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _context_contracts(self) -> str:
        """Generate context contract section for each task."""
        if not self.doc.tasks:
            return ""
        
        lines = [
            "# Context Contracts",
            "",
            "> Sub-agents may only see:",
            "> - Their specific task definition",
            "> - Files explicitly listed for that task", 
            "> - Relevant prior attempt summaries",
            "",
            "| Task ID | Allowed Files |",
            "|---------|---------------|",
        ]
        
        for task in self.doc.tasks:
            files = ", ".join(f"`{f}`" for f in task.files_touched) or "*none*"
            lines.append(f"| `{task.id}` | {files} |")
        
        lines.append("")
        return "\n".join(lines)
    
    def _test_strategy(self) -> str:
        """Generate test strategy section."""
        if not self.doc.test_strategy:
            return """# Test Strategy

*Test strategy not yet defined.*
"""
        
        ts = self.doc.test_strategy
        lines = ["# Test Strategy", ""]
        
        if ts.existing_tests:
            lines.append("## Existing Tests to Run")
            for t in ts.existing_tests:
                lines.append(f"- `{t}`")
            lines.append("")
        
        if ts.new_tests_required:
            lines.append("## New Tests Required")
            for t in ts.new_tests_required:
                lines.append(f"- {t}")
            lines.append("")
        
        if ts.test_commands:
            lines.append("## Test Commands")
            lines.append("```bash")
            for cmd in ts.test_commands:
                lines.append(cmd)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _success_criteria(self) -> str:
        """Generate success criteria section."""
        if not self.doc.success_criteria:
            return """# Success Criteria

*Success criteria not yet defined.*
"""
        
        items = "\n".join(f"- [ ] {c}" for c in self.doc.success_criteria)
        return f"""# Success Criteria

{items}
"""
    
    def _plan_lock(self) -> str:
        """Generate plan lock protocol section."""
        return """# Plan Lock

> **Once this plan is approved:**
> - Tasks may NOT be added or reordered
> - Scope changes require a new plan version
> - Sub-agents must reference this plan verbatim

This prevents silent scope drift and ensures reproducibility.
"""
    
    def _escalation_rules(self) -> str:
        """Generate escalation rules section."""
        threshold = self.doc.escalation_threshold
        return f"""# Escalation Rules

If a task fails **{threshold} times**:
1. Execution halts for that task
2. Task is escalated to Master Agent
3. Master Agent will either:
   - Re-plan the task with additional context
   - Ask the user for clarification
   - Mark the task as blocked

Sub-agents do NOT retry indefinitely.
"""
    
    def _footer(self) -> str:
        """Generate document footer."""
        return """---

*This document was generated by the Agentic CLI Master Agent.*
*Do not edit manually after approval.*
"""


def create_plan(
    goal: str,
    tasks: list[dict],
    non_goals: Optional[list[str]] = None,
    assumptions: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    success_criteria: Optional[list[str]] = None,
    test_strategy: Optional[dict] = None,
    escalation_threshold: int = 3,
) -> str:
    """
    Convenience function to create a plan document.
    
    Args:
        goal: Project goal description
        tasks: List of task dicts with keys: id, title, description, files_touched, risks
        non_goals: List of non-goal statements
        assumptions: List of assumptions
        constraints: List of constraints
        success_criteria: List of success criteria
        test_strategy: Dict with existing_tests, new_tests_required, test_commands
        escalation_threshold: Number of failures before escalation
        
    Returns:
        Markdown content for plan.md
    """
    # Convert task dicts to Task objects
    task_objects = [
        Task(
            id=t.get("id", str(i)),
            title=t.get("title", f"Task {i}"),
            description=t.get("description", ""),
            files_touched=t.get("files_touched", []),
            risks=t.get("risks", []),
            dependencies=t.get("dependencies", []),
            estimated_complexity=t.get("complexity", "medium"),
        )
        for i, t in enumerate(tasks, 1)
    ]
    
    # Convert test strategy dict
    ts = None
    if test_strategy:
        ts = TestStrategy(
            existing_tests=test_strategy.get("existing_tests", []),
            new_tests_required=test_strategy.get("new_tests_required", []),
            test_commands=test_strategy.get("test_commands", []),
        )
    
    doc = PlanDocument(
        project_goal=goal,
        non_goals=non_goals or [],
        assumptions=assumptions or [],
        constraints=constraints or [],
        tasks=task_objects,
        test_strategy=ts,
        success_criteria=success_criteria or [],
        escalation_threshold=escalation_threshold,
    )
    
    generator = PlanGenerator(doc)
    return generator.generate()


if __name__ == "__main__":
    # Example usage
    plan = create_plan(
        goal="Add user authentication to the API",
        tasks=[
            {
                "id": "auth-models",
                "title": "Create User Model",
                "description": "Add SQLAlchemy User model with password hashing.",
                "files_touched": ["models/user.py", "models/__init__.py"],
                "risks": ["Password hashing library selection"],
                "complexity": "low",
            },
            {
                "id": "auth-routes", 
                "title": "Add Auth Routes",
                "description": "Create /login and /register endpoints.",
                "files_touched": ["routes/auth.py", "routes/__init__.py"],
                "dependencies": ["auth-models"],
                "risks": ["JWT vs session tokens"],
                "complexity": "medium",
            },
        ],
        non_goals=["OAuth integration", "Two-factor authentication"],
        assumptions=["PostgreSQL database is already configured"],
        constraints=["Must use bcrypt for password hashing"],
        success_criteria=[
            "Users can register with email and password",
            "Users can log in and receive a token",
            "Protected routes reject unauthenticated requests",
        ],
        test_strategy={
            "existing_tests": ["tests/test_api.py"],
            "new_tests_required": [
                "Test user registration",
                "Test login flow",
                "Test auth middleware",
            ],
            "test_commands": ["pytest tests/ -v"],
        },
    )
    
    print(plan)

