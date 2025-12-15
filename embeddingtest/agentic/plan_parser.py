"""
Plan Parser

Parses a locked plan.md file into executable task objects.
Extracts task definitions, context contracts, and test strategy.
"""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ParsedTask:
    """A task extracted from plan.md."""
    id: str
    title: str
    description: str
    files_touched: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    complexity: str = "medium"


@dataclass
class ParsedTestStrategy:
    """Test strategy extracted from plan.md."""
    existing_tests: list[str] = field(default_factory=list)
    new_tests_required: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)


@dataclass
class ParsedPlan:
    """Complete parsed plan from plan.md."""
    goal: str = ""
    non_goals: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    tasks: list[ParsedTask] = field(default_factory=list)
    test_strategy: Optional[ParsedTestStrategy] = None
    success_criteria: list[str] = field(default_factory=list)
    escalation_threshold: int = 3
    version: int = 1


class PlanParseError(Exception):
    """Error parsing plan.md."""
    pass


def extract_section(content: str, header: str, next_headers: list[str] = None) -> str:
    """
    Extract content between a header and the next header.
    
    Args:
        content: Full markdown content
        header: Header to find (e.g., "# Project Goal")
        next_headers: List of possible next headers (defaults to any # header)
        
    Returns:
        Content between headers, stripped
    """
    # Find the header
    header_pattern = re.compile(rf'^{re.escape(header)}\s*$', re.MULTILINE)
    header_match = header_pattern.search(content)
    
    if not header_match:
        return ""
    
    start_pos = header_match.end()
    
    # Find the next header (only top-level # headers, not ## subsections)
    if next_headers:
        next_pattern = '|'.join(re.escape(h) for h in next_headers)
        next_match = re.search(rf'^({next_pattern})\s*$', content[start_pos:], re.MULTILINE)
    else:
        # Only match single # headers (not ## or ###)
        next_match = re.search(r'^# [^#]', content[start_pos:], re.MULTILINE)
    
    if next_match:
        end_pos = start_pos + next_match.start()
    else:
        end_pos = len(content)
    
    return content[start_pos:end_pos].strip()


def extract_list_items(section_content: str) -> list[str]:
    """Extract bullet points or numbered items from a section."""
    items = []
    for line in section_content.split('\n'):
        line = line.strip()
        # Match bullet points: -, *, or numbered lists
        match = re.match(r'^[-*]\s+(.+)$|^\d+\.\s+(.+)$|^- \[ \]\s+(.+)$', line)
        if match:
            item = match.group(1) or match.group(2) or match.group(3)
            if item:
                items.append(item.strip())
    return items


def extract_code_items(section_content: str) -> list[str]:
    """Extract backtick-wrapped items from a section."""
    items = []
    for line in section_content.split('\n'):
        line = line.strip()
        match = re.match(r'^[-*]\s+`([^`]+)`', line)
        if match:
            items.append(match.group(1))
    return items


def extract_code_block(section_content: str) -> list[str]:
    """Extract content from a code block."""
    match = re.search(r'```(?:\w+)?\n([\s\S]*?)```', section_content)
    if match:
        return [line.strip() for line in match.group(1).strip().split('\n') if line.strip()]
    return []


def parse_task_section(task_content: str, task_num: int) -> ParsedTask:
    """Parse a single task section into a ParsedTask."""
    lines = task_content.strip().split('\n')
    
    # Extract task ID and title from first lines
    task_id = f"task-{task_num}"
    title = f"Task {task_num}"
    description = ""
    files_touched = []
    dependencies = []
    risks = []
    complexity = "medium"
    
    current_section = None
    section_content = []
    
    for line in lines:
        # Check for ID
        id_match = re.match(r'\*\*ID:\*\*\s*`([^`]+)`', line)
        if id_match:
            task_id = id_match.group(1)
            continue
        
        # Check for complexity
        complexity_match = re.match(r'\*\*Complexity:\*\*\s*(\w+)', line)
        if complexity_match:
            complexity = complexity_match.group(1).lower()
            continue
        
        # Check for subsection headers
        if line.startswith('### Description'):
            if current_section == 'description':
                description = '\n'.join(section_content).strip()
            current_section = 'description'
            section_content = []
            continue
        
        if line.startswith('### Files Touched'):
            if current_section == 'description':
                description = '\n'.join(section_content).strip()
            current_section = 'files'
            section_content = []
            continue
        
        if line.startswith('### Dependencies'):
            if current_section == 'description':
                description = '\n'.join(section_content).strip()
            current_section = 'dependencies'
            section_content = []
            continue
        
        if line.startswith('### Risks'):
            if current_section == 'description':
                description = '\n'.join(section_content).strip()
            current_section = 'risks'
            section_content = []
            continue
        
        # Collect content for current section
        if current_section:
            if current_section == 'files':
                match = re.match(r'^[-*]\s+`([^`]+)`', line.strip())
                if match:
                    files_touched.append(match.group(1))
            elif current_section == 'dependencies':
                match = re.match(r'^[-*]\s+Task\s+`([^`]+)`', line.strip())
                if match:
                    dependencies.append(match.group(1))
            elif current_section == 'risks':
                match = re.match(r'^[-*]\s+(.+)$', line.strip())
                if match:
                    risks.append(match.group(1))
            else:
                section_content.append(line)
    
    # Handle final section
    if current_section == 'description' and section_content:
        description = '\n'.join(section_content).strip()
    
    return ParsedTask(
        id=task_id,
        title=title,
        description=description,
        files_touched=files_touched,
        dependencies=dependencies,
        risks=risks,
        complexity=complexity,
    )


def parse_plan(content: str) -> ParsedPlan:
    """
    Parse a plan.md file into a ParsedPlan object.
    
    Args:
        content: Full markdown content of plan.md
        
    Returns:
        ParsedPlan with all extracted data
    """
    plan = ParsedPlan()
    
    # Extract version from header
    version_match = re.search(r'\*\*Version:\*\*\s*(\d+)', content)
    if version_match:
        plan.version = int(version_match.group(1))
    
    # Extract project goal
    goal_section = extract_section(content, '# Project Goal')
    plan.goal = goal_section.strip()
    
    # Extract non-goals
    non_goals_section = extract_section(content, '# Non-Goals')
    if '*No explicit non-goals' not in non_goals_section:
        plan.non_goals = extract_list_items(non_goals_section)
    
    # Extract assumptions
    assumptions_section = extract_section(content, '# Assumptions')
    if '*No assumptions' not in assumptions_section:
        plan.assumptions = extract_list_items(assumptions_section)
    
    # Extract constraints
    constraints_section = extract_section(content, '# Constraints')
    if '*No constraints' not in constraints_section:
        plan.constraints = extract_list_items(constraints_section)
    
    # Extract tasks
    task_breakdown = extract_section(content, '# Task Breakdown')
    
    # Split by task headers (## Task N: Title)
    task_pattern = re.compile(r'^## Task (\d+):\s*(.+)$', re.MULTILINE)
    task_matches = list(task_pattern.finditer(task_breakdown))
    
    for i, match in enumerate(task_matches):
        task_num = int(match.group(1))
        title = match.group(2).strip()
        
        # Get content until next task or end
        start = match.end()
        end = task_matches[i + 1].start() if i + 1 < len(task_matches) else len(task_breakdown)
        task_content = task_breakdown[start:end]
        
        task = parse_task_section(task_content, task_num)
        task.title = title
        plan.tasks.append(task)
    
    # Extract test strategy
    test_section = extract_section(content, '# Test Strategy')
    if test_section and '*Test strategy not yet defined' not in test_section:
        test_strategy = ParsedTestStrategy()
        
        # Existing tests
        existing_match = re.search(r'## Existing Tests to Run\n([\s\S]*?)(?=\n##|\Z)', test_section)
        if existing_match:
            test_strategy.existing_tests = extract_code_items(existing_match.group(1))
        
        # New tests
        new_match = re.search(r'## New Tests Required\n([\s\S]*?)(?=\n##|\Z)', test_section)
        if new_match:
            test_strategy.new_tests_required = extract_list_items(new_match.group(1))
        
        # Test commands
        commands_match = re.search(r'## Test Commands\n([\s\S]*?)(?=\n#|\Z)', test_section)
        if commands_match:
            test_strategy.test_commands = extract_code_block(commands_match.group(1))
        
        plan.test_strategy = test_strategy
    
    # Extract success criteria
    success_section = extract_section(content, '# Success Criteria')
    if success_section and '*Success criteria not yet defined' not in success_section:
        plan.success_criteria = extract_list_items(success_section)
    
    # Extract escalation threshold
    escalation_section = extract_section(content, '# Escalation Rules')
    threshold_match = re.search(r'fails \*\*(\d+) times\*\*', escalation_section)
    if threshold_match:
        plan.escalation_threshold = int(threshold_match.group(1))
    
    return plan


def load_plan(plan_path: Path) -> ParsedPlan:
    """
    Load and parse a plan.md file.
    
    Args:
        plan_path: Path to plan.md
        
    Returns:
        ParsedPlan with all extracted data
        
    Raises:
        PlanParseError if file doesn't exist or can't be parsed
    """
    if not plan_path.exists():
        raise PlanParseError(f"Plan file not found: {plan_path}")
    
    try:
        content = plan_path.read_text(encoding='utf-8')
        return parse_plan(content)
    except Exception as e:
        raise PlanParseError(f"Failed to parse plan: {e}")


if __name__ == "__main__":
    # Test with a sample plan
    sample_plan = """# Agentic Plan Document

> **Version:** 1  
> **Created:** 2024-01-01T00:00:00Z  
> **Status:** Draft (pending approval)

---

# Project Goal

Add an admin API endpoint that returns the count of total current users.

# Non-Goals

- User management functionality
- Authentication changes

# Assumptions

- Database is already configured
- User model exists

# Constraints

- Must follow existing API patterns

# Task Breakdown

## Task 1: Add Admin User Count Endpoint

**ID:** `admin-user-count`
**Complexity:** low

### Description

Create a new admin endpoint at `/admin/users/count` that returns the total number of users.

### Files Touched

- `routes/admin.py`
- `tests/test_admin.py`

### Dependencies

### Risks / Unknowns

- Need to verify admin authentication is in place

---

# Context Contracts

> Sub-agents may only see:
> - Their specific task definition
> - Files explicitly listed for that task
> - Relevant prior attempt summaries

| Task ID | Allowed Files |
|---------|---------------|
| `admin-user-count` | `routes/admin.py`, `tests/test_admin.py` |

# Test Strategy

## Existing Tests to Run

- `tests/test_admin.py`

## New Tests Required

- Test user count endpoint returns correct count
- Test endpoint requires admin authentication

## Test Commands

```bash
pytest tests/test_admin.py -v
```

# Success Criteria

- [ ] Endpoint returns accurate user count
- [ ] Endpoint is protected by admin auth
- [ ] All tests pass

# Plan Lock

> **Once this plan is approved:**
> - Tasks may NOT be added or reordered
> - Scope changes require a new plan version
> - Sub-agents must reference this plan verbatim

This prevents silent scope drift and ensures reproducibility.

# Escalation Rules

If a task fails **3 times**:
1. Execution halts for that task
2. Task is escalated to Master Agent
3. Master Agent will either:
   - Re-plan the task with additional context
   - Ask the user for clarification
   - Mark the task as blocked

Sub-agents do NOT retry indefinitely.

---

*This document was generated by the Agentic CLI Master Agent.*
*Do not edit manually after approval.*
"""
    
    plan = parse_plan(sample_plan)
    
    print(f"Goal: {plan.goal[:50]}...")
    print(f"Non-goals: {plan.non_goals}")
    print(f"Tasks: {len(plan.tasks)}")
    
    for task in plan.tasks:
        print(f"\n  Task: {task.id} - {task.title}")
        print(f"  Files: {task.files_touched}")
        print(f"  Complexity: {task.complexity}")
    
    if plan.test_strategy:
        print(f"\nTest commands: {plan.test_strategy.test_commands}")
    
    print(f"\nEscalation threshold: {plan.escalation_threshold}")

