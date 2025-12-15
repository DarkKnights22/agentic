"""
Execution Engine

Orchestrates Phase 2 task execution:
- Loads and validates the locked plan
- Spawns sub-agents per task
- Applies patches and runs tests
- Manages retry logic and escalation
- Updates status and attempt records
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from .plan_parser import load_plan, ParsedPlan, ParsedTask
from .sub_agent import SubAgent, build_task_context, SubAgentResult
from .tools import apply_patch, revert_patch, run_tests, PatchResult, TestResult
from .state_manager import (
    StateManager, TaskStatus, TaskState, TaskAttempt,
    PlanStatus, ContextViolationError,
)
from .openrouter_client import DEFAULT_MODEL, TokenUsage


class ExecutionStatus(Enum):
    """Overall execution status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    HALTED = "halted"  # User intervention required


@dataclass
class TaskExecutionResult:
    """Result of executing a single task."""
    task_id: str
    success: bool
    attempts: int
    final_status: str
    error_message: Optional[str] = None
    patches_applied: list[str] = field(default_factory=list)
    test_log: Optional[str] = None


@dataclass
class ExecutionSummary:
    """Summary of the complete execution run."""
    status: ExecutionStatus
    tasks_completed: int
    tasks_failed: int
    tasks_escalated: int
    total_attempts: int
    duration_seconds: float
    task_results: list[TaskExecutionResult] = field(default_factory=list)


class ExecutionEngine:
    """
    Orchestrates execution of tasks from a locked plan.
    
    Workflow:
    1. Load and validate the locked plan
    2. For each task (sequential):
       - Build minimal context
       - Spawn sub-agent
       - Apply patches
       - Run tests
       - Handle success/failure/retry
    3. Update status after each task
    4. Generate execution summary
    
    Usage:
        engine = ExecutionEngine(repo_path, api_key=api_key)
        summary = await engine.run()
    """
    
    def __init__(
        self,
        repo_path: str | Path,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_attempts: int = 3,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.api_key = api_key
        self.model = model
        self.max_attempts = max_attempts
        self.on_status = on_status or (lambda x: None)
        
        self.state = StateManager(self.repo_path)
        self.plan: Optional[ParsedPlan] = None
        self.logs_path = self.repo_path / ".agentic" / "logs"
        self.tasks_path = self.repo_path / ".agentic" / "tasks"
        
        # Token usage tracking
        self.total_token_usage = TokenUsage()
    
    def _log(self, message: str):
        """Log a status message."""
        self.on_status(message)
    
    def _validate_plan(self) -> bool:
        """Validate that the plan is locked and ready for execution."""
        # Load session metadata
        meta = self.state.load_session()
        if not meta:
            self._log("[ERROR] No session found. Run planning phase first.")
            return False
        
        if meta.plan_status != PlanStatus.LOCKED.value:
            self._log(f"[ERROR] Plan is not locked. Current status: {meta.plan_status}")
            return False
        
        # Verify plan integrity
        if not self.state.verify_plan_integrity():
            self._log("[ERROR] Plan integrity check failed. Plan may have been modified.")
            return False
        
        return True
    
    def _load_plan(self) -> bool:
        """Load and parse the plan."""
        try:
            self.plan = load_plan(self.state.plan_path)
            return True
        except Exception as e:
            self._log(f"[ERROR] Failed to load plan: {e}")
            return False
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.tasks_path.mkdir(parents=True, exist_ok=True)
    
    def _save_task_definition(self, task: ParsedTask):
        """Save individual task definition to disk."""
        task_file = self.tasks_path / f"{task.id}.json"
        task_data = {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "files_touched": task.files_touched,
            "dependencies": task.dependencies,
            "risks": task.risks,
            "complexity": task.complexity,
        }
        task_file.write_text(json.dumps(task_data, indent=2), encoding='utf-8')
    
    def _save_attempt_record(
        self,
        task_id: str,
        attempt_num: int,
        sub_agent_result: SubAgentResult,
        patch_results: list[PatchResult],
        test_result: Optional[TestResult],
        success: bool,
        error: Optional[str] = None,
    ):
        """Save a detailed attempt record as markdown."""
        attempts_path = self.repo_path / ".agentic" / "attempts"
        attempts_path.mkdir(parents=True, exist_ok=True)
        
        record_file = attempts_path / f"{task_id}_attempt_{attempt_num}.md"
        
        lines = [
            f"# Attempt {attempt_num} for Task: {task_id}",
            "",
            f"**Timestamp:** {datetime.utcnow().isoformat()}Z",
            f"**Success:** {success}",
            "",
        ]
        
        if error:
            lines.extend([
                "## Error",
                "",
                f"```\n{error}\n```",
                "",
            ])
        
        if sub_agent_result.analysis:
            lines.extend([
                "## Analysis",
                "",
                sub_agent_result.analysis,
                "",
            ])
        
        if sub_agent_result.patches:
            lines.extend([
                "## Patches",
                "",
            ])
            for patch in sub_agent_result.patches:
                lines.extend([
                    f"### `{patch.file}`",
                    "",
                    "```diff",
                    patch.diff,
                    "```",
                    "",
                ])
        
        if patch_results:
            lines.extend([
                "## Patch Application Results",
                "",
            ])
            for pr in patch_results:
                status = "Success" if pr.success else "Failed"
                lines.append(f"- `{pr.file_path}`: {status}")
                if pr.error_message:
                    lines.append(f"  - Error: {pr.error_message}")
            lines.append("")
        
        if test_result:
            lines.extend([
                "## Test Results",
                "",
                f"**Command:** `{test_result.command}`",
                f"**Exit Code:** {test_result.exit_code}",
                f"**Duration:** {test_result.duration_seconds:.2f}s",
                "",
            ])
            if test_result.log_file:
                lines.append(f"**Log File:** `{test_result.log_file}`")
                lines.append("")
            
            if not test_result.success and test_result.stderr:
                lines.extend([
                    "### Error Output",
                    "",
                    "```",
                    test_result.stderr[:2000],
                    "```",
                    "",
                ])
        
        record_file.write_text('\n'.join(lines), encoding='utf-8')
    
    def _get_previous_attempts(self, task_id: str) -> list[dict]:
        """Get summaries of previous attempts for a task."""
        attempts_path = self.repo_path / ".agentic" / "attempts"
        attempts = []
        
        attempt_num = 1
        while True:
            attempt_file = attempts_path / f"{task_id}_attempt_{attempt_num}.md"
            if not attempt_file.exists():
                break
            
            # Parse minimal info from the file
            content = attempt_file.read_text(encoding='utf-8')
            success = "**Success:** True" in content
            
            # Extract error if present
            error = None
            if "## Error" in content:
                import re
                error_match = re.search(r'## Error\n\n```\n([\s\S]*?)\n```', content)
                if error_match:
                    error = error_match.group(1)
            
            attempts.append({
                "attempt": attempt_num,
                "success": success,
                "error": error,
            })
            
            attempt_num += 1
        
        return attempts
    
    def _update_status_file(self, task_results: list[TaskExecutionResult]):
        """Update status.md with current progress."""
        lines = [
            "# Execution Status",
            "",
            f"**Updated:** {datetime.utcnow().isoformat()}Z",
            "",
            "## Task Progress",
            "",
        ]
        
        for result in task_results:
            if result.success:
                emoji = "✅"
            elif result.final_status == TaskStatus.ESCALATED.value:
                emoji = "🚨"
            elif result.final_status == TaskStatus.IN_PROGRESS.value:
                emoji = "🔄"
            elif result.final_status == TaskStatus.PENDING.value:
                emoji = "⏳"
            else:
                emoji = "❌"
            
            lines.append(f"- {emoji} **{result.task_id}**: {result.final_status} ({result.attempts} attempts)")
            
            if result.error_message:
                lines.append(f"  - Error: {result.error_message[:100]}")
        
        lines.append("")
        
        self.state.update_status('\n'.join(lines))
    
    def _get_test_command(self) -> Optional[str]:
        """Get the test command from plan or auto-detect."""
        if self.plan and self.plan.test_strategy:
            commands = self.plan.test_strategy.test_commands
            if commands:
                return commands[0]  # Use first command
        
        # Auto-detection happens in run_tests
        return None
    
    async def _execute_task(self, task: ParsedTask) -> TaskExecutionResult:
        """
        Execute a single task with retry logic.
        
        Returns TaskExecutionResult with final status.
        """
        self._log(f"\n[Task: {task.id}] Starting execution...")
        self._log(f"[Task: {task.id}] Title: {task.title}")
        self._log(f"[Task: {task.id}] Files: {', '.join(task.files_touched)}")
        
        # Save task definition
        self._save_task_definition(task)
        
        # Register task with state manager
        self.state.register_task(
            task_id=task.id,
            title=task.title,
            allowed_files=task.files_touched,
            max_attempts=self.max_attempts,
        )
        
        result = TaskExecutionResult(
            task_id=task.id,
            success=False,
            attempts=0,
            final_status=TaskStatus.PENDING.value,
        )
        
        for attempt in range(1, self.max_attempts + 1):
            result.attempts = attempt
            self._log(f"\n[Task: {task.id}] Attempt {attempt}/{self.max_attempts}")
            
            # Build context for sub-agent
            previous_attempts = self._get_previous_attempts(task.id)
            
            context = build_task_context(
                repo_path=self.repo_path,
                task_id=task.id,
                task_title=task.title,
                task_description=task.description,
                allowed_files=task.files_touched,
                previous_attempts=previous_attempts,
            )
            
            # Spawn sub-agent with debug logging
            logs_dir = self.repo_path / ".agentic" / "logs"
            sub_agent = SubAgent(
                api_key=self.api_key,
                model=self.model,
                logs_dir=logs_dir,
            )
            
            try:
                self._log(f"[Task: {task.id}] Sub-agent executing...")
                self._log(f"[Task: {task.id}] Debug log: {logs_dir / f'{task.id}_attempt_{attempt}_debug.txt'}")
                sub_result = await sub_agent.execute(context, attempt_num=attempt)
                
                if not sub_result.success:
                    self._log(f"[Task: {task.id}] Sub-agent failed: {sub_result.error_message}")
                    
                    self._save_attempt_record(
                        task_id=task.id,
                        attempt_num=attempt,
                        sub_agent_result=sub_result,
                        patch_results=[],
                        test_result=None,
                        success=False,
                        error=sub_result.error_message,
                    )
                    
                    self.state.record_attempt(
                        task_id=task.id,
                        sub_agent_id=sub_agent.sub_agent_id,
                        success=False,
                        error_message=sub_result.error_message,
                    )
                    
                    continue
                
                self._log(f"[Task: {task.id}] Sub-agent analysis: {sub_result.analysis}")
                self._log(f"[Task: {task.id}] Tokens used: {sub_result.token_usage.format()}")
                
                # Accumulate token usage
                self.total_token_usage = self.total_token_usage.add(sub_result.token_usage)
                
                self._log(f"[Task: {task.id}] Patches to apply: {len(sub_result.patches)}")
                
                # Apply patches
                patch_results = []
                all_patches_ok = True
                
                for patch in sub_result.patches:
                    self._log(f"[Task: {task.id}] Applying patch to {patch.file}...")
                    pr = apply_patch(self.repo_path, patch.file, patch.diff)
                    patch_results.append(pr)
                    
                    if pr.success:
                        self._log(f"[Task: {task.id}] Patch applied (+{pr.lines_added} -{pr.lines_removed})")
                        result.patches_applied.append(patch.file)
                    else:
                        self._log(f"[Task: {task.id}] Patch failed: {pr.error_message}")
                        all_patches_ok = False
                
                if not all_patches_ok:
                    # Revert all patches
                    self._log(f"[Task: {task.id}] Reverting patches due to failure...")
                    for pr in patch_results:
                        if pr.success:
                            revert_patch(self.repo_path, pr)
                    
                    error_msg = "One or more patches failed to apply"
                    self._save_attempt_record(
                        task_id=task.id,
                        attempt_num=attempt,
                        sub_agent_result=sub_result,
                        patch_results=patch_results,
                        test_result=None,
                        success=False,
                        error=error_msg,
                    )
                    
                    self.state.record_attempt(
                        task_id=task.id,
                        sub_agent_id=sub_agent.sub_agent_id,
                        success=False,
                        error_message=error_msg,
                        changes_made=[p.file for p in sub_result.patches],
                    )
                    
                    continue
                
                # Run tests if requested
                test_result = None
                if sub_result.run_tests:
                    self._log(f"[Task: {task.id}] Running tests...")
                    
                    test_command = self._get_test_command()
                    log_file = self.logs_path / f"{task.id}_tests.log"
                    
                    test_result = run_tests(
                        repo_path=self.repo_path,
                        command=test_command,
                        log_file=log_file,
                    )
                    
                    result.test_log = str(log_file)
                    
                    if test_result.skipped:
                        self._log(f"[Task: {task.id}] No test framework found - tests skipped")
                    elif test_result.success:
                        self._log(f"[Task: {task.id}] Tests passed! ({test_result.duration_seconds:.1f}s)")
                    else:
                        self._log(f"[Task: {task.id}] Tests failed (exit code {test_result.exit_code})")
                        self._log(f"[Task: {task.id}] See log: {log_file}")
                        
                        # Don't revert patches on test failure - keep changes for retry
                        error_msg = f"Tests failed with exit code {test_result.exit_code}"
                        
                        self._save_attempt_record(
                            task_id=task.id,
                            attempt_num=attempt,
                            sub_agent_result=sub_result,
                            patch_results=patch_results,
                            test_result=test_result,
                            success=False,
                            error=error_msg,
                        )
                        
                        self.state.record_attempt(
                            task_id=task.id,
                            sub_agent_id=sub_agent.sub_agent_id,
                            success=False,
                            error_message=error_msg,
                            changes_made=[p.file for p in sub_result.patches],
                        )
                        
                        continue
                
                # Success!
                self._log(f"[Task: {task.id}] Task completed successfully!")
                
                self._save_attempt_record(
                    task_id=task.id,
                    attempt_num=attempt,
                    sub_agent_result=sub_result,
                    patch_results=patch_results,
                    test_result=test_result,
                    success=True,
                )
                
                self.state.record_attempt(
                    task_id=task.id,
                    sub_agent_id=sub_agent.sub_agent_id,
                    success=True,
                    changes_made=[p.file for p in sub_result.patches],
                )
                
                result.success = True
                result.final_status = TaskStatus.COMPLETED.value
                return result
                
            finally:
                await sub_agent.close()
        
        # Max attempts reached - escalate
        self._log(f"[Task: {task.id}] Max attempts reached. Escalating...")
        result.final_status = TaskStatus.ESCALATED.value
        result.error_message = f"Failed after {self.max_attempts} attempts"
        
        return result
    
    async def run(self) -> ExecutionSummary:
        """
        Run the complete execution of all tasks.
        
        Returns ExecutionSummary with results.
        """
        start_time = datetime.now()
        
        self._log("[Execution Engine] Starting Phase 2 execution...")
        
        # Validate plan
        if not self._validate_plan():
            return ExecutionSummary(
                status=ExecutionStatus.FAILED,
                tasks_completed=0,
                tasks_failed=0,
                tasks_escalated=0,
                total_attempts=0,
                duration_seconds=0,
            )
        
        # Load plan
        if not self._load_plan():
            return ExecutionSummary(
                status=ExecutionStatus.FAILED,
                tasks_completed=0,
                tasks_failed=0,
                tasks_escalated=0,
                total_attempts=0,
                duration_seconds=0,
            )
        
        self._log(f"[Execution Engine] Loaded plan with {len(self.plan.tasks)} tasks")
        
        # Ensure directories
        self._ensure_directories()
        
        # Execute tasks sequentially
        task_results = []
        halted = False
        
        for i, task in enumerate(self.plan.tasks, 1):
            self._log(f"\n{'='*60}")
            self._log(f"[Execution Engine] Task {i}/{len(self.plan.tasks)}")
            
            result = await self._execute_task(task)
            task_results.append(result)
            
            # Update status file
            self._update_status_file(task_results)
            
            # Check for escalation (halt execution)
            if result.final_status == TaskStatus.ESCALATED.value:
                self._log(f"\n[Execution Engine] Task escalated. Halting execution.")
                halted = True
                break
        
        # Calculate summary
        duration = (datetime.now() - start_time).total_seconds()
        
        completed = sum(1 for r in task_results if r.success)
        failed = sum(1 for r in task_results if not r.success and r.final_status != TaskStatus.ESCALATED.value)
        escalated = sum(1 for r in task_results if r.final_status == TaskStatus.ESCALATED.value)
        total_attempts = sum(r.attempts for r in task_results)
        
        if halted:
            status = ExecutionStatus.HALTED
        elif completed == len(self.plan.tasks):
            status = ExecutionStatus.COMPLETED
        elif escalated > 0:
            status = ExecutionStatus.HALTED
        else:
            status = ExecutionStatus.FAILED
        
        summary = ExecutionSummary(
            status=status,
            tasks_completed=completed,
            tasks_failed=failed,
            tasks_escalated=escalated,
            total_attempts=total_attempts,
            duration_seconds=duration,
            task_results=task_results,
        )
        
        self._log(f"\n{'='*60}")
        self._log(f"[Execution Engine] Execution complete!")
        self._log(f"[Execution Engine] Status: {status.value}")
        self._log(f"[Execution Engine] Completed: {completed}/{len(self.plan.tasks)}")
        self._log(f"[Execution Engine] Duration: {duration:.1f}s")
        self._log(f"[Execution Engine] Total tokens used: {self.total_token_usage.format()}")
        
        return summary


async def run_execution(
    repo_path: str | Path,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_attempts: int = 3,
    on_status: Optional[Callable[[str], None]] = None,
) -> ExecutionSummary:
    """
    Convenience function to run execution.
    
    Args:
        repo_path: Path to repository
        api_key: OpenRouter API key
        model: Model to use
        max_attempts: Max retries per task
        on_status: Status callback
        
    Returns:
        ExecutionSummary with results
    """
    engine = ExecutionEngine(
        repo_path=repo_path,
        api_key=api_key,
        model=model,
        max_attempts=max_attempts,
        on_status=on_status,
    )
    
    return await engine.run()


if __name__ == "__main__":
    import sys
    
    async def main():
        repo = sys.argv[1] if len(sys.argv) > 1 else "."
        
        def status_callback(msg: str):
            print(msg)
        
        summary = await run_execution(
            repo_path=repo,
            on_status=status_callback,
        )
        
        print(f"\nFinal status: {summary.status.value}")
        print(f"Tasks completed: {summary.tasks_completed}")
        print(f"Tasks escalated: {summary.tasks_escalated}")
    
    asyncio.run(main())

