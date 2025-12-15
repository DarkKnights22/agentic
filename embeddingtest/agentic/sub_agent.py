"""
Sub-Agent

Isolated task executor for Phase 2. Each sub-agent:
- Receives only its specific task and minimal context
- Proposes code changes via unified diffs
- Never sees unrelated tasks or full repo history
- Is stateless and disposable after task completion
"""

import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from .openrouter_client import OpenRouterClient, Message, TokenUsage, DEFAULT_MODEL
from .tools import read_file_safe


class SubAgentDebugLogger:
    """Debug logger for sub-agent executions."""
    
    def __init__(self, logs_dir: Path, task_id: str, attempt_num: int):
        self.logs_dir = logs_dir
        self.task_id = task_id
        self.attempt_num = attempt_num
        self.log_file = logs_dir / f"{task_id}_attempt_{attempt_num}_debug.txt"
        self._ensure_dir()
    
    def _ensure_dir(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log_execution_start(self, context: 'TaskContext', system_prompt: str):
        """Log the full context given to the sub-agent."""
        content = f"""{'#' * 80}
#  SUB-AGENT DEBUG LOG
#  Task: {context.task_id}
#  Attempt: {self.attempt_num}
#  Started: {self._timestamp()}
{'#' * 80}

================================================================================
TASK CONTEXT
================================================================================
Task ID: {context.task_id}
Task Title: {context.task_title}
Allowed Files: {', '.join(context.allowed_files)}
Previous Attempts: {len(context.previous_attempts)}

TASK DESCRIPTION:
{context.task_description}

================================================================================
FILE CONTENTS PROVIDED
================================================================================
"""
        for path, file_content in context.file_contents.items():
            if file_content:
                content += f"\n--- {path} ({len(file_content)} chars) ---\n"
                content += file_content[:5000]
                if len(file_content) > 5000:
                    content += f"\n... [truncated, {len(file_content)} total chars]"
                content += "\n"
            else:
                content += f"\n--- {path} (NEW FILE - does not exist yet) ---\n"
        
        content += f"""

================================================================================
FULL SYSTEM PROMPT SENT TO LLM
================================================================================
{system_prompt}

================================================================================
WAITING FOR LLM RESPONSE...
================================================================================
"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def log_response(self, raw_response: str, result: 'SubAgentResult'):
        """Log the LLM's response and parsed result."""
        content = f"""

================================================================================
LLM RESPONSE RECEIVED
Time: {self._timestamp()}
================================================================================

--- RAW RESPONSE ---
{raw_response}

================================================================================
PARSED RESULT
================================================================================
Success: {result.success}
Analysis: {result.analysis}
Run Tests: {result.run_tests}
Notes: {result.notes}
Error: {result.error_message or 'None'}
Patches: {len(result.patches)}

"""
        for i, patch in enumerate(result.patches):
            content += f"\n--- PATCH {i+1}: {patch.file} ---\n"
            content += patch.diff
            content += "\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def log_error(self, error: str):
        """Log an error."""
        content = f"""

================================================================================
ERROR
Time: {self._timestamp()}
================================================================================
{error}
"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)


# System prompt for sub-agents - focused on producing diffs
SUB_AGENT_SYSTEM_PROMPT = '''You are a Sub-Agent in an agentic coding system. Your role is to execute a SINGLE task.

## Your Task
{task_description}

## Files You May Modify
{allowed_files}

## Your Constraints
1. You may ONLY modify files listed above
2. You must produce changes as UNIFIED DIFFS
3. Follow the existing code style exactly
4. Make minimal, focused changes
5. Do not add unnecessary features or refactoring

## Previous Attempts (if any)
{previous_attempts}

## Output Format

Respond with a JSON object:

```json
{{
  "analysis": "Brief explanation of your approach (1-2 sentences)",
  "patches": [
    {{
      "file": "path/to/file.py",
      "diff": "unified diff content here"
    }}
  ],
  "run_tests": true,
  "notes": "Any important notes about the changes (optional)"
}}
```

## Unified Diff Format

Your diffs MUST follow the standard unified diff format:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,6 +10,9 @@ def existing_function():
     existing_line()
 
+def new_function():
+    new_code_here()
+
 def another_existing():
```

Key rules:
- Lines starting with `-` are REMOVED
- Lines starting with `+` are ADDED
- Lines starting with ` ` (space) are CONTEXT (unchanged)
- The @@ line shows line numbers: -old_start,old_count +new_start,new_count
- Include 3 lines of context before and after changes

## File Contents

{file_contents}

Now implement the task. Produce valid JSON with unified diffs.
'''


@dataclass
class Patch:
    """A single file patch."""
    file: str
    diff: str


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""
    success: bool
    analysis: str = ""
    patches: list[Patch] = field(default_factory=list)
    run_tests: bool = True
    notes: str = ""
    error_message: Optional[str] = None
    raw_response: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class TaskContext:
    """Context provided to a sub-agent for a task."""
    task_id: str
    task_title: str
    task_description: str
    allowed_files: list[str]
    file_contents: dict[str, str]  # path -> content
    previous_attempts: list[dict] = field(default_factory=list)


class SubAgent:
    """
    Isolated sub-agent for executing a single task.
    
    Each sub-agent:
    - Is created fresh for each task attempt
    - Receives minimal, scoped context
    - Produces unified diffs as output
    - Is disposed after completion
    
    Usage:
        context = TaskContext(
            task_id="add-endpoint",
            task_title="Add user count endpoint",
            task_description="Create /admin/users/count endpoint",
            allowed_files=["routes/admin.py"],
            file_contents={"routes/admin.py": "..."},
        )
        
        agent = SubAgent(api_key="...", logs_dir=Path(".agentic/logs"))
        result = await agent.execute(context, attempt_num=1)
        
        if result.success:
            for patch in result.patches:
                apply_patch(repo_path, patch.file, patch.diff)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        logs_dir: Optional[Path] = None,
    ):
        self.llm = OpenRouterClient(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more deterministic code
        )
        self._sub_agent_id = self._generate_id()
        self.logs_dir = logs_dir
    
    @property
    def sub_agent_id(self) -> str:
        """Unique identifier for this sub-agent instance."""
        return self._sub_agent_id
    
    def _generate_id(self) -> str:
        """Generate a unique sub-agent ID."""
        import hashlib
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:12]
    
    def _build_prompt(self, context: TaskContext) -> str:
        """Build the system prompt with task context."""
        # Format task description
        task_desc = f"""**Task ID:** {context.task_id}
**Title:** {context.task_title}

{context.task_description}"""
        
        # Format allowed files
        allowed = '\n'.join(f'- `{f}`' for f in context.allowed_files)
        
        # Format previous attempts
        if context.previous_attempts:
            attempts_text = []
            for i, attempt in enumerate(context.previous_attempts, 1):
                status = "Success" if attempt.get('success') else "Failed"
                error = attempt.get('error', 'N/A')
                attempts_text.append(f"**Attempt {i}:** {status}\nError: {error}")
            previous = '\n\n'.join(attempts_text)
        else:
            previous = "This is the first attempt."
        
        # Format file contents
        file_sections = []
        for path, content in context.file_contents.items():
            if content:
                file_sections.append(f"### `{path}`\n\n```\n{content}\n```")
            else:
                file_sections.append(f"### `{path}`\n\n*File does not exist yet (will be created)*")
        
        file_contents = '\n\n'.join(file_sections) if file_sections else "*No files provided*"
        
        return SUB_AGENT_SYSTEM_PROMPT.format(
            task_description=task_desc,
            allowed_files=allowed,
            previous_attempts=previous,
            file_contents=file_contents,
        )
    
    def _parse_response(self, content: str) -> SubAgentResult:
        """Parse the LLM's JSON response."""
        # Try to extract JSON from response
        json_str = None
        
        # Strategy 1: Find ```json ... ``` block
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
        
        # Strategy 2: Find any ``` ... ``` block that looks like JSON
        if not json_str:
            code_block_match = re.search(r'```\s*([\s\S]*?)\s*```', content)
            if code_block_match:
                potential = code_block_match.group(1).strip()
                if potential.startswith('{'):
                    json_str = potential
        
        # Strategy 3: Try the whole content
        if not json_str:
            json_str = content.strip()
        
        try:
            data = json.loads(json_str)
            
            # Extract patches
            patches = []
            for p in data.get('patches', []):
                patches.append(Patch(
                    file=p.get('file', ''),
                    diff=p.get('diff', ''),
                ))
            
            return SubAgentResult(
                success=True,
                analysis=data.get('analysis', ''),
                patches=patches,
                run_tests=data.get('run_tests', True),
                notes=data.get('notes', ''),
                raw_response=content,
            )
            
        except json.JSONDecodeError as e:
            return SubAgentResult(
                success=False,
                error_message=f"Failed to parse JSON response: {e}",
                raw_response=content,
            )
    
    async def execute(self, context: TaskContext, attempt_num: int = 1) -> SubAgentResult:
        """
        Execute the task and return proposed changes.
        
        Args:
            context: TaskContext with all required information
            attempt_num: Current attempt number for logging
            
        Returns:
            SubAgentResult with patches or error
        """
        # Set up debug logging if logs_dir is configured
        debug_logger = None
        if self.logs_dir:
            debug_logger = SubAgentDebugLogger(self.logs_dir, context.task_id, attempt_num)
        
        try:
            # Build prompt
            system_prompt = self._build_prompt(context)
            
            # Log execution start
            if debug_logger:
                debug_logger.log_execution_start(context, system_prompt)
            
            # Create messages
            messages = [
                Message("system", system_prompt),
                Message("user", "Execute the task now. Produce the JSON with unified diffs."),
            ]
            
            # Get LLM response
            response = await self.llm.chat(messages)
            
            # Parse response
            result = self._parse_response(response.content)
            
            # Attach token usage to result
            result.token_usage = response.token_usage
            
            # Log response
            if debug_logger:
                debug_logger.log_response(response.content, result)
            
            # Validate patches reference allowed files only
            for patch in result.patches:
                if patch.file not in context.allowed_files:
                    error_result = SubAgentResult(
                        success=False,
                        error_message=f"Patch references unauthorized file: {patch.file}",
                        raw_response=response.content,
                        token_usage=response.token_usage,
                    )
                    if debug_logger:
                        debug_logger.log_error(f"Unauthorized file access: {patch.file}")
                    return error_result
            
            return result
            
        except Exception as e:
            error_result = SubAgentResult(
                success=False,
                error_message=str(e),
            )
            if debug_logger:
                debug_logger.log_error(str(e))
            return error_result
    
    async def close(self):
        """Clean up resources."""
        await self.llm.close()


def build_task_context(
    repo_path: Path,
    task_id: str,
    task_title: str,
    task_description: str,
    allowed_files: list[str],
    previous_attempts: list[dict] = None,
) -> TaskContext:
    """
    Build a TaskContext by reading the allowed files.
    
    Args:
        repo_path: Path to repository root
        task_id: Task identifier
        task_title: Task title
        task_description: Task description
        allowed_files: List of file paths to include
        previous_attempts: List of previous attempt records
        
    Returns:
        TaskContext ready for sub-agent execution
    """
    file_contents = {}
    
    for file_path in allowed_files:
        content = read_file_safe(repo_path, file_path)
        file_contents[file_path] = content  # May be None for new files
    
    return TaskContext(
        task_id=task_id,
        task_title=task_title,
        task_description=task_description,
        allowed_files=allowed_files,
        file_contents=file_contents,
        previous_attempts=previous_attempts or [],
    )


if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Create a test context
        context = TaskContext(
            task_id="test-task",
            task_title="Add hello function",
            task_description="Add a function called hello() that prints 'Hello, World!'",
            allowed_files=["test.py"],
            file_contents={
                "test.py": """def goodbye():
    print("Goodbye!")
""",
            },
        )
        
        agent = SubAgent()
        result = await agent.execute(context)
        
        print(f"Success: {result.success}")
        print(f"Analysis: {result.analysis}")
        print(f"Patches: {len(result.patches)}")
        
        for patch in result.patches:
            print(f"\n--- Patch for {patch.file} ---")
            print(patch.diff)
        
        await agent.close()
    
    asyncio.run(test())

