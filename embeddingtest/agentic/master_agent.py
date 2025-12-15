"""
Master Agent

The core orchestrator for Phase 1 planning. Handles:
- User prompt interpretation
- Repository reconnaissance (file tree + code analysis + semantic search)
- Clarification loops with the LLM
- Plan generation and approval workflow

Enterprise Integration:
- Supports custom LLM clients via the LLMClient protocol
- Semantic search can be disabled via config for enterprises without embedding models
"""

import asyncio
import json
import subprocess
import re
from pathlib import Path
from typing import Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

from .llm_client import LLMClient, Message, ChatResponse, TokenUsage
from .openrouter_client import OpenRouterClient, DEFAULT_MODEL
from .repo_scanner import RepoScanner, RepoStructure
from .state_manager import StateManager, PlanLockError
from .plan_generator import create_plan
from .context_discovery_agent import DiscoveredContext
from .config import AgenticConfig, get_config

# Try to import ChromaDB for semantic search
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class DebugLogger:
    """
    Comprehensive debug logger for the Master Agent.
    Creates detailed, human-readable logs of all LLM interactions.
    """
    
    def __init__(self, agentic_dir: Path):
        self.agentic_dir = agentic_dir
        self.log_file = agentic_dir / "debug_log.txt"
        self.context_file = agentic_dir / "debug_context.txt"
        self.round_num = 0
        self._ensure_dir()
    
    def _ensure_dir(self):
        self.agentic_dir.mkdir(parents=True, exist_ok=True)
    
    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _write_log(self, content: str, mode: str = "a"):
        """Append to the debug log file."""
        with open(self.log_file, mode, encoding="utf-8") as f:
            f.write(content)
    
    def _write_context(self, content: str):
        """Write the current context snapshot (overwrites)."""
        with open(self.context_file, "w", encoding="utf-8") as f:
            f.write(content)
    
    def start_session(self, user_prompt: str, repo_name: str):
        """Log the start of a new planning session."""
        header = f"""
{'#' * 80}
#  MASTER AGENT DEBUG LOG
#  Started: {self._timestamp()}
#  Repository: {repo_name}
{'#' * 80}

================================================================================
USER PROMPT
================================================================================
{user_prompt}

"""
        self._write_log(header, mode="w")  # New session = new log
    
    def log_round_start(self, round_num: int, messages: list):
        """Log the start of a tool round with full context."""
        prev_msg_count = getattr(self, '_prev_msg_count', 0)
        self.round_num = round_num
        
        # Build full context snapshot (OVERWRITES - this is the full context file)
        context_lines = [
            f"MASTER AGENT CONTEXT SNAPSHOT (Full Context)",
            f"Generated: {self._timestamp()}",
            f"Round: {round_num}",
            f"Total Messages: {len(messages)}",
            "=" * 80,
            "",
            "This file shows the COMPLETE context sent to the LLM this round.",
            "For incremental changes, see debug_log.txt",
            ""
        ]
        
        for i, msg in enumerate(messages):
            context_lines.append(f"\n{'─' * 80}")
            context_lines.append(f"MESSAGE {i} | Role: {msg.role.upper()}")
            context_lines.append(f"{'─' * 80}")
            context_lines.append(msg.content)
        
        self._write_context("\n".join(context_lines))
        
        # For the log, only show NEW messages (avoid duplication)
        new_messages = messages[prev_msg_count:]
        self._prev_msg_count = len(messages)
        
        log_entry = f"""
================================================================================
ROUND {round_num} - SENDING TO LLM
Time: {self._timestamp()}
Total Messages: {len(messages)} | New This Round: {len(new_messages)}
================================================================================
(Full context in debug_context.txt)

"""
        if new_messages:
            for i, msg in enumerate(new_messages):
                msg_idx = prev_msg_count + i
                log_entry += f"--- NEW Message {msg_idx} ({msg.role}) ---\n"
                log_entry += msg.content[:3000]  # Limit for log
                if len(msg.content) > 3000:
                    log_entry += f"\n... [truncated, {len(msg.content)} total chars]"
                log_entry += "\n\n"
        else:
            log_entry += "(No new messages - retry or continuation)\n\n"
        
        self._write_log(log_entry)
    
    def log_llm_response(self, response_content: str, parsed_keys: list):
        """Log the LLM's response."""
        log_entry = f"""
================================================================================
ROUND {self.round_num} - LLM RESPONSE
Time: {self._timestamp()}
Parsed keys: {parsed_keys}
================================================================================

{response_content}

"""
        self._write_log(log_entry)
        
        # Update context file with response
        with open(self.context_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'─' * 80}")
            f.write(f"\nLLM RESPONSE (Round {self.round_num})")
            f.write(f"\n{'─' * 80}\n")
            f.write(response_content)
    
    def log_tool_calls(self, tool_calls: list, was_limited: bool = False, original_count: int = 0):
        """Log tool calls being executed."""
        warning = ""
        if was_limited:
            warning = f"""
⚠️ WARNING: LLM requested {original_count} tool calls but was limited to {len(tool_calls)}!
   The LLM should explore incrementally, not batch speculative reads.
   
"""
        
        log_entry = f"""
--------------------------------------------------------------------------------
TOOL CALLS (Round {self.round_num})
Time: {self._timestamp()}
Count: {len(tool_calls)}
--------------------------------------------------------------------------------
{warning}
"""
        for i, tc in enumerate(tool_calls):
            tool_name = tc.get('tool', 'unknown')
            log_entry += f"  [{i+1}] {tool_name}\n"
            for k, v in tc.items():
                if k != 'tool':
                    log_entry += f"      {k}: {v}\n"
        
        self._write_log(log_entry)
    
    def log_tool_results(self, results: str):
        """Log tool execution results."""
        log_entry = f"""
--------------------------------------------------------------------------------
TOOL RESULTS (Round {self.round_num})
Time: {self._timestamp()}
Length: {len(results)} chars
--------------------------------------------------------------------------------

{results}

"""
        self._write_log(log_entry)
        
        # Update context file
        with open(self.context_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'─' * 80}")
            f.write(f"\nTOOL RESULTS (Round {self.round_num})")
            f.write(f"\n{'─' * 80}\n")
            f.write(results)
    
    def log_plan_detected(self, plan_data: dict):
        """Log when a plan is detected."""
        log_entry = f"""
================================================================================
✓ PLAN DETECTED
Time: {self._timestamp()}
================================================================================

Goal: {plan_data.get('goal', 'N/A')}
Tasks: {len(plan_data.get('tasks', []))}

Task Details:
"""
        for task in plan_data.get('tasks', []):
            log_entry += f"  - [{task.get('id')}] {task.get('title')}\n"
            log_entry += f"    Files: {task.get('files_touched', [])}\n"
        
        self._write_log(log_entry)
    
    def log_questions(self, questions: list):
        """Log questions being asked to user."""
        log_entry = f"""
================================================================================
? QUESTIONS FOR USER
Time: {self._timestamp()}
================================================================================

"""
        for i, q in enumerate(questions, 1):
            log_entry += f"  {i}. {q}\n"
        
        self._write_log(log_entry)
    
    def log_user_answer(self, answer: str):
        """Log user's answer to questions."""
        log_entry = f"""
================================================================================
USER ANSWER
Time: {self._timestamp()}
================================================================================

{answer}

"""
        self._write_log(log_entry)
    
    def log_error(self, error: str):
        """Log an error."""
        log_entry = f"""
================================================================================
⚠️ ERROR
Time: {self._timestamp()}
================================================================================

{error}

"""
        self._write_log(log_entry)


# System prompt for the Master Agent
# Note: Double curly braces {{ }} are escaped for Python .format()
MASTER_AGENT_SYSTEM_PROMPT = '''You are the Master Agent of an agentic coding system. Your role in Phase 1 is PLANNING ONLY.

## ⚠️ CRITICAL: How Tool Calling Works

This is an INTERACTIVE conversation. When you output tool calls:
1. Your message ENDS after the tool calls
2. The tools are EXECUTED by the system
3. You receive the ACTUAL RESULTS in the next message
4. THEN you decide what to do next

**NEVER output `ready_to_plan` in the same message as tool calls!**
**NEVER guess what tool results will be - WAIT for actual results!**

Each response must be ONE of:
- Tool calls ONLY (then stop and wait for results)
- Questions for the user (only if truly needed)
- Final plan with `ready_to_plan: true` (ONLY after you've read actual code)

## Your Responsibilities
1. Understand the user's high-level request
2. **USE TOOLS to explore the codebase** before asking any questions
3. Only ask the user questions that genuinely require human judgment
4. Create a detailed, actionable task plan

## AVAILABLE TOOLS - USE THESE FIRST!

Before asking ANY questions, you MUST use tools to understand the codebase:

1. **grep** - Search for patterns in code (regex supported)
   ```json
   {{"tool_calls": [{{"tool": "grep", "pattern": "@app.route|@router|FastAPI|Flask"}}]}}
   ```
   Optional: Add `"context_lines": 5` to include lines before/after matches:
   ```json
   {{"tool_calls": [{{"tool": "grep", "pattern": "def get_user", "context_lines": 10}}]}}
   ```
   NOTE: Do NOT use shell-style flags like `-A 10`. Use `context_lines` parameter instead.

2. **read_file** - Read a specific file (max 250 lines)
   ```json
   {{"tool_calls": [{{"tool": "read_file", "path": "main.py"}}]}}
   ```

3. **semantic_search** - Find code by meaning
   ```json
   {{"tool_calls": [{{"tool": "semantic_search", "query": "user authentication"}}]}}
   ```

4. **list_files** - Find files by pattern (returns max 10 results)
   ```json
   {{"tool_calls": [{{"tool": "list_files", "pattern": "*.py"}}]}}
   ```

5. **run_command** - Execute shell commands (git, build tools, etc.)
   ```json
   {{"tool_calls": [{{"tool": "run_command", "command": "git status"}}]}}
   ```

6. **get_lints** - Get linter errors for a file (Python, JS, TS, Rust, Go, etc.)
   ```json
   {{"tool_calls": [{{"tool": "get_lints", "file": "src/main.py"}}]}}
   ```

7. **get_symbols** - List functions/classes in a file (quick overview)
   ```json
   {{"tool_calls": [{{"tool": "get_symbols", "file": "src/main.py"}}]}}
   ```

8. **find_references** - Find all usages of a symbol across codebase
   ```json
   {{"tool_calls": [{{"tool": "find_references", "symbol": "UserModel"}}]}}
   ```

## WORKFLOW (follow this exactly)

1. **FIRST**: Call tools to explore (grep for routes, models, existing patterns)
2. **SECOND**: Read relevant files to understand implementation details
3. **VERIFY**: Confirm patterns you want to reference actually exist in the codebase
4. **THIRD**: Only if truly ambiguous after searching, ask the user
5. **FINALLY**: Generate the plan with ready_to_plan: true

**IMPORTANT**: Do NOT generate a plan until you have actually read the key files. 
If you mention "existing admin.py patterns", you MUST have read admin.py first.
If you mention "use the require_admin dependency", you MUST have verified it exists.

## CRITICAL: INCREMENTAL EXPLORATION (VERY IMPORTANT!)

**Explore the codebase ONE STEP AT A TIME:**

1. **Start with discovery tools ONLY** (grep, list_files) - never assume file paths!
2. **WAIT for results** before making more tool calls
3. **Use ACTUAL paths** from discovery results - don't guess paths like "app/main.py"
4. **Maximum 3-5 tool calls per response** - see results before deciding next steps

**BAD - Don't do this:**
```json
{{"tool_calls": [{{"tool": "grep", "pattern": "..."}}, {{"tool": "read_file", "path": "app/main.py"}}, {{"tool": "read_file", "path": "app/models/user.py"}}]}}
```
This guesses paths that may not exist! The actual structure might be `src/backend/main.py`.

**GOOD - Do this instead:**
```json
{{"tool_calls": [{{"tool": "list_files", "pattern": "main.py"}}, {{"tool": "grep", "pattern": "@app|FastAPI|Flask"}}]}}
```
Then WAIT for results to see the actual file paths before trying to read them.

## CRITICAL RULES
- **ALWAYS use tools before asking questions**
- Questions like "what framework?" should be answered by grepping, not asking
- You NEVER edit code directly
- You ONLY produce a structured plan
- Be specific about which files each task will touch

## PRE-DISCOVERED CONTEXT (TRUST IT!)

If the user message includes "Pre-Discovered Codebase Context", this has ALREADY been verified by the Context Discovery Agent. You should:
- **TRUST this context** - it's already been verified by reading actual files
- **Proceed to planning quickly** - don't re-explore what's already been found
- **Only use tools** if you need specific details NOT in the pre-discovered context

When good context is provided, you should often be able to create a plan in 1-2 rounds, not 8+.

## ANTI-HALLUCINATION RULES
- If NO pre-discovered context is provided, verify via tools before referencing patterns
- If pre-discovered context IS provided, trust it - it's already verified
- **Cite your sources**: Reference the pre-discovered context or files you read
- If the codebase is missing expected patterns, your plan should CREATE them

## TASK GRANULARITY (IMPORTANT!)
- **DO NOT over-split tasks** - one feature = one task, not 4 micro-tasks
- A single sub-agent should handle a coherent unit of work
- BAD: Separate tasks for "add function", "add model", "add endpoint", "add error handling"
- GOOD: One task "Add admin user count endpoint" that includes all related changes
- Only split into multiple tasks when there are genuinely independent features
- Error handling, response models, etc. are PART of implementing a feature, not separate tasks

## Output Format
When you have enough information to create a plan, respond with a JSON block:

```json
{{
  "ready_to_plan": true,
  "plan": {{
    "goal": "Clear description of what will be built/changed",
    "non_goals": ["Things explicitly NOT in scope"],
    "assumptions": ["Assumptions made due to missing info - ONLY if you could not verify via tools"],
    "constraints": ["Technical or process constraints - cite file if referencing existing code"],
    "tasks": [
      {{
        "id": "unique-task-id",
        "title": "Short task title",
        "description": "Detailed description. When referencing existing patterns, cite the file: 'Following the pattern in X.py line Y'",
        "files_touched": ["path/to/file.py"],
        "dependencies": ["other-task-id"],
        "risks": ["Potential issues or unknowns"],
        "complexity": "low|medium|high"
      }}
    ],
    "test_strategy": {{
      "existing_tests": ["tests/to/run.py - ONLY list files you verified exist via tools"],
      "new_tests_required": ["Description of new tests"],
      "test_commands": ["pytest tests/ -v"]
    }},
    "success_criteria": ["Measurable definition of done"]
  }}
}}
```

If you need more information, respond with:

```json
{{
  "ready_to_plan": false,
  "questions": [
    "Specific question 1?",
    "Specific question 2?"
  ]
}}
```
'''


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class PlanningSession:
    """State of an active planning session."""
    user_prompt: str
    repo_structure: Optional[RepoStructure] = None
    conversation: list[ConversationTurn] = field(default_factory=list)
    plan_data: Optional[dict] = None
    is_complete: bool = False
    discovered_context: Optional[DiscoveredContext] = None  # Pre-discovered context from Discovery Agent


class MasterAgent:
    """
    Master Agent for Phase 1 planning.
    
    Orchestrates the planning workflow:
    1. Accept user prompt
    2. Scan repository
    3. Engage in clarification loop with LLM
    4. Generate and save plan
    
    Usage (default - uses OpenRouter):
        agent = MasterAgent("/path/to/repo")
        await agent.run_interactive()
    
    Usage (enterprise - custom LLM client):
        from agentic.llm_client import LLMClient, Message, ChatResponse
        
        class MyLLM(LLMClient):
            async def chat(self, messages, **kwargs):
                # Your enterprise API call here
                return ChatResponse(content="...", model="...")
            async def close(self):
                pass
        
        agent = MasterAgent("/path/to/repo", llm_client=MyLLM())
    
    Usage (disable semantic search):
        from agentic.config import AgenticConfig
        
        config = AgenticConfig(disable_semantic_search=True)
        agent = MasterAgent("/path/to/repo", config=config)
    """
    
    def __init__(
        self,
        repo_path: str | Path,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        chroma_db_path: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        config: Optional[AgenticConfig] = None,
    ):
        """
        Initialize the Master Agent.
        
        Args:
            repo_path: Path to the repository to analyze
            api_key: OpenRouter API key (ignored if llm_client provided)
            model: Model to use (ignored if llm_client provided)
            chroma_db_path: Path to ChromaDB for semantic search
            llm_client: Custom LLM client (for enterprise integration)
            config: Configuration object (or loads from file/env)
        """
        self.repo_path = Path(repo_path).resolve()
        self.model = model
        self.chroma_db_path = chroma_db_path
        
        # Load or use provided config
        self.config = config or get_config(self.repo_path)
        
        # Initialize LLM client
        if llm_client:
            # Use provided custom client (enterprise mode)
            self.llm = llm_client
        else:
            # Use OpenRouter (default mode)
            effective_api_key = api_key or self.config.openrouter_api_key
            effective_model = model if model != DEFAULT_MODEL else self.config.openrouter_model
            self.llm = OpenRouterClient(api_key=effective_api_key, model=effective_model)
        
        # Initialize other components
        self.scanner = RepoScanner(self.repo_path)
        self.state = StateManager(self.repo_path)
        
        # Initialize debug logger
        self.debug = DebugLogger(self.repo_path / ".agentic")
        
        self.session: Optional[PlanningSession] = None
        
        # Code context discovered during analysis
        self._code_context: dict = {}
        
        # Token usage tracking
        self.token_usage = TokenUsage()
        self.context_limit = self.config.context_limit
        
        # Callbacks for UI integration
        self.on_thinking: Optional[Callable[[str], None]] = None
        self.on_question: Optional[Callable[[list[str]], None]] = None
        self.on_plan_ready: Optional[Callable[[str], None]] = None
    
    def _read_file_content(self, filepath: Path, max_lines: int = 200) -> Optional[str]:
        """Read file content, truncated if too long."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    content = ''.join(lines[:max_lines])
                    content += f"\n... (truncated, {len(lines) - max_lines} more lines)"
                else:
                    content = ''.join(lines)
                return content
        except Exception:
            return None
    
    def _grep_search(self, pattern: str, file_extensions: list[str] = None, context_lines: int = 0) -> list[dict]:
        """
        Search for pattern in codebase using grep-like search.
        
        Args:
            pattern: Regex pattern to search for
            file_extensions: List of extensions to search (default: common code files)
            context_lines: Number of lines to include before and after each match
        """
        results = []
        extensions = file_extensions or ['.py', '.js', '.ts', '.json', '.yaml', '.yml']
        
        for file_info in self.session.repo_structure.files if self.session else []:
            if file_info.extension not in extensions:
                continue
            
            try:
                with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    all_lines = f.readlines()
                    
                for line_num, line in enumerate(all_lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        if context_lines > 0:
                            # Include context lines
                            start_idx = max(0, line_num - 1 - context_lines)
                            end_idx = min(len(all_lines), line_num + context_lines)
                            context_content = ''.join(all_lines[start_idx:end_idx]).strip()
                            results.append({
                                'file': file_info.relative_path,
                                'line': line_num,
                                'content': context_content[:500],  # More room for context
                            })
                        else:
                            results.append({
                                'file': file_info.relative_path,
                                'line': line_num,
                                'content': line.strip()[:200],
                            })
                        if len(results) >= 50:  # Limit grep results
                            return results
            except Exception:
                continue
        
        return results
    
    def _semantic_search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search codebase using ChromaDB embeddings if available."""
        if not CHROMADB_AVAILABLE:
            return []
        
        db_path = self.chroma_db_path or str(self.repo_path / "chroma_db")
        if not Path(db_path).exists():
            return []
        
        try:
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection("codebase")
            
            # Use simple keyword search since we don't have embedder here
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas']
            )
            
            if not results['ids'][0]:
                return []
            
            return [
                {
                    'file': results['metadatas'][0][i].get('filepath', 'unknown'),
                    'content': results['documents'][0][i][:500],
                    'type': results['metadatas'][0][i].get('chunk_type', 'unknown'),
                }
                for i in range(len(results['ids'][0]))
            ]
        except Exception as e:
            return []
    
    def _analyze_codebase(self, user_prompt: str) -> dict:
        """
        Deep analysis of codebase relevant to user's request.
        Returns structured context for the LLM.
        """
        context = {
            'key_files': {},
            'grep_results': [],
            'semantic_results': [],
            'frameworks_detected': [],
            'patterns_found': [],
        }
        
        if not self.session or not self.session.repo_structure:
            return context
        
        rs = self.session.repo_structure
        
        # 1. Read all entry point files
        for ep in rs.entry_points[:10]:
            filepath = self.repo_path / ep
            if filepath.exists():
                content = self._read_file_content(filepath)
                if content:
                    context['key_files'][ep] = content
        
        # 2. Read config files (requirements.txt, package.json, etc.)
        config_files = ['requirements.txt', 'pyproject.toml', 'package.json', 
                       'setup.py', 'Dockerfile', 'docker-compose.yml', '.env.example']
        for cf in config_files:
            filepath = self.repo_path / cf
            if filepath.exists():
                content = self._read_file_content(filepath, max_lines=100)
                if content:
                    context['key_files'][cf] = content
        
        # 3. Detect frameworks from imports/dependencies
        framework_patterns = {
            'flask': r'from flask|import flask|Flask\(',
            'fastapi': r'from fastapi|import fastapi|FastAPI\(',
            'django': r'from django|import django|DJANGO_',
            'express': r'require\([\'"]express[\'"]\)|from [\'"]express[\'"]',
            'react': r'from [\'"]react[\'"]|import React',
            'pytest': r'import pytest|from pytest',
        }
        
        for framework, pattern in framework_patterns.items():
            results = self._grep_search(pattern)
            if results:
                context['frameworks_detected'].append(framework)
        
        # 4. Search for relevant code based on user prompt
        # Extract key terms from prompt
        keywords = re.findall(r'\b(api|endpoint|admin|user|auth|database|model|route|controller|service)\b', 
                             user_prompt.lower())
        
        for keyword in set(keywords):
            grep_results = self._grep_search(keyword)
            context['grep_results'].extend(grep_results[:5])
        
        # 5. Semantic search if available
        context['semantic_results'] = self._semantic_search(user_prompt)
        
        # 6. Look for existing patterns (routes, models, etc.)
        pattern_searches = {
            'routes': r'@app\.(get|post|put|delete|route)|router\.|@router\.',
            'models': r'class.*Model|class.*Schema|@dataclass',
            'tests': r'def test_|class Test',
        }
        
        for pattern_name, pattern in pattern_searches.items():
            results = self._grep_search(pattern)
            if results:
                context['patterns_found'].append({
                    'type': pattern_name,
                    'examples': results[:3]
                })
        
        return context
    
    async def start_session(
        self,
        user_prompt: str,
        discovered_context: Optional[DiscoveredContext] = None,
    ) -> str:
        """
        Start a new planning session.
        
        Args:
            user_prompt: The user's high-level request
            discovered_context: Optional pre-discovered context from Context Discovery Agent
            
        Returns:
            Session ID
        """
        # Initialize state
        session_id = self.state.initialize_session(user_prompt, self.model)
        
        # Scan repository
        repo_structure = self.scanner.scan()
        
        # Create session
        self.session = PlanningSession(
            user_prompt=user_prompt,
            repo_structure=repo_structure,
            discovered_context=discovered_context,
        )
        
        # No upfront analysis - let the LLM use tools to explore
        self._code_context = {}
        
        # Initialize debug logging for this session
        self.debug.start_session(user_prompt, self.repo_path.name)
        
        return session_id
    
    def _auto_explore(self, user_prompt: str):
        """
        Automatically explore the codebase before LLM interaction.
        This ensures the LLM has code context regardless of whether it uses tools.
        """
        explorations = []
        
        # Always search for key patterns
        patterns_to_search = [
            (r"@app\.(route|get|post|put|delete)|@router\.", "API routes"),
            (r"class.*Model|class.*Schema|@dataclass", "Models/Schemas"),
            (r"def.*user|User|user", "User-related code"),
            (r"def.*admin|Admin|admin", "Admin-related code"),
            (r"import|from.*import", "Imports (to detect frameworks)"),
        ]
        
        for pattern, desc in patterns_to_search:
            results = self._grep_search(pattern)
            if results:
                explorations.append(f"\n### {desc} (grep: {pattern}):")
                for r in results[:8]:
                    explorations.append(f"  {r['file']}:{r['line']}: {r['content'][:100]}")
        
        # Store exploration results
        exploration_text = "\n".join(explorations) if explorations else "No API patterns found - this appears to be a new project or non-API codebase"
        self._code_context['auto_exploration'] = exploration_text
        
        # Also print to console for debugging
        print(f"\n[Auto-exploration found {len(explorations)} lines of patterns]")
        if not explorations:
            print("[WARNING: No existing API routes, models, or user code found in codebase]")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt - LLM uses tools for exploration."""
        prompt = MASTER_AGENT_SYSTEM_PROMPT
        
        # If semantic search is disabled, remove it from the tool list
        if self.config.disable_semantic_search:
            # Remove the semantic_search tool section
            prompt = prompt.replace(
                '''3. **semantic_search** - Find code by meaning
   ```json
   {{"tool_calls": [{{"tool": "semantic_search", "query": "user authentication"}}]}}
   ```

''', '')
        
        return prompt
    
    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "grep":
                pattern = params.get("pattern", "")
                context_lines = params.get("context_lines", 0)
                results = self._grep_search(pattern, context_lines=context_lines)
                if not results:
                    return f"⚠️ NO MATCHES FOUND for pattern: {pattern}\n(This pattern does NOT exist in the codebase. Do not assume it exists elsewhere.)"
                output = [f"✓ Found {len(results)} matches for '{pattern}':"]
                for r in results[:20]:  # Show first 20 in output
                    output.append(f"  {r['file']}:{r['line']}: {r['content']}")
                return "\n".join(output)
            
            elif tool_name == "read_file":
                path = params.get("path", "")
                filepath = self.repo_path / path
                if not filepath.exists():
                    return f"⚠️ FILE NOT FOUND: {path}\n(This file does NOT exist. Do not reference it in your plan.)"
                content = self._read_file_content(filepath, max_lines=250)
                return f"✓ Contents of {path}:\n```\n{content}\n```"
            
            elif tool_name == "semantic_search":
                # Check if semantic search is disabled (enterprise mode)
                if self.config.disable_semantic_search:
                    return "⚠️ SEMANTIC SEARCH DISABLED: This tool is not available in the current configuration. Use grep or read_file instead."
                
                query = params.get("query", "")
                results = self._semantic_search(query)
                if not results:
                    return f"⚠️ NO SEMANTIC MATCHES for: {query}\n(Nothing in the codebase matches this concept. Do not assume it exists.)"
                output = [f"✓ Semantic search results for '{query}':"]
                for r in results:
                    output.append(f"\n### {r['file']} ({r['type']}):\n{r['content'][:400]}")
                return "\n".join(output)
            
            elif tool_name == "list_files":
                pattern = params.get("pattern", "*")
                if not self.session or not self.session.repo_structure:
                    return "No repository scanned"
                matching = []
                import fnmatch
                for f in self.session.repo_structure.files:
                    if fnmatch.fnmatch(f.relative_path, pattern) or fnmatch.fnmatch(Path(f.relative_path).name, pattern):
                        matching.append(f.relative_path)
                if not matching:
                    return f"⚠️ NO FILES FOUND matching: {pattern}\n(No files with this pattern exist in the codebase.)"
                return f"✓ Files matching '{pattern}':\n" + "\n".join(f"  - {m}" for m in matching[:10])
            
            elif tool_name == "run_command":
                from .tools import run_command as exec_command
                cmd = params.get("command", "")
                timeout = params.get("timeout", 30)
                result = exec_command(self.repo_path, cmd, timeout=timeout)
                status = "✓" if result.success else "✗"
                output = f"{status} Command: {cmd}\nExit code: {result.exit_code}\n"
                if result.stdout:
                    output += f"\nSTDOUT:\n{result.stdout[:2000]}"
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr[:1000]}"
                return output
            
            elif tool_name == "get_lints":
                from .tools import read_lints
                file_path = params.get("file", params.get("path", ""))
                result = read_lints(self.repo_path, file_path)
                if not result.success:
                    return f"⚠️ Linting failed: {result.error_message}"
                if not result.diagnostics:
                    return f"✓ No lint errors in {file_path} (linter: {result.linter_used})"
                output = [f"Found {len(result.diagnostics)} issues in {file_path} (linter: {result.linter_used}):"]
                for d in result.diagnostics[:20]:
                    output.append(f"  {d.severity.upper()} L{d.line}: {d.message}" + (f" [{d.rule}]" if d.rule else ""))
                return "\n".join(output)
            
            elif tool_name == "get_symbols":
                from .tools import get_file_symbols
                file_path = params.get("file", params.get("path", ""))
                symbols = get_file_symbols(self.repo_path, file_path)
                if not symbols:
                    return f"⚠️ No symbols found in {file_path} (file may not exist or language not supported)"
                output = [f"✓ Symbols in {file_path}:"]
                for s in symbols[:30]:
                    output.append(f"  L{s.line}: {s.type} {s.name}")
                return "\n".join(output)
            
            elif tool_name == "find_references":
                from .tools import find_references
                symbol = params.get("symbol", "")
                refs = find_references(self.repo_path, symbol, max_results=30)
                if not refs:
                    return f"⚠️ No references found for: {symbol}"
                output = [f"✓ Found {len(refs)} references to '{symbol}':"]
                for r in refs[:20]:
                    output.append(f"  {r.file}:{r.line}: {r.context[:80]}")
                return "\n".join(output)
            
            else:
                return f"Unknown tool: {tool_name}"
        
        except Exception as e:
            return f"Tool error ({tool_name}): {str(e)}"
    
    def _execute_tool_calls(self, tool_calls: list[dict]) -> str:
        """Execute multiple tool calls and return combined results."""
        results = []
        for call in tool_calls[:5]:  # Limit to 5 tool calls
            tool_name = call.get("tool", "")
            result = self._execute_tool(tool_name, call)
            results.append(f"## Tool: {tool_name}\n{result}")
        return "\n\n".join(results)
    
    def _log_llm_output(self, content: str, log: Callable[[str], None]) -> None:
        """Log the LLM's thinking and response to terminal (truncated for readability)."""
        import re
        
        # Check for thinking blocks (Claude extended thinking)
        thinking_match = re.search(r'<thinking>([\s\S]*?)</thinking>', content)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            # Truncate thinking to first 300 chars
            if len(thinking) > 300:
                thinking = thinking[:300] + "..."
            log(f"💭 Thinking: {thinking}")
        
        # Get the non-thinking part of the response
        response_text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', content).strip()
        
        # Remove tool call blocks for display (we show those separately)
        response_text = re.sub(r'<function_calls>[\s\S]*?</function_calls>', '', response_text)
        response_text = re.sub(r'```json[\s\S]*?```', '', response_text)
        response_text = response_text.strip()
        
        # Show first 300 chars of actual response
        if response_text:
            if len(response_text) > 300:
                response_text = response_text[:300] + "..."
            log(f"💬 Response: {response_text}")
    
    def _build_messages(self) -> list[Message]:
        """Build the message list for the LLM."""
        messages = [Message("system", self._build_system_prompt())]
        
        if self.session:
            # Build the initial user message with optional discovered context
            user_message_parts = [self.session.user_prompt]
            
            # If we have pre-discovered context, include it
            if self.session.discovered_context:
                context_text = self.session.discovered_context.to_prompt_context()
                if context_text.strip():
                    user_message_parts.append(f"""

---
## Pre-Discovered Codebase Context (ALREADY VERIFIED - TRUST THIS!)

The Context Discovery Agent has ALREADY explored the codebase and verified this information. 
**You do NOT need to re-verify this with tools.** 

ONLY use tools if:
1. The pre-discovered context is missing something specific you need
2. You need to see the EXACT implementation details (not just patterns)

If the context below shows file paths, patterns, and examples - proceed directly to planning!

{context_text}
---
""")
            
            messages.append(Message("user", "\n".join(user_message_parts)))
            
            # Add conversation history
            for turn in self.session.conversation:
                messages.append(Message(turn.role, turn.content))
        
        return messages
    
    def _parse_llm_response(self, content: str) -> dict:
        """
        Parse the LLM's JSON response.
        
        IMPORTANT: Only takes the FIRST tool_calls block found.
        This prevents the LLM from batching many speculative tool calls before seeing any results.
        The LLM should explore incrementally: make a few calls, see results, then decide next steps.
        
        Handles multiple formats:
        1. <function_calls><invoke name="...">...</invoke></function_calls> - Anthropic-style XML
        2. <tool_calls>[...]</tool_calls> - Simple JSON in XML
        3. ```json {"tool_calls": [...]} ``` code blocks
        4. ```json {"ready_to_plan": true, ...} ``` final response
        """
        import re
        
        # Only take the FIRST tool_calls block - don't batch speculative calls!
        first_tool_calls = None
        plan_result = None
        questions_result = None
        
        # 1. Check for Anthropic-style <function_calls><invoke> XML format
        # This is what Claude often uses: <function_calls><invoke name="grep"><parameter name="pattern">...</parameter></invoke></function_calls>
        function_calls_blocks = re.findall(r'<function_calls>\s*([\s\S]*?)\s*</function_calls>', content)
        for block in function_calls_blocks:
            if first_tool_calls is not None:
                break  # Only take first block
            # Parse each <invoke> within the block
            invokes = re.findall(r'<invoke\s+name="([^"]+)"[^>]*>([\s\S]*?)</invoke>', block)
            if invokes:
                first_tool_calls = []
                for tool_name, params_block in invokes:
                    tool_call = {"tool": tool_name}
                    # Parse parameters
                    params = re.findall(r'<parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)</parameter>', params_block)
                    for param_name, param_value in params:
                        tool_call[param_name] = param_value.strip()
                    first_tool_calls.append(tool_call)
        
        # 2. Check for <tool_calls> JSON format
        xml_tool_calls = re.findall(r'<tool_calls>\s*([\s\S]*?)\s*</tool_calls>', content)
        for xml_block in xml_tool_calls:
            if first_tool_calls is not None:
                break  # Only take first block
            try:
                parsed = json.loads(xml_block.strip())
                if isinstance(parsed, list):
                    first_tool_calls = parsed
                elif isinstance(parsed, dict) and parsed.get("tool"):
                    first_tool_calls = [parsed]
            except json.JSONDecodeError:
                continue
        
        # 2. Find ALL ```json ... ``` blocks
        json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', content)
        
        # Also find ``` ... ``` blocks that look like JSON
        code_blocks = re.findall(r'```\s*([\s\S]*?)\s*```', content)
        for block in code_blocks:
            block = block.strip()
            if block.startswith('{') and block not in json_blocks:
                json_blocks.append(block)
        
        for block in json_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    # Only take the FIRST tool_calls block we find
                    if parsed.get("tool_calls") and first_tool_calls is None:
                        calls = parsed["tool_calls"]
                        if isinstance(calls, list):
                            first_tool_calls = calls
                    
                    # Track plan (but don't prioritize it over tools)
                    if parsed.get("ready_to_plan") and "plan" in parsed:
                        plan_result = parsed
                    
                    # Track questions
                    if parsed.get("questions"):
                        questions_result = parsed
                        
            except json.JSONDecodeError:
                continue
        
        # PRIORITY: Return tool_calls if found (even if plan also present)
        # This ensures the LLM actually reads the codebase before we accept a plan
        if first_tool_calls:
            original_count = len(first_tool_calls)
            was_limited = False
            
            # Limit to 5 tool calls per round to force incremental exploration
            if original_count > 5:
                print(f"[WARNING] LLM requested {original_count} tool calls - limiting to first 5")
                first_tool_calls = first_tool_calls[:5]
                was_limited = True
            
            result = {
                "tool_calls": first_tool_calls,
                "_tool_call_metadata": {
                    "was_limited": was_limited,
                    "original_count": original_count,
                }
            }
            # Also include plan/questions so the caller knows they were present
            if plan_result:
                result["_pending_plan"] = plan_result  # Mark as pending, not ready
            return result
        
        # Only return plan if there were NO tool calls
        if plan_result:
            return plan_result
        
        # Return questions if present
        if questions_result:
            return questions_result
        
        # Fallback: try single JSON extraction strategies
        json_str = None
        
        # Strategy 1: Find first ```json ... ``` block
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
        
        # Strategy 3: Find raw JSON object with tool_calls or ready_to_plan
        if not json_str:
            json_match = re.search(r'\{[\s\S]*"(ready_to_plan|tool_calls)"[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
        
        # Strategy 4: Try the whole content
        if not json_str:
            json_str = content.strip()
        
        # Attempt to parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Return as conversation if not valid JSON
            return {
                "ready_to_plan": False,
                "message": content,
                "parse_error": str(e),
            }
    
    async def think(self, max_tool_rounds: int = 20, on_status: Optional[Callable[[str], None]] = None) -> dict:
        """
        Send context to LLM and get response.
        Handles tool calls automatically, executing them and continuing.
        
        Returns:
            Parsed response dict with either questions or plan
        """
        if not self.session:
            raise ValueError("No active session")
        
        def log(msg: str):
            if on_status:
                on_status(msg)
        
        for tool_round in range(max_tool_rounds):
            messages = self._build_messages()
            
            # Log round start with full context
            self.debug.log_round_start(tool_round + 1, messages)
            log(f"[Round {tool_round + 1}] Sending {len(messages)} messages to LLM...")
            log(f"[Debug logs: .agentic/debug_log.txt, .agentic/debug_context.txt]")
            
            # Get LLM response
            response = await self.llm.chat(messages, max_tokens=4096)
            
            # Track token usage
            self.token_usage = self.token_usage.add(response.token_usage)
            log(f"[Tokens] {self.token_usage.format(self.context_limit)}")
            
            # Store in conversation
            self.session.conversation.append(
                ConversationTurn("assistant", response.content)
            )
            
            # Parse response
            parsed = self._parse_llm_response(response.content)
            
            # Log LLM response
            self.debug.log_llm_response(response.content, list(parsed.keys()))
            log(f"[LLM responded] Parsed keys: {list(parsed.keys())}")
            
            # Show LLM's thinking/response in terminal (truncated for readability)
            self._log_llm_output(response.content, log)
            
            # PRIORITY: Execute tools FIRST before accepting any plan
            # This prevents the LLM from hallucinating a plan without actually reading the codebase
            if parsed.get("tool_calls"):
                tool_calls = parsed["tool_calls"]
                if isinstance(tool_calls, list) and tool_calls:
                    # Check if tool calls were limited
                    metadata = parsed.get("_tool_call_metadata", {})
                    was_limited = metadata.get("was_limited", False)
                    original_count = metadata.get("original_count", len(tool_calls))
                    
                    self.debug.log_tool_calls(tool_calls, was_limited=was_limited, original_count=original_count)
                    
                    if was_limited:
                        log(f"[TOOLS] ⚠️ LLM requested {original_count} tool calls - limited to {len(tool_calls)}")
                    
                    log(f"[TOOLS] Executing {len(tool_calls)} tool(s)...")
                    for tc in tool_calls:
                        log(f"  → {tc.get('tool', 'unknown')}: {tc}")
                    
                    # Execute the tools
                    tool_results = self._execute_tool_calls(tool_calls)
                    
                    # Log tool results
                    self.debug.log_tool_results(tool_results)
                    log(f"[TOOLS] Results received ({len(tool_results)} chars)")
                    
                    # Add tool results to conversation
                    self.session.conversation.append(
                        ConversationTurn("user", f"Tool results:\n\n{tool_results}")
                    )
                    
                    # If LLM also included a plan, log a warning - it should wait for tool results
                    if parsed.get("_pending_plan"):
                        log(f"[WARNING] LLM provided plan before seeing tool results - ignoring plan, executing tools first")
                        self.debug.log_error("LLM tried to provide plan before seeing tool results - forcing tool execution first")
                    
                    # Continue to next round to let LLM process results
                    continue
            
            # Only accept a plan if there were NO tool calls (LLM has finished exploring)
            if parsed.get("ready_to_plan") and "plan" in parsed:
                self.session.plan_data = parsed["plan"]
                self.debug.log_plan_detected(parsed["plan"])
                log("[Plan detected! Returning to planning loop.]")
                return parsed
            
            # Check if LLM has questions
            if parsed.get("questions"):
                self.debug.log_questions(parsed["questions"])
            
            # No tool calls and no plan - return whatever we have
            return parsed
        
        # Max rounds reached
        self.debug.log_error("Max tool call rounds reached")
        return {
            "ready_to_plan": False,
            "message": "Max tool call rounds reached. Please provide more direction.",
        }
    
    async def answer_questions(self, answers: str):
        """
        Provide answers to the LLM's questions.
        
        Args:
            answers: User's answers as free-form text
        """
        if not self.session:
            raise ValueError("No active session")
        
        # Log the user's answer
        self.debug.log_user_answer(answers)
        
        self.session.conversation.append(
            ConversationTurn("user", answers)
        )
    
    async def generate_plan(self) -> str:
        """
        Generate the plan.md content from collected data.
        
        Returns:
            Markdown content for plan.md
        """
        if not self.session or not self.session.plan_data:
            raise ValueError("No plan data available")
        
        pd = self.session.plan_data
        
        # Use plan generator
        plan_content = create_plan(
            goal=pd.get("goal", ""),
            tasks=pd.get("tasks", []),
            non_goals=pd.get("non_goals", []),
            assumptions=pd.get("assumptions", []),
            constraints=pd.get("constraints", []),
            success_criteria=pd.get("success_criteria", []),
            test_strategy=pd.get("test_strategy"),
            escalation_threshold=3,
        )
        
        return plan_content
    
    async def save_plan(self, content: str):
        """Save the plan to disk."""
        self.state.save_plan(content)
    
    async def approve_and_lock(self) -> str:
        """
        Approve and lock the plan.
        
        Returns:
            The plan hash for verification
        """
        self.state.approve_plan()
        return self.state.lock_plan()
    
    async def run_planning_loop(
        self,
        get_user_input: Callable[[str], str],
        display_message: Callable[[str], None],
        display_questions: Callable[[list[str]], None],
        display_plan: Callable[[str], None],
        confirm_plan: Callable[[], bool],
    ) -> bool:
        """
        Run the complete planning loop.
        
        Args:
            get_user_input: Function to get user input (prompt text -> response)
            display_message: Function to display a message to user
            display_questions: Function to display questions
            display_plan: Function to display the generated plan
            confirm_plan: Function to confirm plan approval (returns bool)
            
        Returns:
            True if plan was approved and locked, False otherwise
        """
        if not self.session:
            raise ValueError("No active session. Call start_session first.")
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            display_message(f"\n[Thinking... (iteration {iteration})]")
            
            try:
                result = await self.think(on_status=display_message)
            except Exception as e:
                import traceback
                display_message(f"\nError communicating with LLM: {e}")
                display_message(f"\nTraceback: {traceback.format_exc()}")
                return False
            
            if result.get("ready_to_plan") and "plan" in result:
                # Plan is ready
                display_message("\n[Plan generated!]")
                
                plan_content = await self.generate_plan()
                display_plan(plan_content)
                
                if confirm_plan():
                    # Save and lock
                    await self.save_plan(plan_content)
                    plan_hash = await self.approve_and_lock()
                    display_message(f"\nPlan saved and locked!")
                    display_message(f"Hash: {plan_hash}")
                    display_message(f"Location: {self.state.plan_path}")
                    self.session.is_complete = True
                    return True
                else:
                    # User wants changes
                    feedback = get_user_input("What changes would you like?")
                    await self.answer_questions(feedback)
            
            elif "questions" in result:
                # LLM has questions
                questions = result["questions"]
                display_questions(questions)
                
                answers = get_user_input("Your answers")
                await self.answer_questions(answers)
            
            elif "message" in result:
                # Free-form response (couldn't parse as JSON)
                display_message(f"\nAgent: {result['message']}")
                
                user_response = get_user_input("Your response")
                await self.answer_questions(user_response)
            
            else:
                display_message("\nUnexpected response format. Continuing...")
                display_message(str(result))
        
        display_message("\nMax iterations reached without completing plan.")
        return False
    
    async def close(self):
        """Clean up resources."""
        await self.llm.close()
    
    async def run_execution(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        max_attempts: int = 3,
    ):
        """
        Run Phase 2 execution of the locked plan.
        
        Args:
            on_status: Callback for status updates
            max_attempts: Maximum retries per task
            
        Returns:
            ExecutionSummary with results
        """
        from .execution_engine import run_execution
        
        return await run_execution(
            repo_path=self.repo_path,
            api_key=self.llm.api_key,
            model=self.model,
            max_attempts=max_attempts,
            on_status=on_status,
        )


async def quick_plan(
    repo_path: str,
    prompt: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Quick function to generate a plan non-interactively.
    
    For testing and simple use cases.
    
    Args:
        repo_path: Path to repository
        prompt: User prompt
        api_key: OpenRouter API key
        
    Returns:
        Plan content if successful, None otherwise
    """
    agent = MasterAgent(repo_path, api_key=api_key)
    
    try:
        await agent.start_session(prompt)
        result = await agent.think()
        
        if result.get("ready_to_plan"):
            plan_content = await agent.generate_plan()
            await agent.save_plan(plan_content)
            return plan_content
        else:
            print("Plan not ready, questions remain:")
            print(result.get("questions", result))
            return None
    finally:
        await agent.close()


if __name__ == "__main__":
    import sys
    
    async def main():
        repo = sys.argv[1] if len(sys.argv) > 1 else "."
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Analyze this repository"
        
        print(f"Creating plan for: {repo}")
        print(f"Prompt: {prompt}")
        
        plan = await quick_plan(repo, prompt)
        if plan:
            print("\n" + "="*60)
            print(plan)
    
    asyncio.run(main())

