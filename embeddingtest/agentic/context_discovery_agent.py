"""
Context Discovery Agent

Pre-planning agent that explores the codebase to find relevant context
for the user's prompt. Runs before the Master Agent to provide curated,
filtered context so the Master Agent can focus purely on planning.

Workflow:
1. Initial exploration (list_files, grep for patterns related to prompt)
2. Deep dive into discovered files via read_file
3. Semantic search for conceptual matches
4. LLM filters/summarizes to only what's actually relevant
5. Returns structured DiscoveredContext for Master Agent
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from .openrouter_client import OpenRouterClient, Message, TokenUsage, DEFAULT_MODEL
from .repo_scanner import RepoScanner, RepoStructure


class DiscoveryDebugLogger:
    """
    Debug logger for the Context Discovery Agent.
    Creates detailed logs of all exploration activity.
    
    Log files:
    - discovery_debug_log.txt: Incremental log of all activity
    - discovery_context.txt: Full context snapshot (what we're sending to LLM)
    - discovery_handoff.txt: What gets passed to Master Agent
    """
    
    def __init__(self, agentic_dir: Path):
        self.agentic_dir = agentic_dir
        self.log_file = agentic_dir / "discovery_debug_log.txt"
        self.context_file = agentic_dir / "discovery_context.txt"
        self.handoff_file = agentic_dir / "discovery_handoff.txt"
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
    
    def _write_handoff(self, content: str):
        """Write the handoff context (what goes to Master Agent)."""
        with open(self.handoff_file, "w", encoding="utf-8") as f:
            f.write(content)
    
    def start_session(self, user_prompt: str, repo_name: str):
        """Log the start of a discovery session."""
        header = f"""
{'#' * 80}
#  CONTEXT DISCOVERY AGENT DEBUG LOG
#  Started: {self._timestamp()}
#  Repository: {repo_name}
{'#' * 80}

================================================================================
USER PROMPT (What we're finding context for)
================================================================================
{user_prompt}

"""
        self._write_log(header, mode="w")
    
    def log_repo_scan(self, total_files: int, total_dirs: int, entry_points: list[str], file_tree: str):
        """Log repository scan results."""
        log_entry = f"""
================================================================================
REPOSITORY SCAN
Time: {self._timestamp()}
================================================================================

Total Files: {total_files}
Total Directories: {total_dirs}
Entry Points: {', '.join(entry_points[:5]) if entry_points else 'None detected'}

File Tree (first 2000 chars):
{file_tree[:2000]}
{'...(truncated)' if len(file_tree) > 2000 else ''}

"""
        self._write_log(log_entry)
    
    def log_round_start(self, round_num: int, messages: list[Message]):
        """Log the start of an exploration round with full context."""
        self.round_num = round_num
        
        # Build full context snapshot
        context_lines = [
            f"CONTEXT DISCOVERY AGENT - FULL CONTEXT SNAPSHOT",
            f"Generated: {self._timestamp()}",
            f"Round: {round_num}",
            f"Total Messages: {len(messages)}",
            "=" * 80,
            "",
            "This file shows the COMPLETE context sent to the LLM this round.",
            ""
        ]
        
        for i, msg in enumerate(messages):
            context_lines.append(f"\n{'─' * 80}")
            context_lines.append(f"MESSAGE {i} | Role: {msg.role.upper()}")
            context_lines.append(f"{'─' * 80}")
            context_lines.append(msg.content)
        
        self._write_context("\n".join(context_lines))
        
        # Log to incremental log
        log_entry = f"""
================================================================================
ROUND {round_num} - SENDING TO LLM
Time: {self._timestamp()}
Total Messages: {len(messages)}
================================================================================
(Full context in discovery_context.txt)

"""
        self._write_log(log_entry)
    
    def log_llm_response(self, response_content: str, parsed_keys: list):
        """Log the LLM's response."""
        # Detect format used
        has_function_calls = '<function_calls>' in response_content
        has_invoke = '<invoke' in response_content
        has_xml_tools = '<tool_calls>' in response_content
        has_json_tools = '"tool_calls"' in response_content
        has_exploration_complete = '"exploration_complete"' in response_content
        
        format_notes = []
        if has_function_calls and has_invoke:
            format_notes.append("Anthropic-style <function_calls><invoke> detected")
        if has_xml_tools:
            format_notes.append("XML <tool_calls> detected")
        if has_json_tools:
            format_notes.append("JSON tool_calls detected")
        if has_exploration_complete:
            format_notes.append("exploration_complete detected")
        
        log_entry = f"""
--------------------------------------------------------------------------------
ROUND {self.round_num} - LLM RESPONSE
Time: {self._timestamp()}
Parsed keys: {parsed_keys}
Format notes: {', '.join(format_notes) if format_notes else 'None'}
--------------------------------------------------------------------------------

{response_content}

"""
        self._write_log(log_entry)
    
    def log_tool_calls(self, tool_calls: list[dict], was_limited: bool = False, original_count: int = 0):
        """Log tool calls being executed."""
        warning = ""
        if was_limited:
            warning = f"""
⚠️ WARNING: LLM requested {original_count} tool calls but was limited to {len(tool_calls)}!
   The LLM should explore incrementally, not batch speculative calls.
   
"""
        
        log_entry = f"""
--------------------------------------------------------------------------------
TOOL CALLS (Round {self.round_num})
Time: {self._timestamp()}
Count: {len(tool_calls)}{f' (limited from {original_count})' if was_limited else ''}
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
    
    def log_discovery_complete(self, context: 'DiscoveredContext'):
        """Log when discovery is complete and what we found."""
        summary = f"""
================================================================================
✓ DISCOVERY COMPLETE
Time: {self._timestamp()}
================================================================================

Exploration Stats:
  - Rounds: {context.exploration_rounds}
  - Files explored: {context.files_explored}

Frameworks Detected: {', '.join(context.frameworks_detected) if context.frameworks_detected else 'None'}
Entry Points: {', '.join(context.entry_points) if context.entry_points else 'None'}
Relevant Files: {len(context.relevant_files)}
Patterns Found: {len(context.patterns)}
Test Files: {len(context.test_files)}
Config Files: {len(context.config_files)}

"""
        self._write_log(summary)
        
        # Detailed breakdown
        if context.relevant_files:
            self._write_log("Relevant Files Found:\n")
            for f in context.relevant_files:
                self._write_log(f"  - {f.path}\n")
                self._write_log(f"    Summary: {f.summary[:100]}...\n" if len(f.summary) > 100 else f"    Summary: {f.summary}\n")
                if f.symbols:
                    self._write_log(f"    Symbols: {', '.join(f.symbols[:5])}\n")
        
        if context.patterns:
            self._write_log("\nPatterns Found:\n")
            for p in context.patterns:
                self._write_log(f"  - {p.pattern_type}: {p.description[:80]}...\n" if len(p.description) > 80 else f"  - {p.pattern_type}: {p.description}\n")
    
    def log_handoff_to_master(self, prompt_context: str):
        """Log exactly what we're passing to the Master Agent."""
        handoff_content = f"""
{'#' * 80}
#  CONTEXT DISCOVERY → MASTER AGENT HANDOFF
#  Generated: {self._timestamp()}
{'#' * 80}

This is EXACTLY what gets injected into the Master Agent's context.

================================================================================
HANDOFF CONTEXT
================================================================================

{prompt_context}
"""
        self._write_handoff(handoff_content)
        
        # Also log to main log
        log_entry = f"""
================================================================================
HANDOFF TO MASTER AGENT
Time: {self._timestamp()}
Context Length: {len(prompt_context)} chars
================================================================================
(Full handoff in discovery_handoff.txt)

Preview (first 1000 chars):
{prompt_context[:1000]}
{'...(truncated)' if len(prompt_context) > 1000 else ''}

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

# Try to import ChromaDB for semantic search
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@dataclass
class CodeSnippet:
    """A relevant code snippet from the codebase."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    relevance: str  # Why this is relevant


@dataclass
class DiscoveredFile:
    """A relevant file discovered in the codebase."""
    path: str
    summary: str  # What this file does / why it's relevant
    key_snippets: list[CodeSnippet] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)  # Key functions/classes


@dataclass
class DiscoveredPattern:
    """A code pattern found in the codebase."""
    pattern_type: str  # e.g., "api_routes", "models", "tests"
    description: str
    examples: list[dict] = field(default_factory=list)  # {file, line, content}


@dataclass
class DiscoveredContext:
    """
    Complete discovered context from codebase exploration.
    
    This is what gets passed to the Master Agent for planning.
    """
    # Core discovered information
    relevant_files: list[DiscoveredFile] = field(default_factory=list)
    patterns: list[DiscoveredPattern] = field(default_factory=list)
    frameworks_detected: list[str] = field(default_factory=list)
    
    # Additional context
    entry_points: list[str] = field(default_factory=list)
    config_files: dict[str, str] = field(default_factory=dict)  # {path: summary}
    test_files: list[str] = field(default_factory=list)
    
    # Exploration metadata
    files_explored: int = 0
    exploration_rounds: int = 0
    
    def to_prompt_context(self) -> str:
        """Convert to a formatted string for the Master Agent's context."""
        sections = []
        
        # Frameworks
        if self.frameworks_detected:
            sections.append(f"## Detected Frameworks/Technologies\n{', '.join(self.frameworks_detected)}")
        
        # Entry points
        if self.entry_points:
            sections.append(f"## Entry Points\n" + "\n".join(f"- {ep}" for ep in self.entry_points[:5]))
        
        # Patterns
        if self.patterns:
            pattern_text = "## Code Patterns Found\n"
            for p in self.patterns:
                pattern_text += f"\n### {p.pattern_type}\n{p.description}\n"
                for ex in p.examples[:3]:
                    pattern_text += f"  - `{ex.get('file', 'unknown')}:{ex.get('line', 0)}`: {ex.get('content', '')[:100]}\n"
            sections.append(pattern_text)
        
        # Relevant files (the meat of the context)
        if self.relevant_files:
            files_text = "## Relevant Files\n"
            for f in self.relevant_files:
                files_text += f"\n### {f.path}\n{f.summary}\n"
                if f.symbols:
                    files_text += f"Key symbols: {', '.join(f.symbols[:10])}\n"
                for snippet in f.key_snippets[:2]:  # Limit snippets per file
                    files_text += f"\n```\n{snippet.content[:500]}\n```\n"
                    files_text += f"*{snippet.relevance}*\n"
            sections.append(files_text)
        
        # Config files
        if self.config_files:
            config_text = "## Configuration Files\n"
            for path, summary in list(self.config_files.items())[:5]:
                config_text += f"- **{path}**: {summary}\n"
            sections.append(config_text)
        
        # Test files
        if self.test_files:
            sections.append(f"## Related Test Files\n" + "\n".join(f"- {tf}" for tf in self.test_files[:5]))
        
        return "\n\n".join(sections)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "relevant_files": [
                {
                    "path": f.path,
                    "summary": f.summary,
                    "symbols": f.symbols,
                    "snippets": [
                        {"file": s.file_path, "start": s.start_line, "end": s.end_line, 
                         "content": s.content, "relevance": s.relevance}
                        for s in f.key_snippets
                    ]
                }
                for f in self.relevant_files
            ],
            "patterns": [
                {"type": p.pattern_type, "description": p.description, "examples": p.examples}
                for p in self.patterns
            ],
            "frameworks": self.frameworks_detected,
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "test_files": self.test_files,
            "metadata": {
                "files_explored": self.files_explored,
                "exploration_rounds": self.exploration_rounds,
            }
        }


# System prompt for Context Discovery Agent
CONTEXT_DISCOVERY_SYSTEM_PROMPT = '''You are the Context Discovery Agent. Your job is to INTERACTIVELY explore a codebase.

## ⚠️ CRITICAL: How Tool Calling Works

This is an INTERACTIVE conversation. When you output tool calls:
1. Your message ENDS after the tool calls
2. The tools are EXECUTED by the system  
3. You receive the ACTUAL RESULTS in the next message
4. THEN you decide what to do next

**NEVER output `exploration_complete` in the same message as tool calls!**
**NEVER guess or assume what tool results will be - WAIT for actual results!**

Each response must be ONE of:
- Tool calls ONLY (then stop and wait for results)
- Final summary with `exploration_complete: true` (ONLY after you've seen tool results)

## Your Goal

Find code relevant to the user's request:
1. Files that will need to be modified or referenced
2. Existing patterns the implementation should follow  
3. Related code that the user should be aware of

## Available Tools

```json
{{"tool_calls": [{{"tool": "grep", "pattern": "def user|class User"}}]}}
```

```json
{{"tool_calls": [{{"tool": "read_file", "path": "src/models/user.py"}}]}}
```

```json
{{"tool_calls": [{{"tool": "list_files", "pattern": "*.py"}}]}}
```

```json  
{{"tool_calls": [{{"tool": "get_symbols", "file": "src/main.py"}}]}}
```

```json
{{"tool_calls": [{{"tool": "semantic_search", "query": "user authentication"}}]}}
```

## Exploration Strategy

**Round 1**: Start broad - grep for keywords, list files
**Round 2-3**: Read the most promising files from Round 1 results  
**Round 4-5**: Follow up on specific patterns found
**Round 6+**: Complete when you have enough context

**MAXIMUM 3-5 tool calls per response** - then STOP and wait for results!

## Final Output (ONLY after seeing actual tool results)

When you have gathered enough context from ACTUAL tool results:

```json
{{
  "exploration_complete": true,
  "discovered_context": {{
    "frameworks": ["fastapi", "sqlalchemy"],
    "relevant_files": [
      {{
        "path": "src/api/users.py",
        "summary": "User API endpoints - follow this pattern",
        "symbols": ["get_users", "create_user"],
        "key_code": "Router pattern with auth dependency"
      }}
    ],
    "patterns": [
      {{
        "type": "api_routes",
        "description": "Routes use /api/v1 prefix with require_auth",
        "examples": [{{"file": "src/api/users.py", "line": 15, "content": "@router.get..."}}]
      }}
    ],
    "entry_points": ["src/main.py"],
    "config_files": {{}},
    "test_files": []
  }}
}}
```

## CRITICAL RULES

1. **WAIT FOR RESULTS** - Never assume what tools will return
2. **ONE THING AT A TIME** - Either tool calls OR final summary, never both
3. **VERIFY BEFORE INCLUDING** - Only report what you actually saw in results
4. **NO HALLUCINATION** - If you haven't read a file, don't describe its contents
5. **COMPLETE IN 5-10 ROUNDS** - Don't over-explore
'''


class ContextDiscoveryAgent:
    """
    Context Discovery Agent for Phase 0.
    
    Explores the codebase before planning to find relevant context.
    This keeps the Master Agent's context clean and focused.
    
    Debug logs are written to .agentic/ directory:
    - discovery_debug_log.txt: Full exploration log
    - discovery_context.txt: Current LLM context snapshot
    - discovery_handoff.txt: What gets passed to Master Agent
    """
    
    def __init__(
        self,
        repo_path: str | Path,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        chroma_db_path: Optional[str] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.model = model
        self.chroma_db_path = chroma_db_path
        
        # Initialize components
        self.llm = OpenRouterClient(api_key=api_key, model=model)
        self.scanner = RepoScanner(self.repo_path)
        
        # Initialize debug logger
        self.debug = DiscoveryDebugLogger(self.repo_path / ".agentic")
        
        # State
        self.repo_structure: Optional[RepoStructure] = None
        self.conversation: list[Message] = []
        
        # Token tracking
        self.token_usage = TokenUsage()
    
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
    
    def _grep_search(self, pattern: str, file_extensions: list[str] = None) -> list[dict]:
        """Search for pattern in codebase."""
        results = []
        extensions = file_extensions or ['.py', '.js', '.ts', '.tsx', '.json', '.yaml', '.yml', '.go', '.rs', '.java']
        
        if not self.repo_structure:
            return results
        
        for file_info in self.repo_structure.files:
            if file_info.extension not in extensions:
                continue
            
            try:
                with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            results.append({
                                'file': file_info.relative_path,
                                'line': line_num,
                                'content': line.strip()[:200],
                            })
                            if len(results) >= 50:
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
        except Exception:
            return []
    
    def _get_file_symbols(self, file_path: str) -> list[dict]:
        """Get symbols from a file using the tools module."""
        from .tools import get_file_symbols
        symbols = get_file_symbols(self.repo_path, file_path)
        return [{"name": s.name, "type": s.type, "line": s.line} for s in symbols]
    
    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "grep":
                pattern = params.get("pattern", "")
                results = self._grep_search(pattern)
                if not results:
                    return f"⚠️ NO MATCHES for pattern: {pattern}"
                output = [f"✓ Found {len(results)} matches for '{pattern}':"]
                for r in results[:25]:
                    output.append(f"  {r['file']}:{r['line']}: {r['content']}")
                return "\n".join(output)
            
            elif tool_name == "read_file":
                path = params.get("path", "")
                filepath = self.repo_path / path
                if not filepath.exists():
                    return f"⚠️ FILE NOT FOUND: {path}"
                content = self._read_file_content(filepath, max_lines=200)
                return f"✓ Contents of {path}:\n```\n{content}\n```"
            
            elif tool_name == "semantic_search":
                query = params.get("query", "")
                results = self._semantic_search(query)
                if not results:
                    return f"⚠️ NO SEMANTIC MATCHES for: {query}"
                output = [f"✓ Semantic search results for '{query}':"]
                for r in results:
                    output.append(f"\n### {r['file']} ({r['type']}):\n{r['content'][:400]}")
                return "\n".join(output)
            
            elif tool_name == "list_files":
                pattern = params.get("pattern", "*")
                if not self.repo_structure:
                    return "No repository scanned"
                matching = []
                import fnmatch
                for f in self.repo_structure.files:
                    if fnmatch.fnmatch(f.relative_path, pattern) or fnmatch.fnmatch(Path(f.relative_path).name, pattern):
                        matching.append(f.relative_path)
                if not matching:
                    return f"⚠️ NO FILES matching: {pattern}"
                return f"✓ Files matching '{pattern}':\n" + "\n".join(f"  - {m}" for m in matching[:15])
            
            elif tool_name == "get_symbols":
                file_path = params.get("file", params.get("path", ""))
                symbols = self._get_file_symbols(file_path)
                if not symbols:
                    return f"⚠️ No symbols found in {file_path}"
                output = [f"✓ Symbols in {file_path}:"]
                for s in symbols[:30]:
                    output.append(f"  L{s['line']}: {s['type']} {s['name']}")
                return "\n".join(output)
            
            else:
                return f"Unknown tool: {tool_name}"
        
        except Exception as e:
            return f"Tool error ({tool_name}): {str(e)}"
    
    def _execute_tool_calls(self, tool_calls: list[dict]) -> str:
        """Execute multiple tool calls and return combined results."""
        results = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            result = self._execute_tool(tool_name, call)
            results.append(f"## Tool: {tool_name}\n{result}")
        return "\n\n".join(results)
    
    def _log_llm_output(self, content: str, log: Callable[[str], None]) -> None:
        """Log the LLM's thinking and response to terminal (truncated for readability)."""
        # Check for thinking blocks (Claude extended thinking)
        thinking_match = re.search(r'<thinking>([\s\S]*?)</thinking>', content)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            # Truncate thinking to first 300 chars
            if len(thinking) > 300:
                thinking = thinking[:300] + "..."
            log(f"[Discovery] 💭 Thinking: {thinking}")
        
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
            log(f"[Discovery] 💬 Response: {response_text}")
    
    def _parse_response(self, content: str) -> dict:
        """
        Parse the LLM response.
        
        IMPORTANT: Prioritizes tool_calls over exploration_complete to prevent
        the LLM from hallucinating results without actually reading files.
        
        Handles multiple formats:
        1. <function_calls><invoke name="...">...</invoke></function_calls> - Anthropic-style XML
        2. <tool_calls>[...]</tool_calls> - Simple JSON in XML
        3. ```json {"tool_calls": [...]} ``` code blocks
        4. ```json {"exploration_complete": true, ...} ``` final response
        """
        all_tool_calls = []
        exploration_result = None
        
        # 1. Check for Anthropic-style <function_calls><invoke> XML format
        # This is what Claude often uses: <function_calls><invoke name="grep"><parameter name="pattern">...</parameter></invoke></function_calls>
        function_calls_blocks = re.findall(r'<function_calls>\s*([\s\S]*?)\s*</function_calls>', content)
        for block in function_calls_blocks:
            # Parse each <invoke> within the block
            invokes = re.findall(r'<invoke\s+name="([^"]+)"[^>]*>([\s\S]*?)</invoke>', block)
            for tool_name, params_block in invokes:
                tool_call = {"tool": tool_name}
                # Parse parameters
                params = re.findall(r'<parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)</parameter>', params_block)
                for param_name, param_value in params:
                    tool_call[param_name] = param_value.strip()
                all_tool_calls.append(tool_call)
        
        # 2. Check for <tool_calls> JSON format
        xml_tool_calls = re.findall(r'<tool_calls>\s*([\s\S]*?)\s*</tool_calls>', content)
        for xml_block in xml_tool_calls:
            try:
                parsed = json.loads(xml_block.strip())
                if isinstance(parsed, list):
                    all_tool_calls.extend(parsed)
                elif isinstance(parsed, dict) and parsed.get("tool"):
                    all_tool_calls.append(parsed)
            except json.JSONDecodeError:
                continue
        
        # 2. Check for ```json blocks
        json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', content)
        for block in json_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    # Check if this is a tool_calls block
                    if parsed.get("tool_calls"):
                        calls = parsed["tool_calls"]
                        if isinstance(calls, list):
                            all_tool_calls.extend(calls)
                    # Check if this is the final exploration_complete block
                    elif parsed.get("exploration_complete"):
                        exploration_result = parsed
            except json.JSONDecodeError:
                continue
        
        # 3. Check for bare ``` blocks that might be JSON
        bare_blocks = re.findall(r'```\s*([\s\S]*?)\s*```', content)
        for block in bare_blocks:
            block = block.strip()
            if block.startswith('{') or block.startswith('['):
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict):
                        if parsed.get("tool_calls"):
                            calls = parsed["tool_calls"]
                            if isinstance(calls, list):
                                all_tool_calls.extend(calls)
                        elif parsed.get("exploration_complete"):
                            exploration_result = parsed
                except json.JSONDecodeError:
                    continue
        
        # 4. Try finding raw JSON with tool_calls or exploration_complete
        try:
            json_match = re.search(r'\{[^{}]*"tool_calls"\s*:\s*\[[^\]]*\][^{}]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if parsed.get("tool_calls"):
                    all_tool_calls.extend(parsed["tool_calls"])
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # PRIORITY: Return tool_calls if we found any (prevents hallucination)
        # The LLM must actually execute tools before we accept exploration_complete
        if all_tool_calls:
            # Warn if LLM tried to output both (it shouldn't!)
            if exploration_result:
                print("[Discovery] ⚠️ LLM output both tool_calls AND exploration_complete in same response!")
                print("[Discovery] ⚠️ Ignoring exploration_complete - will execute tools first.")
            return {"tool_calls": all_tool_calls}
        
        # Only return exploration_complete if there were NO tool calls
        if exploration_result:
            return exploration_result
        
        return {"raw_response": content}
    
    def _build_discovered_context(self, data: dict) -> DiscoveredContext:
        """Build DiscoveredContext from LLM's final response."""
        ctx_data = data.get("discovered_context", {})
        
        context = DiscoveredContext(
            frameworks_detected=ctx_data.get("frameworks", []),
            entry_points=ctx_data.get("entry_points", []),
            config_files=ctx_data.get("config_files", {}),
            test_files=ctx_data.get("test_files", []),
        )
        
        # Parse relevant files
        for f_data in ctx_data.get("relevant_files", []):
            file = DiscoveredFile(
                path=f_data.get("path", ""),
                summary=f_data.get("summary", ""),
                symbols=f_data.get("symbols", []),
            )
            
            # Add key code as a snippet if provided
            if f_data.get("key_code"):
                file.key_snippets.append(CodeSnippet(
                    file_path=file.path,
                    start_line=0,
                    end_line=0,
                    content=f_data.get("key_code", ""),
                    relevance="Key pattern to follow",
                ))
            
            context.relevant_files.append(file)
        
        # Parse patterns
        for p_data in ctx_data.get("patterns", []):
            context.patterns.append(DiscoveredPattern(
                pattern_type=p_data.get("type", ""),
                description=p_data.get("description", ""),
                examples=p_data.get("examples", []),
            ))
        
        return context
    
    async def discover(
        self,
        user_prompt: str,
        max_rounds: int = 20,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> DiscoveredContext:
        """
        Discover relevant context for the user's prompt.
        
        Args:
            user_prompt: The user's request/prompt
            max_rounds: Maximum exploration rounds
            on_status: Optional callback for status updates
            
        Returns:
            DiscoveredContext with all relevant information
        """
        def log(msg: str):
            if on_status:
                on_status(msg)
        
        # Initialize debug logging
        self.debug.start_session(user_prompt, self.repo_path.name)
        
        # Scan repository
        log("[Discovery] Scanning repository structure...")
        self.repo_structure = self.scanner.scan()
        log(f"[Discovery] Found {self.repo_structure.total_files} files")
        log(f"[Discovery] Debug logs: .agentic/discovery_debug_log.txt")
        
        # Log repo scan
        self.debug.log_repo_scan(
            self.repo_structure.total_files,
            self.repo_structure.total_dirs,
            self.repo_structure.entry_points,
            self.repo_structure.file_tree,
        )
        
        # Build initial messages
        messages = [
            Message("system", CONTEXT_DISCOVERY_SYSTEM_PROMPT),
            Message("user", f"""Please explore this codebase to find context relevant to the following user request:

---
{user_prompt}
---

Start by exploring the codebase structure and searching for relevant patterns. Use the tools available to understand what already exists and what will be relevant for this task.

Repository structure overview:
{self.repo_structure.file_tree[:3000]}

Entry points detected: {', '.join(self.repo_structure.entry_points[:5]) if self.repo_structure.entry_points else 'None detected'}
"""),
        ]
        
        files_explored = 0
        
        for round_num in range(max_rounds):
            log(f"[Discovery] Round {round_num + 1}/{max_rounds}...")
            
            # Log round start with full context
            self.debug.log_round_start(round_num + 1, messages)
            
            # Call LLM
            response = await self.llm.chat(messages, max_tokens=4096)
            self.token_usage = self.token_usage.add(response.token_usage)
            
            # Parse response
            parsed = self._parse_response(response.content)
            
            # Log LLM response
            self.debug.log_llm_response(response.content, list(parsed.keys()))
            
            # Show LLM's thinking/response in terminal (truncated for readability)
            self._log_llm_output(response.content, log)
            
            # Log what we parsed
            if parsed.get("tool_calls"):
                log(f"[Discovery] Parsed {len(parsed['tool_calls'])} tool call(s)")
            elif parsed.get("exploration_complete"):
                log(f"[Discovery] LLM signaled exploration complete")
            elif parsed.get("raw_response"):
                log(f"[Discovery] Warning: Could not parse structured response")
            
            # Check if exploration is complete
            if parsed.get("exploration_complete"):
                log("[Discovery] Exploration complete! Building context...")
                context = self._build_discovered_context(parsed)
                context.files_explored = files_explored
                context.exploration_rounds = round_num + 1
                
                # Log discovery complete
                self.debug.log_discovery_complete(context)
                
                # Log what we're handing off to Master Agent
                handoff_context = context.to_prompt_context()
                self.debug.log_handoff_to_master(handoff_context)
                
                return context
            
            # Execute tool calls
            if parsed.get("tool_calls"):
                tool_calls = parsed["tool_calls"]
                if isinstance(tool_calls, list) and tool_calls:
                    original_count = len(tool_calls)
                    was_limited = False
                    
                    # Limit to 5 tool calls per round to force incremental exploration
                    if original_count > 5:
                        log(f"[Discovery] ⚠️ LLM requested {original_count} tool calls - limiting to first 5")
                        tool_calls = tool_calls[:5]
                        was_limited = True
                    
                    log(f"[Discovery] Executing {len(tool_calls)} tool(s)...")
                    for tc in tool_calls:
                        tool_name = tc.get('tool', 'unknown')
                        args = {k: v for k, v in tc.items() if k != 'tool'}
                        log(f"  → {tool_name}: {args}")
                    
                    # Log tool calls (only the ones we're actually executing)
                    self.debug.log_tool_calls(tool_calls, was_limited=was_limited, original_count=original_count)
                    
                    # Count file reads
                    files_explored += sum(1 for tc in tool_calls if tc.get("tool") == "read_file")
                    
                    tool_results = self._execute_tool_calls(tool_calls)
                    
                    # Log tool results
                    self.debug.log_tool_results(tool_results)
                    
                    # Add to conversation
                    messages.append(Message("assistant", response.content))
                    
                    # Add warning if approaching max rounds (start warning with 5 rounds left)
                    if round_num >= max_rounds - 5:
                        remaining = max_rounds - round_num - 1
                        tool_results += f"\n\n⚠️ REMINDER: You have {remaining} exploration round(s) remaining. Please wrap up your exploration and provide your final discovered_context summary soon."
                    
                    messages.append(Message("user", f"Tool results:\n\n{tool_results}"))
                    continue
            
            # No tool calls and not complete - add response and continue
            messages.append(Message("assistant", response.content))
            
            # Remind LLM to wrap up if approaching max rounds (urgent warning with 3 rounds left)
            if round_num >= max_rounds - 3:
                messages.append(Message("user", f"⚠️ You have {max_rounds - round_num - 1} rounds left. Please finalize your discovered context soon. Continue exploring or provide your final summary."))
            else:
                messages.append(Message("user", "Continue exploring or finalize your discovered context."))
        
        # Max rounds reached - force the LLM to summarize what it found
        log("[Discovery] Max rounds reached, requesting summary...")
        
        # Send a final message forcing completion
        messages.append(Message("user", """You have reached the maximum exploration rounds. 

STOP making tool calls. Based on what you've explored so far, provide your discovered context NOW.

Respond with your findings in this EXACT format:

```json
{
  "exploration_complete": true,
  "discovered_context": {
    "frameworks": ["list frameworks you found"],
    "relevant_files": [
      {
        "path": "path/to/file.py",
        "summary": "What this file does and why it's relevant",
        "symbols": ["key", "functions", "or", "classes"],
        "key_code": "Brief description of important patterns"
      }
    ],
    "patterns": [
      {
        "type": "pattern_type",
        "description": "Description of the pattern",
        "examples": []
      }
    ],
    "entry_points": ["main entry files"],
    "config_files": {},
    "test_files": []
  }
}
```

DO NOT make any more tool calls. Summarize what you found."""))
        
        # One more LLM call to get the summary
        self.debug.log_round_start(max_rounds + 1, messages)
        response = await self.llm.chat(messages, max_tokens=4096)
        self.token_usage = self.token_usage.add(response.token_usage)
        
        parsed = self._parse_response(response.content)
        self.debug.log_llm_response(response.content, list(parsed.keys()))
        
        # Check if we got a valid response
        if parsed.get("exploration_complete"):
            log("[Discovery] Got summary from LLM")
            context = self._build_discovered_context(parsed)
            context.files_explored = files_explored
            context.exploration_rounds = max_rounds
            
            self.debug.log_discovery_complete(context)
            handoff_context = context.to_prompt_context()
            self.debug.log_handoff_to_master(handoff_context)
            
            return context
        else:
            # Still couldn't get a summary - return empty
            log("[Discovery] Warning: Could not get summary, returning empty context")
            self.debug.log_error("Max exploration rounds reached and LLM did not provide summary")
            
            context = DiscoveredContext(
                files_explored=files_explored,
                exploration_rounds=max_rounds,
            )
            
            self.debug.log_handoff_to_master(context.to_prompt_context() or "(No context discovered)")
            return context
    
    async def close(self):
        """Clean up resources."""
        await self.llm.close()


async def discover_context(
    repo_path: str | Path,
    user_prompt: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    on_status: Optional[Callable[[str], None]] = None,
) -> DiscoveredContext:
    """
    Convenience function to discover context.
    
    Args:
        repo_path: Path to repository
        user_prompt: User's request
        api_key: OpenRouter API key
        model: Model to use
        on_status: Status callback
        
    Returns:
        DiscoveredContext with relevant codebase information
    """
    agent = ContextDiscoveryAgent(repo_path, api_key=api_key, model=model)
    
    try:
        return await agent.discover(user_prompt, on_status=on_status)
    finally:
        await agent.close()


if __name__ == "__main__":
    import sys
    
    async def main():
        repo = sys.argv[1] if len(sys.argv) > 1 else "."
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Analyze this repository"
        
        print(f"Discovering context for: {repo}")
        print(f"Prompt: {prompt}")
        print()
        
        def status(msg):
            print(msg)
        
        context = await discover_context(repo, prompt, on_status=status)
        
        print("\n" + "=" * 60)
        print("DISCOVERED CONTEXT")
        print("=" * 60)
        print(context.to_prompt_context())
    
    asyncio.run(main())

