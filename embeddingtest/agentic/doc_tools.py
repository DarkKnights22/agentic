"""
Tools for Documentation Agent.

Provides budget-aware exploration tools and memory operations.
All exploration tools check budget before executing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import re
import json

if TYPE_CHECKING:
    from .doc_memory import MemoryManager, MemoryQuery
    from .doc_phases import PhaseOrchestrator


# =============================================================================
# Tool Result Types
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    content: str
    budget_warning: Optional[str] = None


# =============================================================================
# Documentation Agent Tools
# =============================================================================

class DocTools:
    """
    Budget-aware tools for documentation exploration.

    All exploration tools check budget before executing.
    Memory tools enforce the retrieval contract.
    """

    # File extensions to search
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.rb', '.c', '.cpp', '.h', '.hpp', '.cs'}
    CONFIG_EXTENSIONS = {'.json', '.yaml', '.yml', '.toml', '.ini', '.env'}

    def __init__(
        self,
        repo_path: Path,
        memory_manager: "MemoryManager",
        orchestrator: "PhaseOrchestrator"
    ):
        self.repo_path = Path(repo_path)
        self.memory = memory_manager
        self.orchestrator = orchestrator

    # -------------------------------------------------------------------------
    # Exploration Tools (Budget-Checked)
    # -------------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        file_extensions: Optional[list[str]] = None,
        context_lines: int = 0,
        max_results: int = 20
    ) -> ToolResult:
        """
        Search for pattern across codebase.

        Budget-checked: counts against grep_calls limit.
        """
        # Check budget
        allowed, msg = self.orchestrator.check_budget("grep")
        if not allowed:
            return ToolResult(success=False, content=msg)

        # Record operation
        self.orchestrator.record_operation("grep")

        extensions = set(file_extensions) if file_extensions else self.CODE_EXTENSIONS | self.CONFIG_EXTENSIONS
        results = []

        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(success=False, content=f"Invalid regex pattern: {e}")

        for file_path in self._iter_files(extensions):
            if len(results) >= max_results:
                break

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if compiled_pattern.search(line):
                        relative_path = file_path.relative_to(self.repo_path)

                        if context_lines > 0:
                            start_idx = max(0, line_num - 1 - context_lines)
                            end_idx = min(len(lines), line_num + context_lines)
                            context_content = ''.join(lines[start_idx:end_idx])
                            content = context_content[:500]
                        else:
                            content = line.strip()[:200]

                        results.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'content': content
                        })

                        if len(results) >= max_results:
                            break

            except (IOError, OSError):
                continue

        # Format output
        if not results:
            output = f"No matches found for pattern: {pattern}"
        else:
            output_lines = [f"Found {len(results)} matches for '{pattern}':"]
            for r in results:
                output_lines.append(f"  {r['file']}:{r['line']}: {r['content']}")
            output = "\n".join(output_lines)

        # Check remaining budget
        budget_warning = self._get_budget_warning("grep")

        return ToolResult(success=True, content=output, budget_warning=budget_warning)

    def read_file(
        self,
        file_path: str,
        max_lines: int = 200
    ) -> ToolResult:
        """
        Read a file's contents.

        Budget-checked: counts against files_read limit.
        """
        # Check budget
        allowed, msg = self.orchestrator.check_budget("file_read")
        if not allowed:
            return ToolResult(success=False, content=msg)

        # Record operation
        self.orchestrator.record_operation("file_read")

        full_path = self.repo_path / file_path
        if not full_path.exists():
            return ToolResult(success=False, content=f"File not found: {file_path}")

        if not full_path.is_file():
            return ToolResult(success=False, content=f"Not a file: {file_path}")

        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            if len(lines) > max_lines:
                content = ''.join(lines[:max_lines])
                content += f"\n... (truncated, {len(lines) - max_lines} more lines)"
            else:
                content = ''.join(lines)

            output = f"Contents of {file_path}:\n```\n{content}\n```"

        except (IOError, OSError) as e:
            return ToolResult(success=False, content=f"Error reading file: {e}")

        # Check remaining budget
        budget_warning = self._get_budget_warning("file_read")

        return ToolResult(success=True, content=output, budget_warning=budget_warning)

    def list_files(
        self,
        directory: str = "",
        pattern: str = "*",
        max_depth: int = 3
    ) -> ToolResult:
        """
        List files in a directory.

        NOT budget-checked - lightweight exploration allowed.
        """
        target_dir = self.repo_path / directory if directory else self.repo_path

        if not target_dir.exists():
            return ToolResult(success=False, content=f"Directory not found: {directory}")

        files = []
        dirs = []

        try:
            for item in sorted(target_dir.iterdir()):
                # Skip hidden and common ignore patterns
                if item.name.startswith('.') or item.name in ('node_modules', '__pycache__', 'venv', '.git'):
                    continue

                relative = item.relative_to(self.repo_path)

                if item.is_dir():
                    dirs.append(f"  [DIR] {relative}/")
                elif item.is_file():
                    if pattern == "*" or item.match(pattern):
                        files.append(f"  {relative}")

        except (IOError, OSError) as e:
            return ToolResult(success=False, content=f"Error listing directory: {e}")

        output_lines = [f"Contents of {directory or '.'}:"]
        output_lines.extend(dirs[:50])  # Limit directories
        output_lines.extend(files[:100])  # Limit files

        if len(dirs) > 50 or len(files) > 100:
            output_lines.append(f"  ... (truncated)")

        return ToolResult(success=True, content="\n".join(output_lines))

    def get_symbols(self, file_path: str) -> ToolResult:
        """
        Extract symbols (classes, functions) from a file.

        Budget-checked: counts against symbols_calls limit.
        """
        # Check budget
        allowed, msg = self.orchestrator.check_budget("symbols")
        if not allowed:
            return ToolResult(success=False, content=msg)

        # Record operation
        self.orchestrator.record_operation("symbols")

        full_path = self.repo_path / file_path
        if not full_path.exists():
            return ToolResult(success=False, content=f"File not found: {file_path}")

        symbols = self._extract_symbols(full_path)

        if not symbols:
            output = f"No symbols found in {file_path}"
        else:
            output_lines = [f"Symbols in {file_path}:"]
            for sym in symbols:
                output_lines.append(f"  {sym['type']:10} {sym['name']:30} (line {sym['line']})")
            output = "\n".join(output_lines)

        # Check remaining budget
        budget_warning = self._get_budget_warning("symbols")

        return ToolResult(success=True, content=output, budget_warning=budget_warning)

    # -------------------------------------------------------------------------
    # Memory Tools
    # -------------------------------------------------------------------------

    def query_memory(
        self,
        query_type: str,
        max_entries: int,
        filter_by: Optional[dict] = None,
        required_fields: Optional[list[str]] = None,
        min_confidence: float = 0.0,
        sort_by: str = "relevance"
    ) -> ToolResult:
        """
        Query memory with bounded retrieval.

        Enforces the retrieval contract from doc_memory.py.
        """
        from .doc_memory import MemoryQuery, MemoryQueryError

        # Check budget for doc generation phase
        allowed, msg = self.orchestrator.check_budget("memory_query")
        if not allowed:
            return ToolResult(success=False, content=msg)

        self.orchestrator.record_operation("memory_query")

        query = MemoryQuery(
            query_type=query_type,
            max_entries=max_entries,
            filter_by=filter_by or {},
            required_fields=required_fields or [],
            min_confidence=min_confidence,
            sort_by=sort_by,
        )

        try:
            results = self.memory.query_memory(query)
        except MemoryQueryError as e:
            return ToolResult(success=False, content=f"Memory query error: {e}")

        if not results:
            output = f"No {query_type} entries found matching criteria"
        else:
            output = f"Found {len(results)} {query_type} entries:\n"
            output += json.dumps(results, indent=2)

        return ToolResult(success=True, content=output)

    def store_discovery(self, entry_type: str, data: dict) -> ToolResult:
        """
        Store a discovery to memory.

        Validates entry against schema before storing.
        """
        try:
            self.memory.store_discovery(entry_type, data)
            return ToolResult(
                success=True,
                content=f"Successfully stored {entry_type} entry"
            )
        except (ValueError, TypeError) as e:
            return ToolResult(
                success=False,
                content=f"Error storing {entry_type}: {e}"
            )

    def get_phase_context(self) -> ToolResult:
        """Get current phase context including budget status."""
        context = self.orchestrator.get_phase_context()
        return ToolResult(
            success=True,
            content=json.dumps(context, indent=2)
        )

    def mark_explored(self, target_type: str, target_id: str) -> ToolResult:
        """Mark a target (component, file, etc.) as fully explored."""
        if target_type == "component":
            self.orchestrator.complete_component(target_id)
            self.memory.mark_component_explored(target_id)
            return ToolResult(success=True, content=f"Marked component '{target_id}' as explored")
        else:
            return ToolResult(success=False, content=f"Unknown target type: {target_type}")

    def estimate_token_usage(
        self,
        query_type: str,
        max_entries: int,
        filter_by: Optional[dict] = None,
        required_fields: Optional[list[str]] = None,
        min_confidence: float = 0.0
    ) -> ToolResult:
        """Estimate token usage for a memory query before executing."""
        from .doc_memory import MemoryQuery

        query = MemoryQuery(
            query_type=query_type,
            max_entries=max_entries,
            filter_by=filter_by or {},
            required_fields=required_fields or [],
            min_confidence=min_confidence,
        )

        estimated_tokens = self.memory.estimate_token_usage(query)
        return ToolResult(
            success=True,
            content=f"Estimated tokens for query: {estimated_tokens}"
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _iter_files(self, extensions: set[str]):
        """Iterate over files with given extensions."""
        for path in self.repo_path.rglob('*'):
            if path.is_file() and path.suffix in extensions:
                # Skip common ignore patterns
                parts = path.parts
                if any(p.startswith('.') or p in ('node_modules', '__pycache__', 'venv') for p in parts):
                    continue
                yield path

    def _get_budget_warning(self, operation: str) -> Optional[str]:
        """Get a warning if budget is running low."""
        context = self.orchestrator.get_phase_context()
        remaining = context.get("remaining", {})

        if operation == "grep":
            left = remaining.get("grep_calls", 999)
            if left <= 2:
                return f"WARNING: Only {left} grep calls remaining. Consider summarizing."
        elif operation == "file_read":
            left = remaining.get("files_read", 999)
            if left <= 2:
                return f"WARNING: Only {left} file reads remaining. Consider summarizing."
        elif operation == "symbols":
            left = remaining.get("symbols_calls", 999)
            if left <= 1:
                return f"WARNING: Only {left} symbols calls remaining."

        return None

    def _extract_symbols(self, file_path: Path) -> list[dict]:
        """Extract symbols from a file using regex patterns."""
        SYMBOL_PATTERNS = {
            '.py': [
                (r'^class\s+(\w+)', 'class'),
                (r'^def\s+(\w+)', 'function'),
                (r'^\s+def\s+(\w+)', 'method'),
                (r'^(\w+)\s*=\s*', 'variable'),
            ],
            '.js': [
                (r'^class\s+(\w+)', 'class'),
                (r'^function\s+(\w+)', 'function'),
                (r'^const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>', 'function'),
                (r'^const\s+(\w+)\s*=\s*function', 'function'),
                (r'^export\s+(?:const|let|var)\s+(\w+)', 'export'),
            ],
            '.ts': [
                (r'^(?:export\s+)?class\s+(\w+)', 'class'),
                (r'^(?:export\s+)?interface\s+(\w+)', 'interface'),
                (r'^(?:export\s+)?type\s+(\w+)', 'type'),
                (r'^(?:export\s+)?function\s+(\w+)', 'function'),
                (r'^(?:export\s+)?const\s+(\w+)\s*=', 'const'),
            ],
            '.go': [
                (r'^type\s+(\w+)\s+struct', 'struct'),
                (r'^type\s+(\w+)\s+interface', 'interface'),
                (r'^func\s+(\w+)', 'function'),
                (r'^func\s+\([^)]+\)\s*(\w+)', 'method'),
            ],
            '.rs': [
                (r'^(?:pub\s+)?struct\s+(\w+)', 'struct'),
                (r'^(?:pub\s+)?enum\s+(\w+)', 'enum'),
                (r'^(?:pub\s+)?trait\s+(\w+)', 'trait'),
                (r'^(?:pub\s+)?fn\s+(\w+)', 'function'),
                (r'^\s+(?:pub\s+)?fn\s+(\w+)', 'method'),
            ],
        }

        ext = file_path.suffix.lower()
        # Also handle .tsx, .jsx
        if ext in ('.tsx', '.jsx'):
            ext = '.ts' if ext == '.tsx' else '.js'

        patterns = SYMBOL_PATTERNS.get(ext, [])
        if not patterns:
            return []

        symbols = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern, symbol_type in patterns:
                        match = re.match(pattern, line)
                        if match:
                            symbols.append({
                                'name': match.group(1),
                                'type': symbol_type,
                                'line': line_num,
                            })
        except (IOError, OSError):
            pass

        return symbols


# =============================================================================
# Tool Definitions for LLM
# =============================================================================

TOOL_DEFINITIONS = """
Available tools for documentation exploration:

## Exploration Tools (Budget-Checked)

### grep
Search for a pattern across the codebase.
Parameters:
  - pattern: regex pattern to search for (required)
  - file_extensions: list of extensions to search (optional, defaults to code files)
  - context_lines: number of context lines to include (optional, default 0)
  - max_results: maximum results to return (optional, default 20)

### read_file
Read a file's contents.
Parameters:
  - file_path: path relative to repo root (required)
  - max_lines: maximum lines to read (optional, default 200)

### list_files
List files in a directory.
Parameters:
  - directory: path relative to repo root (optional, defaults to root)
  - pattern: glob pattern to filter files (optional, default "*")

### get_symbols
Extract class/function definitions from a file.
Parameters:
  - file_path: path relative to repo root (required)

## Memory Tools

### query_memory
Query stored memory with bounded retrieval.
Parameters:
  - query_type: "architecture" | "component" | "file" | "data_model" | "flow" | "cross_cutting" (required)
  - max_entries: maximum entries to return - REQUIRED, no default (required)
  - filter_by: dict of filters e.g. {"component": "auth"} (optional)
  - required_fields: list of fields to return (optional, returns all if empty)
  - min_confidence: minimum confidence score 0.0-1.0 (optional, default 0.0)
  - sort_by: "relevance" | "recency" | "confidence" (optional, default "relevance")

### store_discovery
Store a discovery to persistent memory.
Parameters:
  - entry_type: "architecture" | "component" | "file" | "data_model" | "flow" | "cross_cutting" (required)
  - data: dict with entry data matching the schema (required)

### get_phase_context
Get current phase, budget status, and progress.
Parameters: none

### mark_explored
Mark a component or file as fully explored.
Parameters:
  - target_type: "component" (required)
  - target_id: identifier of the target (required)

### estimate_token_usage
Estimate tokens for a memory query before executing.
Parameters: same as query_memory
"""


def get_tool_definitions() -> str:
    """Get tool definitions for LLM prompts."""
    return TOOL_DEFINITIONS
