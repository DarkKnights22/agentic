"""
Execution Tools for Sub-Agents

Provides the core tools that sub-agents use during Phase 2 execution:
- apply_patch: Apply unified diffs to files
- run_tests: Execute test commands and capture output
"""

import os
import re
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    file_path: str
    error_message: Optional[str] = None
    lines_added: int = 0
    lines_removed: int = 0
    backup_path: Optional[str] = None


@dataclass
class TestResult:
    """Result of running tests."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: str
    log_file: Optional[str] = None
    skipped: bool = False  # True if tests were skipped (no test framework found)


class PatchError(Exception):
    """Error applying a patch."""
    pass


class TestError(Exception):
    """Error running tests."""
    pass


def parse_unified_diff(diff_content: str) -> list[dict]:
    """
    Parse a unified diff into structured hunks.
    
    Returns list of hunks with:
    - old_start: starting line in original file
    - old_count: number of lines from original
    - new_start: starting line in new file
    - new_count: number of lines in new version
    - lines: list of (type, content) tuples where type is ' ', '+', or '-'
    """
    hunks = []
    current_hunk = None
    
    lines = diff_content.strip().split('\n')
    
    for line in lines:
        # Skip file headers
        if line.startswith('---') or line.startswith('+++'):
            continue
        
        # Hunk header: @@ -10,6 +10,9 @@
        hunk_match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if hunk_match:
            if current_hunk:
                hunks.append(current_hunk)
            
            current_hunk = {
                'old_start': int(hunk_match.group(1)),
                'old_count': int(hunk_match.group(2) or 1),
                'new_start': int(hunk_match.group(3)),
                'new_count': int(hunk_match.group(4) or 1),
                'lines': [],
            }
            continue
        
        # Content lines
        if current_hunk is not None:
            if line.startswith('+'):
                current_hunk['lines'].append(('+', line[1:]))
            elif line.startswith('-'):
                current_hunk['lines'].append(('-', line[1:]))
            elif line.startswith(' ') or line == '':
                current_hunk['lines'].append((' ', line[1:] if line.startswith(' ') else ''))
    
    if current_hunk:
        hunks.append(current_hunk)
    
    return hunks


def apply_hunk(original_lines: list[str], hunk: dict) -> list[str]:
    """
    Apply a single hunk to the original lines.
    
    Returns the modified lines.
    """
    result = []
    old_line_idx = 0
    hunk_start = hunk['old_start'] - 1  # Convert to 0-indexed
    
    # Copy lines before the hunk
    while old_line_idx < hunk_start:
        if old_line_idx < len(original_lines):
            result.append(original_lines[old_line_idx])
        old_line_idx += 1
    
    # Apply the hunk
    for line_type, content in hunk['lines']:
        if line_type == ' ':
            # Context line - should match original
            if old_line_idx < len(original_lines):
                result.append(original_lines[old_line_idx])
                old_line_idx += 1
            else:
                result.append(content)
        elif line_type == '+':
            # Addition
            result.append(content)
        elif line_type == '-':
            # Deletion - skip the original line
            old_line_idx += 1
    
    # Copy remaining lines after the hunk
    while old_line_idx < len(original_lines):
        result.append(original_lines[old_line_idx])
        old_line_idx += 1
    
    return result


def apply_patch(
    repo_path: Path,
    file_path: str,
    diff_content: str,
    create_backup: bool = True,
) -> PatchResult:
    """
    Apply a unified diff to a file.
    
    Args:
        repo_path: Root path of the repository
        file_path: Relative path to the file to patch
        diff_content: Unified diff content
        create_backup: Whether to create a backup before patching
        
    Returns:
        PatchResult with success status and details
    """
    full_path = repo_path / file_path
    backup_path = None
    
    try:
        # Parse the diff
        hunks = parse_unified_diff(diff_content)
        
        if not hunks:
            return PatchResult(
                success=False,
                file_path=file_path,
                error_message="No valid hunks found in diff",
            )
        
        # Read original file (or create empty for new files)
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            original_lines = original_content.split('\n')
            
            # Create backup
            if create_backup:
                backup_path = str(full_path) + f'.backup.{datetime.now().strftime("%Y%m%d%H%M%S")}'
                shutil.copy2(full_path, backup_path)
        else:
            # New file - ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            original_lines = []
        
        # Apply hunks in reverse order to preserve line numbers
        result_lines = original_lines.copy()
        lines_added = 0
        lines_removed = 0
        
        for hunk in reversed(hunks):
            result_lines = apply_hunk(result_lines, hunk)
            for line_type, _ in hunk['lines']:
                if line_type == '+':
                    lines_added += 1
                elif line_type == '-':
                    lines_removed += 1
        
        # Write the patched content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result_lines))
        
        return PatchResult(
            success=True,
            file_path=file_path,
            lines_added=lines_added,
            lines_removed=lines_removed,
            backup_path=backup_path,
        )
        
    except Exception as e:
        return PatchResult(
            success=False,
            file_path=file_path,
            error_message=str(e),
            backup_path=backup_path,
        )


def revert_patch(repo_path: Path, patch_result: PatchResult) -> bool:
    """
    Revert a patch using its backup.
    
    Returns True if successful.
    """
    if not patch_result.backup_path:
        return False
    
    try:
        full_path = repo_path / patch_result.file_path
        backup_path = Path(patch_result.backup_path)
        
        if backup_path.exists():
            shutil.copy2(backup_path, full_path)
            backup_path.unlink()  # Remove backup after restore
            return True
        return False
    except Exception:
        return False


def detect_test_framework(repo_path: Path) -> Optional[str]:
    """
    Auto-detect the test framework and return appropriate command.
    """
    # Python - check for pytest
    if (repo_path / 'pytest.ini').exists() or \
       (repo_path / 'pyproject.toml').exists() or \
       (repo_path / 'setup.cfg').exists():
        # Check if pytest is available
        try:
            subprocess.run(['pytest', '--version'], capture_output=True, check=True)
            return 'pytest'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    # Python - check for unittest
    if (repo_path / 'tests').is_dir() or (repo_path / 'test').is_dir():
        return 'python -m pytest'
    
    # Node.js
    package_json = repo_path / 'package.json'
    if package_json.exists():
        try:
            import json
            with open(package_json) as f:
                pkg = json.load(f)
            scripts = pkg.get('scripts', {})
            if 'test' in scripts:
                return 'npm test'
        except Exception:
            pass
    
    # Go
    if (repo_path / 'go.mod').exists():
        return 'go test ./...'
    
    # Rust
    if (repo_path / 'Cargo.toml').exists():
        return 'cargo test'
    
    return None


def _is_valid_command(command: str) -> bool:
    """
    Check if a string looks like a valid shell command vs a description.
    
    Returns False for things like:
    - "Manual testing with curl/Postman..."
    - "Run the app and check..."
    - Long sentences with multiple spaces
    """
    if not command or not command.strip():
        return False
    
    command = command.strip()
    
    # Descriptions tend to be long and have many words
    words = command.split()
    if len(words) > 10:
        return False
    
    # Common description starters that aren't commands
    description_starters = [
        'manual', 'run the', 'check', 'verify', 'test by', 'use', 
        'open', 'go to', 'navigate', 'click', 'follow', 'ensure',
        'make sure', 'confirm', 'integration test', 'e2e test'
    ]
    lower_cmd = command.lower()
    for starter in description_starters:
        if lower_cmd.startswith(starter):
            return False
    
    # Valid commands usually start with known command patterns
    valid_starters = [
        'pytest', 'python', 'npm', 'npx', 'yarn', 'node', 'cargo',
        'go ', 'make', 'mvn', 'gradle', 'dotnet', 'ruby', 'bundle',
        'php', 'composer', 'mix', 'elixir', 'swift', 'xcodebuild',
        './', 'bash', 'sh ', 'cmd', 'powershell', 'pwsh',
    ]
    for starter in valid_starters:
        if lower_cmd.startswith(starter):
            return True
    
    # If it's short and doesn't look like a sentence, assume it might be valid
    # (could be a custom script name)
    if len(words) <= 5 and not any(word in lower_cmd for word in ['testing', 'with', 'using', 'manually']):
        return True
    
    return False


def run_tests(
    repo_path: Path,
    command: Optional[str] = None,
    log_file: Optional[Path] = None,
    timeout: int = 300,
    env: Optional[dict] = None,
) -> TestResult:
    """
    Run tests and capture output.
    
    Args:
        repo_path: Root path of the repository
        command: Test command to run (auto-detected if None)
        log_file: Path to save test output
        timeout: Maximum seconds to wait for tests
        env: Additional environment variables
        
    Returns:
        TestResult with success status and output
    """
    # Check if command is valid (not a description)
    if command and not _is_valid_command(command):
        return TestResult(
            success=True,  # Not a failure - just can't run this "command"
            exit_code=0,
            stdout=f"Skipped: '{command}' is not an executable command (appears to be a description/instruction).",
            stderr="",
            duration_seconds=0,
            command=command,
            skipped=True,
        )
    
    # Determine test command
    if not command:
        command = detect_test_framework(repo_path)
        if not command:
            # No test framework found - skip tests (this is OK, not a failure)
            return TestResult(
                success=True,  # Not a failure - just no tests to run
                exit_code=0,
                stdout="No test framework detected. Tests skipped.",
                stderr="",
                duration_seconds=0,
                command="",
                skipped=True,
            )
    
    # Prepare environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    # Run tests
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
        success = exit_code == 0
        
    except subprocess.TimeoutExpired:
        duration = timeout
        stdout = ""
        stderr = f"Test execution timed out after {timeout} seconds"
        exit_code = -1
        success = False
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        stdout = ""
        stderr = str(e)
        exit_code = -1
        success = False
    
    # Save to log file
    log_path = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Command: {command}\n")
            f.write(f"Exit Code: {exit_code}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("\n" + "="*60 + " STDOUT " + "="*60 + "\n\n")
            f.write(stdout)
            f.write("\n" + "="*60 + " STDERR " + "="*60 + "\n\n")
            f.write(stderr)
        log_path = str(log_file)
    
    return TestResult(
        success=success,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        command=command,
        log_file=log_path,
    )


def read_file_safe(repo_path: Path, file_path: str, max_lines: int = 500) -> Optional[str]:
    """
    Safely read a file's content, with truncation for large files.
    
    Returns None if file doesn't exist or can't be read.
    """
    full_path = repo_path / file_path
    
    try:
        if not full_path.exists():
            return None
        
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if len(lines) > max_lines:
            content = ''.join(lines[:max_lines])
            content += f"\n\n... (truncated, {len(lines) - max_lines} more lines)"
            return content
        
        return ''.join(lines)
        
    except Exception:
        return None


# =============================================================================
# Additional Tools for Enhanced Code Intelligence
# =============================================================================

@dataclass
class CommandResult:
    """Result of running a shell command."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: str


@dataclass
class LintDiagnostic:
    """A single linter diagnostic."""
    file: str
    line: int
    column: int
    severity: str  # "error", "warning", "info"
    message: str
    rule: Optional[str] = None


@dataclass
class LintResult:
    """Result of running linters."""
    success: bool
    diagnostics: list[LintDiagnostic]
    linter_used: str
    error_message: Optional[str] = None


@dataclass
class Symbol:
    """A code symbol (function, class, etc.)."""
    name: str
    type: str  # "function", "class", "method", "variable", "interface", etc.
    line: int
    end_line: Optional[int] = None
    signature: Optional[str] = None


@dataclass
class Reference:
    """A reference to a symbol."""
    file: str
    line: int
    column: int
    context: str  # The line content


def run_command(
    cwd: Path,
    command: str,
    timeout: int = 60,
    env: Optional[dict] = None,
) -> CommandResult:
    """
    Execute an arbitrary shell command.
    
    Args:
        cwd: Working directory
        command: Command to run
        timeout: Maximum seconds to wait
        env: Additional environment variables
        
    Returns:
        CommandResult with output and exit code
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return CommandResult(
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=duration,
            command=command,
        )
        
    except subprocess.TimeoutExpired:
        return CommandResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            duration_seconds=timeout,
            command=command,
        )
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        return CommandResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_seconds=duration,
            command=command,
        )


# Linter configurations for different languages
LINTER_CONFIGS = {
    '.py': [
        {'cmd': 'ruff check --output-format=json {file}', 'parser': 'ruff'},
        {'cmd': 'python -m py_compile {file}', 'parser': 'py_compile'},
    ],
    '.js': [
        {'cmd': 'eslint --format=json {file}', 'parser': 'eslint'},
    ],
    '.ts': [
        {'cmd': 'tsc --noEmit --pretty false {file}', 'parser': 'tsc'},
        {'cmd': 'eslint --format=json {file}', 'parser': 'eslint'},
    ],
    '.tsx': [
        {'cmd': 'tsc --noEmit --pretty false {file}', 'parser': 'tsc'},
        {'cmd': 'eslint --format=json {file}', 'parser': 'eslint'},
    ],
    '.rs': [
        {'cmd': 'cargo check --message-format=json 2>&1', 'parser': 'cargo'},
    ],
    '.go': [
        {'cmd': 'go vet {file}', 'parser': 'go_vet'},
        {'cmd': 'gofmt -e {file}', 'parser': 'gofmt'},
    ],
    '.java': [
        {'cmd': 'javac -Xlint:all {file}', 'parser': 'javac'},
    ],
    '.rb': [
        {'cmd': 'ruby -c {file}', 'parser': 'ruby'},
        {'cmd': 'rubocop --format json {file}', 'parser': 'rubocop'},
    ],
    # C/C++ - try clang first (better errors), fall back to gcc
    '.c': [
        {'cmd': 'clang -fsyntax-only -Wall {file}', 'parser': 'gcc'},
        {'cmd': 'gcc -fsyntax-only -Wall {file}', 'parser': 'gcc'},
    ],
    '.cpp': [
        {'cmd': 'clang++ -fsyntax-only -Wall -std=c++17 {file}', 'parser': 'gcc'},
        {'cmd': 'g++ -fsyntax-only -Wall -std=c++17 {file}', 'parser': 'gcc'},
    ],
    '.cc': [
        {'cmd': 'clang++ -fsyntax-only -Wall -std=c++17 {file}', 'parser': 'gcc'},
        {'cmd': 'g++ -fsyntax-only -Wall -std=c++17 {file}', 'parser': 'gcc'},
    ],
    '.h': [
        {'cmd': 'clang -fsyntax-only -Wall {file}', 'parser': 'gcc'},
    ],
    '.hpp': [
        {'cmd': 'clang++ -fsyntax-only -Wall -std=c++17 {file}', 'parser': 'gcc'},
    ],
    # C#
    '.cs': [
        {'cmd': 'dotnet build --no-restore 2>&1', 'parser': 'dotnet'},
        {'cmd': 'mcs -parse {file}', 'parser': 'mcs'},
    ],
}


def _parse_lint_output(output: str, stderr: str, parser: str, file_path: str) -> list[LintDiagnostic]:
    """Parse linter output into diagnostics."""
    diagnostics = []
    
    try:
        if parser == 'ruff':
            # Ruff JSON format
            import json
            data = json.loads(output) if output.strip() else []
            for item in data:
                diagnostics.append(LintDiagnostic(
                    file=item.get('filename', file_path),
                    line=item.get('location', {}).get('row', 1),
                    column=item.get('location', {}).get('column', 1),
                    severity='error' if item.get('code', '').startswith('E') else 'warning',
                    message=item.get('message', ''),
                    rule=item.get('code'),
                ))
                
        elif parser == 'eslint':
            # ESLint JSON format
            import json
            data = json.loads(output) if output.strip() else []
            for file_result in data:
                for msg in file_result.get('messages', []):
                    diagnostics.append(LintDiagnostic(
                        file=file_result.get('filePath', file_path),
                        line=msg.get('line', 1),
                        column=msg.get('column', 1),
                        severity='error' if msg.get('severity') == 2 else 'warning',
                        message=msg.get('message', ''),
                        rule=msg.get('ruleId'),
                    ))
                    
        elif parser == 'tsc':
            # TypeScript compiler output: file(line,col): error TS1234: message
            pattern = r'([^(]+)\((\d+),(\d+)\):\s*(error|warning)\s+(TS\d+):\s*(.+)'
            for line in (output + stderr).split('\n'):
                match = re.match(pattern, line)
                if match:
                    diagnostics.append(LintDiagnostic(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        severity=match.group(4),
                        message=match.group(6),
                        rule=match.group(5),
                    ))
                    
        elif parser == 'py_compile':
            # Python syntax errors in stderr
            # SyntaxError: invalid syntax (file.py, line 10)
            if stderr:
                match = re.search(r'File "([^"]+)", line (\d+)', stderr)
                if match:
                    diagnostics.append(LintDiagnostic(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=1,
                        severity='error',
                        message=stderr.split('\n')[-2] if '\n' in stderr else stderr,
                    ))
                    
        elif parser == 'cargo':
            # Cargo JSON format
            import json
            for line in output.split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('reason') == 'compiler-message':
                            msg = data.get('message', {})
                            spans = msg.get('spans', [{}])
                            span = spans[0] if spans else {}
                            diagnostics.append(LintDiagnostic(
                                file=span.get('file_name', file_path),
                                line=span.get('line_start', 1),
                                column=span.get('column_start', 1),
                                severity=msg.get('level', 'error'),
                                message=msg.get('message', ''),
                                rule=msg.get('code', {}).get('code') if msg.get('code') else None,
                            ))
                    except json.JSONDecodeError:
                        pass
                        
        elif parser in ('go_vet', 'gofmt', 'javac', 'ruby', 'gcc', 'mcs'):
            # Generic: file:line:col: message or file:line: message
            # Works for: Go, Java, Ruby, GCC/Clang, Mono C#
            pattern = r'([^:]+):(\d+)(?::(\d+))?:\s*(?:(error|warning|note):\s*)?(.+)'
            for line in (output + stderr).split('\n'):
                match = re.match(pattern, line)
                if match:
                    severity = match.group(4) if match.group(4) else 'error'
                    diagnostics.append(LintDiagnostic(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)) if match.group(3) else 1,
                        severity=severity,
                        message=match.group(5),
                    ))
                    
        elif parser == 'dotnet':
            # dotnet build output: file(line,col): error CS1234: message
            pattern = r'([^(]+)\((\d+),(\d+)\):\s*(error|warning)\s+(CS\d+):\s*(.+)'
            for line in (output + stderr).split('\n'):
                match = re.match(pattern, line)
                if match:
                    diagnostics.append(LintDiagnostic(
                        file=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        severity=match.group(4),
                        message=match.group(6),
                        rule=match.group(5),
                    ))
                    
    except Exception:
        # If parsing fails, return raw output as single diagnostic
        if stderr or output:
            diagnostics.append(LintDiagnostic(
                file=file_path,
                line=1,
                column=1,
                severity='error',
                message=f"Linter output: {stderr or output}"[:500],
            ))
    
    return diagnostics


def read_lints(repo_path: Path, file_path: str) -> LintResult:
    """
    Run appropriate linter for a file and return diagnostics.
    
    Supports: Python, JavaScript, TypeScript, Rust, Go, Java, Ruby
    
    Args:
        repo_path: Repository root path
        file_path: Relative path to file
        
    Returns:
        LintResult with diagnostics
    """
    full_path = repo_path / file_path
    
    if not full_path.exists():
        return LintResult(
            success=False,
            diagnostics=[],
            linter_used="none",
            error_message=f"File not found: {file_path}",
        )
    
    ext = Path(file_path).suffix.lower()
    
    if ext not in LINTER_CONFIGS:
        return LintResult(
            success=True,
            diagnostics=[],
            linter_used="none",
            error_message=f"No linter configured for {ext} files",
        )
    
    # Try linters in order until one works
    for config in LINTER_CONFIGS[ext]:
        cmd = config['cmd'].format(file=str(full_path))
        parser = config['parser']
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            diagnostics = _parse_lint_output(
                result.stdout, result.stderr, parser, file_path
            )
            
            return LintResult(
                success=True,
                diagnostics=diagnostics,
                linter_used=parser,
            )
            
        except FileNotFoundError:
            # Linter not installed, try next one
            continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue
    
    return LintResult(
        success=False,
        diagnostics=[],
        linter_used="none",
        error_message=f"No working linter found for {ext} files",
    )


# Symbol patterns for different languages
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
        (r'^const\s+(\w+)\s*=\s*(?:async\s+)?function', 'function'),
        (r'^const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', 'function'),
        (r'^const\s+(\w+)\s*=', 'variable'),
        (r'^let\s+(\w+)\s*=', 'variable'),
        (r'^\s+(\w+)\s*\([^)]*\)\s*\{', 'method'),
    ],
    '.ts': [
        (r'^(?:export\s+)?class\s+(\w+)', 'class'),
        (r'^(?:export\s+)?interface\s+(\w+)', 'interface'),
        (r'^(?:export\s+)?type\s+(\w+)', 'type'),
        (r'^(?:export\s+)?function\s+(\w+)', 'function'),
        (r'^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function', 'function'),
        (r'^(?:export\s+)?const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', 'function'),
        (r'^\s+(?:public|private|protected)?\s*(\w+)\s*\([^)]*\)', 'method'),
    ],
    '.tsx': [],  # Will inherit from .ts
    '.rs': [
        (r'^pub\s+struct\s+(\w+)', 'struct'),
        (r'^struct\s+(\w+)', 'struct'),
        (r'^pub\s+enum\s+(\w+)', 'enum'),
        (r'^enum\s+(\w+)', 'enum'),
        (r'^pub\s+fn\s+(\w+)', 'function'),
        (r'^fn\s+(\w+)', 'function'),
        (r'^\s+pub\s+fn\s+(\w+)', 'method'),
        (r'^\s+fn\s+(\w+)', 'method'),
        (r'^pub\s+trait\s+(\w+)', 'trait'),
        (r'^trait\s+(\w+)', 'trait'),
        (r'^impl\s+(\w+)', 'impl'),
    ],
    '.go': [
        (r'^type\s+(\w+)\s+struct', 'struct'),
        (r'^type\s+(\w+)\s+interface', 'interface'),
        (r'^func\s+\([^)]+\)\s+(\w+)', 'method'),
        (r'^func\s+(\w+)', 'function'),
    ],
    '.java': [
        (r'^(?:public|private|protected)?\s*class\s+(\w+)', 'class'),
        (r'^(?:public|private|protected)?\s*interface\s+(\w+)', 'interface'),
        (r'^(?:public|private|protected)?\s*enum\s+(\w+)', 'enum'),
        (r'^\s+(?:public|private|protected)?\s*(?:static\s+)?[\w<>,\[\]\s]+\s+(\w+)\s*\(', 'method'),
    ],
    '.rb': [
        (r'^class\s+(\w+)', 'class'),
        (r'^module\s+(\w+)', 'module'),
        (r'^def\s+(\w+)', 'method'),
        (r'^\s+def\s+(\w+)', 'method'),
    ],
    '.c': [
        (r'^(?:static\s+)?(?:inline\s+)?[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*\{', 'function'),
        (r'^typedef\s+struct\s+\w*\s*\{[^}]*\}\s*(\w+)', 'struct'),
        (r'^struct\s+(\w+)', 'struct'),
    ],
    '.cpp': [
        (r'^class\s+(\w+)', 'class'),
        (r'^struct\s+(\w+)', 'struct'),
        (r'^(?:virtual\s+)?[\w\s\*:]+\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*\{', 'function'),
    ],
    '.h': [],  # Will use .c patterns
    '.hpp': [],  # Will use .cpp patterns
    '.cs': [
        (r'^(?:public|private|protected|internal)?\s*(?:static\s+)?(?:partial\s+)?class\s+(\w+)', 'class'),
        (r'^(?:public|private|protected|internal)?\s*(?:static\s+)?interface\s+(\w+)', 'interface'),
        (r'^(?:public|private|protected|internal)?\s*(?:static\s+)?struct\s+(\w+)', 'struct'),
        (r'^(?:public|private|protected|internal)?\s*enum\s+(\w+)', 'enum'),
        (r'^\s+(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?[\w<>,\[\]\s]+\s+(\w+)\s*\(', 'method'),
        (r'^\s+(?:public|private|protected|internal)?\s*(?:static\s+)?[\w<>,\[\]\s]+\s+(\w+)\s*\{?\s*get', 'property'),
    ],
}

# Inheritance for similar languages
SYMBOL_PATTERNS['.tsx'] = SYMBOL_PATTERNS['.ts']
SYMBOL_PATTERNS['.jsx'] = SYMBOL_PATTERNS['.js']
SYMBOL_PATTERNS['.h'] = SYMBOL_PATTERNS['.c']
SYMBOL_PATTERNS['.hpp'] = SYMBOL_PATTERNS['.cpp']
SYMBOL_PATTERNS['.cc'] = SYMBOL_PATTERNS['.cpp']
SYMBOL_PATTERNS['.cxx'] = SYMBOL_PATTERNS['.cpp']


def get_file_symbols(repo_path: Path, file_path: str) -> list[Symbol]:
    """
    Extract symbols (functions, classes, etc.) from a source file.
    
    Supports: Python, JavaScript, TypeScript, Rust, Go, Java, Ruby, C, C++, C#
    
    Args:
        repo_path: Repository root path
        file_path: Relative path to file
        
    Returns:
        List of Symbol objects
    """
    full_path = repo_path / file_path
    
    if not full_path.exists():
        return []
    
    ext = Path(file_path).suffix.lower()
    patterns = SYMBOL_PATTERNS.get(ext, [])
    
    if not patterns:
        return []
    
    symbols = []
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern, symbol_type in patterns:
                match = re.match(pattern, line)
                if match:
                    symbols.append(Symbol(
                        name=match.group(1),
                        type=symbol_type,
                        line=line_num,
                        signature=line.strip()[:100],
                    ))
                    break  # Only match first pattern per line
                    
    except Exception:
        pass
    
    return symbols


def find_references(
    repo_path: Path,
    symbol: str,
    file_extensions: Optional[list[str]] = None,
    max_results: int = 50,
) -> list[Reference]:
    """
    Find all references to a symbol across the codebase.
    
    Uses word-boundary matching to avoid partial matches.
    
    Args:
        repo_path: Repository root path
        symbol: Symbol name to search for
        file_extensions: Extensions to search (default: common code files)
        max_results: Maximum number of results
        
    Returns:
        List of Reference objects
    """
    extensions = file_extensions or [
        '.py', '.js', '.ts', '.tsx', '.jsx',
        '.rs', '.go', '.java', '.rb', '.cs',
        '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
    ]
    
    references = []
    # Word boundary pattern
    pattern = re.compile(r'\b' + re.escape(symbol) + r'\b')
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                   ['node_modules', '__pycache__', 'target', 'build', 'dist', 'venv', '.git']]
        
        for file in files:
            if not any(file.endswith(ext) for ext in extensions):
                continue
            
            file_path = Path(root) / file
            rel_path = file_path.relative_to(repo_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        match = pattern.search(line)
                        if match:
                            references.append(Reference(
                                file=str(rel_path),
                                line=line_num,
                                column=match.start() + 1,
                                context=line.strip()[:200],
                            ))
                            
                            if len(references) >= max_results:
                                return references
                                
            except Exception:
                continue
    
    return references


if __name__ == "__main__":
    # Test the tools
    import tempfile
    
    # Create a test file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test file
        test_file = tmppath / "test.py"
        test_file.write_text("""def hello():
    print("Hello")

def goodbye():
    print("Goodbye")
""")
        
        # Test diff
        diff = """--- a/test.py
+++ b/test.py
@@ -1,5 +1,8 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
+
+def greet(name):
+    print(f"Hello, {name}!")
 
 def goodbye():
     print("Goodbye")
"""
        
        result = apply_patch(tmppath, "test.py", diff)
        print(f"Patch result: {result}")
        print(f"\nPatched content:\n{test_file.read_text()}")

