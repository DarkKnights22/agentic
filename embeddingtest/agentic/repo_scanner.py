"""
Repository Scanner

Scans a repository to generate file tree structure, identify key entry points,
detect test directories, and provide structural context for the Master Agent.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# Directories to exclude from scanning
EXCLUDED_DIRS = {
    'venv', 'node_modules', '.git', '__pycache__', 
    'dist', 'build', '.idea', '.vscode', 'env',
    '.env', 'site-packages', '.tox', '.pytest_cache',
    '.mypy_cache', 'egg-info', '.eggs', '.agentic',
    'chroma_db', '.cache', 'coverage', '.nyc_output',
    '.next', '.nuxt', '.svelte-kit', 'target',
}

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    '.pyc', '.pyo', '.so', '.dll', '.exe', '.bin',
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
    '.pdf', '.zip', '.tar', '.gz', '.rar', '.7z',
    '.mp3', '.mp4', '.wav', '.avi', '.mov',
    '.ttf', '.woff', '.woff2', '.eot',
    '.db', '.sqlite', '.pickle', '.pkl',
    '.o', '.a', '.lib', '.obj', '.class',
    '.onnx', '.onnx_data', '.pyd', '.pdb',
    '.lock', '.lockb',
}

# Entry point files (in priority order)
ENTRY_POINT_FILES = [
    # Python
    'main.py', 'app.py', 'run.py', '__main__.py', 'cli.py',
    'setup.py', 'pyproject.toml', 'setup.cfg',
    # JavaScript/TypeScript  
    'index.js', 'index.ts', 'main.js', 'main.ts',
    'app.js', 'app.ts', 'server.js', 'server.ts',
    'package.json',
    # Other
    'Makefile', 'Dockerfile', 'docker-compose.yml',
    'Cargo.toml', 'go.mod', 'build.gradle', 'pom.xml',
]

# Test directory patterns
TEST_DIR_PATTERNS = {
    'test', 'tests', 'spec', 'specs', '__tests__',
    'test_', '_test', 'testing',
}

# File type categories
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.vue', '.svelte', '.astro',
}

CONFIG_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.xml', '.properties',
}

DOC_EXTENSIONS = {
    '.md', '.rst', '.txt', '.html', '.htm',
}


@dataclass
class FileInfo:
    """Information about a single file."""
    path: Path
    relative_path: str
    extension: str
    file_type: str  # code, config, docs, other
    size_bytes: int
    line_count: Optional[int] = None


@dataclass 
class RepoStructure:
    """Complete repository structure analysis."""
    root_path: Path
    total_files: int = 0
    total_dirs: int = 0
    files: list[FileInfo] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    file_tree: str = ""
    
    # Stats by type
    code_files: int = 0
    config_files: int = 0
    doc_files: int = 0
    other_files: int = 0


def get_file_type(extension: str) -> str:
    """Categorize file by extension."""
    ext = extension.lower()
    if ext in CODE_EXTENSIONS:
        return 'code'
    elif ext in CONFIG_EXTENSIONS:
        return 'config'
    elif ext in DOC_EXTENSIONS:
        return 'docs'
    else:
        return 'other'


def is_binary_file(filepath: Path) -> bool:
    """Check if a file is binary based on extension."""
    return filepath.suffix.lower() in BINARY_EXTENSIONS


def is_test_directory(name: str) -> bool:
    """Check if a directory name indicates tests."""
    name_lower = name.lower()
    return any(
        name_lower == pattern or 
        name_lower.startswith(pattern) or 
        name_lower.endswith(pattern.lstrip('_'))
        for pattern in TEST_DIR_PATTERNS
    )


def count_lines(filepath: Path) -> Optional[int]:
    """Count lines in a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return None


class RepoScanner:
    """
    Scans a repository and extracts structural information.
    
    Usage:
        scanner = RepoScanner("/path/to/repo")
        structure = scanner.scan()
        print(structure.file_tree)
    """
    
    def __init__(
        self,
        root_path: str | Path,
        max_depth: int = 10,
        include_line_counts: bool = False,
        max_files: int = 5000,
    ):
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.include_line_counts = include_line_counts
        self.max_files = max_files
        
        if not self.root_path.exists():
            raise ValueError(f"Path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.root_path}")
    
    def scan(self) -> RepoStructure:
        """
        Perform full repository scan.
        
        Returns:
            RepoStructure with all analysis results
        """
        structure = RepoStructure(root_path=self.root_path)
        
        # Build file tree and collect file info
        tree_lines = [f"{self.root_path.name}/"]
        self._scan_directory(
            self.root_path, 
            structure, 
            tree_lines, 
            prefix="",
            depth=0
        )
        
        structure.file_tree = "\n".join(tree_lines)
        
        # Identify entry points
        structure.entry_points = self._find_entry_points(structure.files)
        
        return structure
    
    def _scan_directory(
        self,
        path: Path,
        structure: RepoStructure,
        tree_lines: list[str],
        prefix: str,
        depth: int,
    ):
        """Recursively scan a directory."""
        if depth > self.max_depth:
            return
        
        if structure.total_files >= self.max_files:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return
        
        # Filter items
        dirs = []
        files = []
        
        for item in items:
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                if item.name not in EXCLUDED_DIRS:
                    dirs.append(item)
            elif item.is_file():
                if not is_binary_file(item):
                    files.append(item)
        
        total_items = len(dirs) + len(files)
        
        for i, item in enumerate(dirs + files):
            is_last = (i == total_items - 1)
            connector = "└── " if is_last else "├── "
            
            if item.is_dir():
                structure.total_dirs += 1
                
                # Check if it's a test directory
                if is_test_directory(item.name):
                    rel_path = str(item.relative_to(self.root_path))
                    if rel_path not in structure.test_dirs:
                        structure.test_dirs.append(rel_path)
                
                tree_lines.append(f"{prefix}{connector}{item.name}/")
                
                # Recurse with updated prefix
                extension = "    " if is_last else "│   "
                self._scan_directory(
                    item,
                    structure,
                    tree_lines,
                    prefix + extension,
                    depth + 1
                )
            else:
                structure.total_files += 1
                
                # Collect file info
                ext = item.suffix.lower()
                file_type = get_file_type(ext)
                
                try:
                    size = item.stat().st_size
                except OSError:
                    size = 0
                
                line_count = None
                if self.include_line_counts and file_type == 'code':
                    line_count = count_lines(item)
                
                file_info = FileInfo(
                    path=item,
                    relative_path=str(item.relative_to(self.root_path)),
                    extension=ext,
                    file_type=file_type,
                    size_bytes=size,
                    line_count=line_count,
                )
                structure.files.append(file_info)
                
                # Update type counts
                if file_type == 'code':
                    structure.code_files += 1
                elif file_type == 'config':
                    structure.config_files += 1
                elif file_type == 'docs':
                    structure.doc_files += 1
                else:
                    structure.other_files += 1
                
                tree_lines.append(f"{prefix}{connector}{item.name}")
    
    def _find_entry_points(self, files: list[FileInfo]) -> list[str]:
        """Identify likely entry point files."""
        entry_points = []
        file_names = {f.relative_path: f for f in files}
        
        # Check for known entry point files
        for ep_name in ENTRY_POINT_FILES:
            for rel_path, file_info in file_names.items():
                if rel_path == ep_name or rel_path.endswith(f"/{ep_name}"):
                    if rel_path not in entry_points:
                        entry_points.append(rel_path)
        
        # Also check for src/ directory entry points
        for rel_path in file_names:
            if rel_path.startswith("src/"):
                base_name = Path(rel_path).name
                if base_name in ENTRY_POINT_FILES:
                    if rel_path not in entry_points:
                        entry_points.append(rel_path)
        
        return entry_points
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the repository."""
        structure = self.scan()
        
        lines = [
            f"Repository: {structure.root_path.name}",
            f"Total files: {structure.total_files}",
            f"Total directories: {structure.total_dirs}",
            "",
            "File breakdown:",
            f"  Code files: {structure.code_files}",
            f"  Config files: {structure.config_files}",
            f"  Documentation: {structure.doc_files}",
            f"  Other: {structure.other_files}",
        ]
        
        if structure.entry_points:
            lines.append("")
            lines.append("Entry points:")
            for ep in structure.entry_points[:10]:  # Limit to 10
                lines.append(f"  - {ep}")
        
        if structure.test_dirs:
            lines.append("")
            lines.append("Test directories:")
            for td in structure.test_dirs[:5]:  # Limit to 5
                lines.append(f"  - {td}/")
        
        return "\n".join(lines)


def scan_repo(path: str | Path, **kwargs) -> RepoStructure:
    """
    Convenience function to scan a repository.
    
    Args:
        path: Path to the repository root
        **kwargs: Additional arguments for RepoScanner
        
    Returns:
        RepoStructure with analysis results
    """
    scanner = RepoScanner(path, **kwargs)
    return scanner.scan()


if __name__ == "__main__":
    import sys
    
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    scanner = RepoScanner(path)
    
    print(scanner.get_summary())
    print("\n" + "="*50 + "\n")
    print("File Tree:")
    print(scanner.scan().file_tree)

