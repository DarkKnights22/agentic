"""
Codebase Embedding Script

Recursively scans a directory, chunks code files intelligently,
generates embeddings using MiniLM, and stores them in ChromaDB.
"""

import os
import sys
import ast
import argparse
from pathlib import Path
from typing import Generator

from light_embed import TextEmbedding
import chromadb
import numpy as np

# Directories to exclude from scanning
EXCLUDED_DIRS = {
    'venv', 'node_modules', '.git', '__pycache__', 
    'dist', 'build', '.idea', '.vscode', 'env',
    '.env', 'site-packages', '.tox', '.pytest_cache',
    '.mypy_cache', 'egg-info', '.eggs'
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
    '.onnx', '.onnx_data', '.pyd', '.pdb'
}

# File type categories for filtering
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.vue', '.svelte', '.astro'
}

CONFIG_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env',
    '.xml', '.properties'
}

DATA_EXTENSIONS = {
    '.csv', '.tsv', '.sql'
}

DOC_EXTENSIONS = {
    '.md', '.rst', '.txt', '.html', '.htm'
}

LOG_EXTENSIONS = {
    '.log'
}


def get_file_type(extension: str) -> str:
    """Categorize file by extension for filtering."""
    ext = extension.lower()
    if ext in CODE_EXTENSIONS:
        return 'code'
    elif ext in CONFIG_EXTENSIONS:
        return 'config'
    elif ext in DATA_EXTENSIONS:
        return 'data'
    elif ext in DOC_EXTENSIONS:
        return 'docs'
    elif ext in LOG_EXTENSIONS:
        return 'log'
    else:
        return 'other'

# Maximum chunk size in characters (roughly ~2000 tokens)
MAX_CHUNK_SIZE = 8000
# Lines per chunk for non-Python files
LINES_PER_CHUNK = 150


class Embedder:
    """Wrapper for MiniLM embedding model using light_embed."""
    
    def __init__(self):
        print("Loading MiniLM embedding model...")
        self.model = TextEmbedding('onnx-models/all-MiniLM-L6-v2-onnx')
        print("Model loaded successfully!")
    
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents."""
        return np.array(self.model.encode(texts))
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return np.array(self.model.encode([query])[0])


def is_binary_file(filepath: Path) -> bool:
    """Check if a file is binary based on extension or content."""
    if filepath.suffix.lower() in BINARY_EXTENSIONS:
        return True
    
    # Check for null bytes in first 1024 bytes
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\x00' in chunk:
                return True
    except Exception:
        return True
    
    return False


def scan_directory(root_path: Path) -> Generator[Path, None, None]:
    """Recursively scan directory for text files, excluding certain directories."""
    for item in root_path.iterdir():
        if item.is_dir():
            if item.name in EXCLUDED_DIRS or item.name.startswith('.'):
                continue
            yield from scan_directory(item)
        elif item.is_file():
            if not is_binary_file(item):
                yield item


def extract_python_chunks(filepath: Path, content: str) -> list[dict]:
    """Use AST to extract functions and classes from Python files."""
    chunks = []
    file_ext = filepath.suffix.lower()
    file_type = get_file_type(file_ext)
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Fall back to line-based chunking if parsing fails
        return chunk_by_lines(filepath, content)
    
    lines = content.split('\n')
    
    # Track which lines are covered by functions/classes
    covered_lines = set()
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno - 1  # 0-indexed
            end_line = node.end_lineno if node.end_lineno else start_line + 1
            
            # Include decorators
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno - 1
            
            chunk_lines = lines[start_line:end_line]
            chunk_content = '\n'.join(chunk_lines)
            
            # Skip if chunk is too large
            if len(chunk_content) > MAX_CHUNK_SIZE:
                # For large classes, we'll let the class be chunked by lines
                continue
            
            covered_lines.update(range(start_line, end_line))
            
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append({
                'content': chunk_content,
                'filepath': str(filepath),
                'start_line': start_line + 1,  # 1-indexed for display
                'end_line': end_line,
                'chunk_type': chunk_type,
                'name': node.name,
                'file_extension': file_ext,
                'file_type': file_type
            })
    
    # Also capture module-level code (imports, constants, etc.)
    uncovered_chunks = []
    current_chunk_start = None
    current_chunk_lines = []
    
    for i, line in enumerate(lines):
        if i not in covered_lines:
            if current_chunk_start is None:
                current_chunk_start = i
            current_chunk_lines.append(line)
        else:
            if current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines).strip()
                if chunk_content:  # Only add non-empty chunks
                    uncovered_chunks.append({
                        'content': chunk_content,
                        'filepath': str(filepath),
                        'start_line': current_chunk_start + 1,
                        'end_line': current_chunk_start + len(current_chunk_lines),
                        'chunk_type': 'module',
                        'name': 'module_level',
                        'file_extension': file_ext,
                        'file_type': file_type
                    })
                current_chunk_start = None
                current_chunk_lines = []
    
    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_content = '\n'.join(current_chunk_lines).strip()
        if chunk_content:
            uncovered_chunks.append({
                'content': chunk_content,
                'filepath': str(filepath),
                'start_line': current_chunk_start + 1,
                'end_line': current_chunk_start + len(current_chunk_lines),
                'chunk_type': 'module',
                'name': 'module_level',
                'file_extension': file_ext,
                'file_type': file_type
            })
    
    chunks.extend(uncovered_chunks)
    return chunks


def chunk_by_lines(filepath: Path, content: str) -> list[dict]:
    """Split content into chunks of approximately LINES_PER_CHUNK lines."""
    lines = content.split('\n')
    chunks = []
    file_ext = filepath.suffix.lower()
    file_type = get_file_type(file_ext)
    
    for i in range(0, len(lines), LINES_PER_CHUNK):
        chunk_lines = lines[i:i + LINES_PER_CHUNK]
        chunk_content = '\n'.join(chunk_lines)
        
        if chunk_content.strip():  # Only add non-empty chunks
            chunks.append({
                'content': chunk_content,
                'filepath': str(filepath),
                'start_line': i + 1,
                'end_line': min(i + LINES_PER_CHUNK, len(lines)),
                'chunk_type': 'lines',
                'name': f'lines_{i+1}_{min(i + LINES_PER_CHUNK, len(lines))}',
                'file_extension': file_ext,
                'file_type': file_type
            })
    
    return chunks


def chunk_file(filepath: Path) -> list[dict]:
    """Read and chunk a file appropriately based on its type."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return []
    
    if not content.strip():
        return []
    
    # Use AST-based chunking for Python files
    if filepath.suffix == '.py':
        return extract_python_chunks(filepath, content)
    else:
        return chunk_by_lines(filepath, content)


def embed_and_store(chunks: list[dict], embedder: Embedder, 
                    collection: chromadb.Collection, batch_size: int = 50):
    """Embed chunks in batches and store in ChromaDB."""
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c['content'] for c in batch]
        
        print(f"  Embedding {i+1}-{min(i+len(batch), total)} of {total}...", end='\r')
        embeddings = embedder.embed_documents(texts)
        print(f"  Embedded {min(i+len(batch), total)}/{total} chunks    ")
        
        # Prepare data for ChromaDB
        ids = [f"{c['filepath']}:{c['start_line']}-{c['end_line']}" for c in batch]
        metadatas = [{
            'filepath': c['filepath'],
            'start_line': c['start_line'],
            'end_line': c['end_line'],
            'chunk_type': c['chunk_type'],
            'name': c['name'],
            'file_extension': c['file_extension'],
            'file_type': c['file_type']
        } for c in batch]
        documents = texts
        
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents
        )


def main():
    parser = argparse.ArgumentParser(
        description="Embed a codebase directory into ChromaDB for semantic search."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory to embed"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="Path to store the ChromaDB database (default: ./chroma_db)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="codebase",
        help="Name of the ChromaDB collection (default: codebase)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and start fresh"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    root_path = Path(args.directory).resolve()
    if not root_path.exists():
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)
    if not root_path.is_dir():
        print(f"Error: '{args.directory}' is not a directory.")
        sys.exit(1)
    
    print(f"Scanning directory: {root_path}")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=args.db_path)
    
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection '{args.collection}'")
        except ValueError:
            pass  # Collection doesn't exist
    
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embedder
    embedder = Embedder()
    
    # Scan and chunk files
    print("\nScanning and chunking files...")
    all_chunks = []
    file_count = 0
    
    for filepath in scan_directory(root_path):
        chunks = chunk_file(filepath)
        if chunks:
            file_count += 1
            all_chunks.extend(chunks)
            print(f"  {filepath.relative_to(root_path)}: {len(chunks)} chunks")
    
    print(f"\nFound {file_count} files with {len(all_chunks)} total chunks")
    
    if not all_chunks:
        print("No content to embed. Exiting.")
        sys.exit(0)
    
    # Embed and store
    print("\nEmbedding and storing in ChromaDB...")
    embed_and_store(all_chunks, embedder, collection)
    
    print(f"\nDone! Embedded {len(all_chunks)} chunks into collection '{args.collection}'")
    print(f"Database stored at: {Path(args.db_path).resolve()}")


if __name__ == "__main__":
    main()


