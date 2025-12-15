"""
Codebase Query Script

Search your embedded codebase using natural language questions.
Returns the most relevant code chunks.
"""

import argparse
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np


class Embedder:
    """Wrapper for Snowflake Arctic Embed model."""
    
    def __init__(self):
        print("Loading Snowflake Arctic Embed model...")
        self.model = SentenceTransformer("Snowflake/snowflake-arctic-embed-xs")
        print("Model loaded!\n")
    
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (with query prompt)."""
        return self.model.encode([query], prompt_name="query", convert_to_numpy=True)[0].tolist()


def format_result(result: dict, index: int, show_content: bool = True) -> str:
    """Format a single search result for display."""
    metadata = result['metadata']
    filepath = metadata['filepath']
    start_line = metadata['start_line']
    end_line = metadata['end_line']
    chunk_type = metadata['chunk_type']
    name = metadata['name']
    file_ext = metadata.get('file_extension', 'unknown')
    file_type = metadata.get('file_type', 'unknown')
    distance = result['distance']
    similarity = 1 - distance  # ChromaDB returns distances, not similarities
    
    output = []
    output.append(f"{'='*60}")
    output.append(f"Result #{index + 1} (similarity: {similarity:.3f}) [{file_type}]")
    output.append(f"File: {filepath}")
    output.append(f"Lines: {start_line}-{end_line} | Type: {chunk_type} | Name: {name} | Ext: {file_ext}")
    output.append(f"{'='*60}")
    
    if show_content:
        content = result['document']
        # Add line numbers to content
        lines = content.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            numbered_lines.append(f"{line_num:4d} | {line}")
        output.append('\n'.join(numbered_lines))
    
    return '\n'.join(output)


def search(query: str, embedder: Embedder, collection: chromadb.Collection,
           n_results: int = 5, show_content: bool = True,
           file_types: list[str] = None, extensions: list[str] = None) -> None:
    """Search the codebase and display results."""
    filter_info = []
    if file_types:
        filter_info.append(f"file_types={file_types}")
    if extensions:
        filter_info.append(f"extensions={extensions}")
    
    filter_str = f" (filters: {', '.join(filter_info)})" if filter_info else ""
    print(f"Searching for: \"{query}\"{filter_str}\n")
    
    # Embed the query
    query_embedding = embedder.embed_query(query)
    
    # Build metadata filter
    where_filter = None
    if file_types and extensions:
        where_filter = {
            "$and": [
                {"file_type": {"$in": file_types}},
                {"file_extension": {"$in": extensions}}
            ]
        }
    elif file_types:
        if len(file_types) == 1:
            where_filter = {"file_type": file_types[0]}
        else:
            where_filter = {"file_type": {"$in": file_types}}
    elif extensions:
        if len(extensions) == 1:
            where_filter = {"file_extension": extensions[0]}
        else:
            where_filter = {"file_extension": {"$in": extensions}}
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=['documents', 'metadatas', 'distances']
    )
    
    if not results['ids'][0]:
        print("No results found.")
        return
    
    # Format and display results
    for i in range(len(results['ids'][0])):
        result = {
            'id': results['ids'][0][i],
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        }
        print(format_result(result, i, show_content))
        print()


def interactive_mode(embedder: Embedder, collection: chromadb.Collection,
                     n_results: int, show_content: bool,
                     file_types: list[str] = None, extensions: list[str] = None) -> None:
    """Run in interactive mode, accepting queries until exit."""
    filter_info = []
    if file_types:
        filter_info.append(f"file_types={file_types}")
    if extensions:
        filter_info.append(f"extensions={extensions}")
    filter_str = f" | Filters: {', '.join(filter_info)}" if filter_info else ""
    
    print(f"Interactive mode.{filter_str}")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        
        if not query:
            continue
        if query.lower() in ('exit', 'quit', 'q'):
            print("Exiting.")
            break
        
        search(query, embedder, collection, n_results, show_content, file_types, extensions)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Search your embedded codebase using natural language."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs='?',
        default=None,
        help="The search query (omit for interactive mode)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="Path to the ChromaDB database (default: ./chroma_db)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="codebase",
        help="Name of the ChromaDB collection (default: codebase)"
    )
    parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Only show file paths and metadata, not the actual code"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode (keep accepting queries)"
    )
    parser.add_argument(
        "--file-type",
        type=str,
        action="append",
        dest="file_types",
        choices=["code", "config", "data", "docs", "log", "other"],
        help="Filter by file type (can be specified multiple times). Options: code, config, data, docs, log, other"
    )
    parser.add_argument(
        "--ext",
        type=str,
        action="append",
        dest="extensions",
        help="Filter by file extension, e.g. --ext .py --ext .ts (can be specified multiple times)"
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Shortcut to filter only code files (equivalent to --file-type code)"
    )
    
    args = parser.parse_args()
    
    # Handle --code-only shortcut
    if args.code_only:
        args.file_types = ["code"]
    
    # Check if database exists
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database not found at '{args.db_path}'")
        print("Run embed_codebase.py first to create the database.")
        sys.exit(1)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=args.db_path)
    
    try:
        collection = client.get_collection(args.collection)
    except ValueError:
        print(f"Error: Collection '{args.collection}' not found in database.")
        print("Run embed_codebase.py first to create the collection.")
        sys.exit(1)
    
    # Show collection stats
    count = collection.count()
    print(f"Loaded collection '{args.collection}' with {count} chunks\n")
    
    # Initialize embedder
    embedder = Embedder()
    
    show_content = not args.no_content
    
    # Normalize extensions (ensure they start with .)
    extensions = None
    if args.extensions:
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions]
    
    # Run query or interactive mode
    if args.interactive or args.query is None:
        interactive_mode(embedder, collection, args.num_results, show_content,
                        args.file_types, extensions)
    else:
        search(args.query, embedder, collection, args.num_results, show_content,
               args.file_types, extensions)


if __name__ == "__main__":
    main()


