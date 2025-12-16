"""
Memory system for Documentation Agent with Long-Term Memory.

Provides schema-driven persistent storage with bounded retrieval.
All memory entries are stored as typed JSON files in .agentic/memory/.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from enum import Enum
import json
import hashlib


class MemoryEntryType(str, Enum):
    ARCHITECTURE = "architecture"
    COMPONENT = "component"
    FILE = "file"
    DATA_MODEL = "data_model"
    FLOW = "flow"
    CROSS_CUTTING = "cross_cutting"


# =============================================================================
# Memory Schemas
# =============================================================================

@dataclass
class ArchitectureOverview:
    """High-level system architecture overview."""
    system_name: str
    purpose: str
    tech_stack: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    major_components: list[str] = field(default_factory=list)
    architectural_style: str = ""
    key_abstractions: list[str] = field(default_factory=list)
    component_boundaries: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0
    verified_by_files: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class ComponentSummary:
    """Detailed summary of a single component."""
    component_id: str
    component_name: str
    root_path: str
    responsibility: str
    key_files: list[str] = field(default_factory=list)
    public_interfaces: list[dict] = field(default_factory=list)  # {name, type, signature, file}
    dependencies: list[str] = field(default_factory=list)  # component_ids this depends on
    dependents: list[str] = field(default_factory=list)  # component_ids that depend on this
    design_patterns_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    explored_files: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class FileRole:
    """Role and metadata for a single file."""
    file_id: str
    file_path: str
    responsibility: str
    layer: str = ""  # api, service, data, util, etc.
    component: str = ""  # component_id this file belongs to
    key_exports: list[str] = field(default_factory=list)
    key_functions: list[dict] = field(default_factory=list)  # {name, signature, purpose}
    key_classes: list[dict] = field(default_factory=list)  # {name, purpose, methods}
    imports_from: list[str] = field(default_factory=list)  # file paths imported
    imported_by: list[str] = field(default_factory=list)  # file paths that import this
    last_read_hash: str = ""  # for change detection
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class DataModel:
    """Data model / schema definition."""
    model_id: str
    model_name: str
    model_type: str = ""  # dataclass, orm, schema, interface, etc.
    file_path: str = ""
    fields: list[dict] = field(default_factory=list)  # {name, type, description}
    relationships: list[dict] = field(default_factory=list)  # {target_model, relation_type}
    used_by_components: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class RuntimeFlow:
    """A runtime flow / sequence through the system."""
    flow_id: str
    flow_name: str
    description: str = ""
    trigger: str = ""  # What initiates this flow
    steps: list[dict] = field(default_factory=list)  # {component, action, next}
    components_involved: list[str] = field(default_factory=list)
    data_models_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class CrossCuttingConcern:
    """A concern that spans multiple components."""
    concern_id: str
    concern_name: str
    concern_type: str = ""  # auth, logging, config, error_handling, etc.
    description: str = ""
    implementation_pattern: str = ""
    files_involved: list[str] = field(default_factory=list)
    components_affected: list[str] = field(default_factory=list)
    key_abstractions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# =============================================================================
# Memory Query Contract
# =============================================================================

@dataclass
class MemoryQuery:
    """
    Bounded memory query specification.

    CRITICAL: max_entries has NO DEFAULT - forces explicit thinking about load size.
    """
    query_type: str  # "architecture" | "component" | "file" | "flow" | "data_model" | "cross_cutting"
    max_entries: int  # REQUIRED - hard cap
    filter_by: dict = field(default_factory=dict)  # e.g., {"component": "auth", "layer": "api"}
    required_fields: list[str] = field(default_factory=list)  # Only return these fields
    min_confidence: float = 0.0  # Filter uncertain entries
    sort_by: str = "relevance"  # "relevance" | "recency" | "confidence"


@dataclass
class MemoryIndex:
    """Index for fast lookups and phase tracking."""
    version: str = "1.0"
    current_phase: str = "not_started"
    phase_progress: dict = field(default_factory=dict)
    components_discovered: list[str] = field(default_factory=list)
    components_explored: list[str] = field(default_factory=list)
    cross_cutting_found: list[str] = field(default_factory=list)
    file_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# =============================================================================
# Memory Manager
# =============================================================================

class MemoryQueryError(Exception):
    """Raised when a memory query violates the retrieval contract."""
    pass


class MemoryManager:
    """
    Manages persistent memory storage with bounded retrieval.

    All memory is stored in .agentic/memory/ as JSON files.
    Enforces the retrieval contract - all queries must be bounded.
    """

    # Type to schema class mapping
    SCHEMA_MAP = {
        MemoryEntryType.ARCHITECTURE: ArchitectureOverview,
        MemoryEntryType.COMPONENT: ComponentSummary,
        MemoryEntryType.FILE: FileRole,
        MemoryEntryType.DATA_MODEL: DataModel,
        MemoryEntryType.FLOW: RuntimeFlow,
        MemoryEntryType.CROSS_CUTTING: CrossCuttingConcern,
    }

    # Directory names for each type
    TYPE_DIRS = {
        MemoryEntryType.ARCHITECTURE: "architecture",
        MemoryEntryType.COMPONENT: "components",
        MemoryEntryType.FILE: "files",
        MemoryEntryType.DATA_MODEL: "data_models",
        MemoryEntryType.FLOW: "flows",
        MemoryEntryType.CROSS_CUTTING: "cross_cutting",
    }

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.memory_path = self.repo_path / ".agentic" / "memory"
        self._ensure_directory_structure()
        self._index = self._load_index()

    def _ensure_directory_structure(self):
        """Create memory directory structure if it doesn't exist."""
        self.memory_path.mkdir(parents=True, exist_ok=True)
        for type_dir in self.TYPE_DIRS.values():
            (self.memory_path / type_dir).mkdir(exist_ok=True)

    def _load_index(self) -> MemoryIndex:
        """Load or create the memory index."""
        index_path = self.memory_path / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding='utf-8'))
                return MemoryIndex(**data)
            except (json.JSONDecodeError, TypeError):
                return MemoryIndex()
        return MemoryIndex()

    def _save_index(self):
        """Save the memory index to disk."""
        self._index.last_updated = datetime.utcnow().isoformat() + "Z"
        index_path = self.memory_path / "index.json"
        index_path.write_text(json.dumps(asdict(self._index), indent=2), encoding='utf-8')

    def _get_entry_path(self, entry_type: MemoryEntryType, entry_id: str) -> Path:
        """Get the file path for a memory entry."""
        type_dir = self.TYPE_DIRS[entry_type]
        return self.memory_path / type_dir / f"{entry_id}.json"

    def _hash_file(self, file_path: str) -> str:
        """Generate hash of a file for change detection."""
        full_path = self.repo_path / file_path
        if not full_path.exists():
            return ""
        try:
            content = full_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    # -------------------------------------------------------------------------
    # Storage Operations
    # -------------------------------------------------------------------------

    def store_architecture(self, overview: ArchitectureOverview) -> None:
        """Store the architecture overview."""
        overview.updated_at = datetime.utcnow().isoformat() + "Z"
        path = self._get_entry_path(MemoryEntryType.ARCHITECTURE, "overview")
        path.write_text(json.dumps(asdict(overview), indent=2), encoding='utf-8')
        self._index.current_phase = "architecture_discovered"
        self._save_index()

    def store_component(self, component: ComponentSummary) -> None:
        """Store a component summary."""
        component.updated_at = datetime.utcnow().isoformat() + "Z"
        path = self._get_entry_path(MemoryEntryType.COMPONENT, component.component_id)
        path.write_text(json.dumps(asdict(component), indent=2), encoding='utf-8')

        if component.component_id not in self._index.components_discovered:
            self._index.components_discovered.append(component.component_id)
        self._save_index()

    def store_file_role(self, file_role: FileRole) -> None:
        """Store a file role entry."""
        file_role.updated_at = datetime.utcnow().isoformat() + "Z"
        # Update hash for change detection
        file_role.last_read_hash = self._hash_file(file_role.file_path)
        path = self._get_entry_path(MemoryEntryType.FILE, file_role.file_id)
        path.write_text(json.dumps(asdict(file_role), indent=2), encoding='utf-8')
        self._index.file_count = len(list((self.memory_path / "files").glob("*.json")))
        self._save_index()

    def store_data_model(self, model: DataModel) -> None:
        """Store a data model entry."""
        model.updated_at = datetime.utcnow().isoformat() + "Z"
        path = self._get_entry_path(MemoryEntryType.DATA_MODEL, model.model_id)
        path.write_text(json.dumps(asdict(model), indent=2), encoding='utf-8')
        self._save_index()

    def store_flow(self, flow: RuntimeFlow) -> None:
        """Store a runtime flow entry."""
        flow.updated_at = datetime.utcnow().isoformat() + "Z"
        path = self._get_entry_path(MemoryEntryType.FLOW, flow.flow_id)
        path.write_text(json.dumps(asdict(flow), indent=2), encoding='utf-8')
        self._save_index()

    def store_cross_cutting(self, concern: CrossCuttingConcern) -> None:
        """Store a cross-cutting concern entry."""
        concern.updated_at = datetime.utcnow().isoformat() + "Z"
        path = self._get_entry_path(MemoryEntryType.CROSS_CUTTING, concern.concern_id)
        path.write_text(json.dumps(asdict(concern), indent=2), encoding='utf-8')

        if concern.concern_id not in self._index.cross_cutting_found:
            self._index.cross_cutting_found.append(concern.concern_id)
        self._save_index()

    def store_discovery(self, entry_type: str, data: dict) -> None:
        """
        Generic storage method for discoveries.
        Validates and stores based on entry type.
        """
        try:
            mem_type = MemoryEntryType(entry_type)
        except ValueError:
            raise ValueError(f"Unknown entry type: {entry_type}")

        schema_class = self.SCHEMA_MAP[mem_type]

        if mem_type == MemoryEntryType.ARCHITECTURE:
            entry = ArchitectureOverview(**data)
            self.store_architecture(entry)
        elif mem_type == MemoryEntryType.COMPONENT:
            entry = ComponentSummary(**data)
            self.store_component(entry)
        elif mem_type == MemoryEntryType.FILE:
            entry = FileRole(**data)
            self.store_file_role(entry)
        elif mem_type == MemoryEntryType.DATA_MODEL:
            entry = DataModel(**data)
            self.store_data_model(entry)
        elif mem_type == MemoryEntryType.FLOW:
            entry = RuntimeFlow(**data)
            self.store_flow(entry)
        elif mem_type == MemoryEntryType.CROSS_CUTTING:
            entry = CrossCuttingConcern(**data)
            self.store_cross_cutting(entry)

    # -------------------------------------------------------------------------
    # Bounded Retrieval Operations
    # -------------------------------------------------------------------------

    def query_memory(self, query: MemoryQuery) -> list[dict]:
        """
        Execute a bounded memory query.

        ENFORCES RETRIEVAL CONTRACT:
        - max_entries must be explicitly set (no default)
        - max_entries must be reasonable (<=100)
        - required_fields filters response payload

        Raises MemoryQueryError if contract violated.
        """
        # Validate query bounds
        if query.max_entries <= 0:
            raise MemoryQueryError("max_entries must be positive")
        if query.max_entries > 100:
            raise MemoryQueryError(f"max_entries={query.max_entries} exceeds limit of 100")

        try:
            mem_type = MemoryEntryType(query.query_type)
        except ValueError:
            raise MemoryQueryError(f"Unknown query_type: {query.query_type}")

        # Load entries from disk
        type_dir = self.memory_path / self.TYPE_DIRS[mem_type]
        entries = []

        for entry_file in type_dir.glob("*.json"):
            try:
                entry_data = json.loads(entry_file.read_text(encoding='utf-8'))
                entries.append(entry_data)
            except (json.JSONDecodeError, IOError):
                continue

        # Apply filters
        filtered = self._apply_filters(entries, query.filter_by)

        # Apply confidence filter
        if query.min_confidence > 0:
            filtered = [e for e in filtered if e.get("confidence", 0) >= query.min_confidence]

        # Sort
        filtered = self._sort_entries(filtered, query.sort_by)

        # Apply max_entries limit
        filtered = filtered[:query.max_entries]

        # Project required fields only
        if query.required_fields:
            filtered = self._project_fields(filtered, query.required_fields)

        return filtered

    def _apply_filters(self, entries: list[dict], filter_by: dict) -> list[dict]:
        """Apply filter conditions to entries."""
        if not filter_by:
            return entries

        result = []
        for entry in entries:
            match = True
            for key, value in filter_by.items():
                entry_value = entry.get(key)
                if entry_value != value:
                    # Also check nested matches for lists
                    if isinstance(entry_value, list) and value not in entry_value:
                        match = False
                        break
                    elif not isinstance(entry_value, list):
                        match = False
                        break
            if match:
                result.append(entry)
        return result

    def _sort_entries(self, entries: list[dict], sort_by: str) -> list[dict]:
        """Sort entries by specified criteria."""
        if sort_by == "confidence":
            return sorted(entries, key=lambda x: x.get("confidence", 0), reverse=True)
        elif sort_by == "recency":
            return sorted(entries, key=lambda x: x.get("updated_at", ""), reverse=True)
        # Default: relevance (currently just returns as-is, could be enhanced)
        return entries

    def _project_fields(self, entries: list[dict], fields: list[str]) -> list[dict]:
        """Project only the required fields from entries."""
        result = []
        for entry in entries:
            projected = {}
            for field in fields:
                if field in entry:
                    projected[field] = entry[field]
            result.append(projected)
        return result

    def estimate_token_usage(self, query: MemoryQuery) -> int:
        """
        Estimate token usage for a query before executing.
        Helps agent decide if query is too large.
        """
        # Execute query without token limit awareness
        results = self.query_memory(query)
        # Rough estimate: 4 chars per token
        json_str = json.dumps(results)
        return len(json_str) // 4

    # -------------------------------------------------------------------------
    # Index and Phase Operations
    # -------------------------------------------------------------------------

    def get_index(self) -> MemoryIndex:
        """Get the current memory index."""
        return self._index

    def update_phase(self, phase: str, progress: Optional[dict] = None):
        """Update the current phase and progress."""
        self._index.current_phase = phase
        if progress:
            self._index.phase_progress.update(progress)
        self._save_index()

    def mark_component_explored(self, component_id: str):
        """Mark a component as fully explored."""
        if component_id not in self._index.components_explored:
            self._index.components_explored.append(component_id)
        self._save_index()

    def get_unexplored_components(self) -> list[str]:
        """Get list of components discovered but not yet explored."""
        return [c for c in self._index.components_discovered
                if c not in self._index.components_explored]

    # -------------------------------------------------------------------------
    # Change Detection
    # -------------------------------------------------------------------------

    def get_changed_files(self) -> list[str]:
        """Get list of files whose content has changed since last read."""
        changed = []
        files_dir = self.memory_path / "files"

        for entry_file in files_dir.glob("*.json"):
            try:
                entry_data = json.loads(entry_file.read_text(encoding='utf-8'))
                file_path = entry_data.get("file_path", "")
                old_hash = entry_data.get("last_read_hash", "")
                current_hash = self._hash_file(file_path)

                if old_hash and current_hash and old_hash != current_hash:
                    changed.append(file_path)
            except (json.JSONDecodeError, IOError):
                continue

        return changed

    def get_memory_summary(self) -> dict:
        """Get a summary of what's in memory."""
        return {
            "current_phase": self._index.current_phase,
            "components_discovered": len(self._index.components_discovered),
            "components_explored": len(self._index.components_explored),
            "files_documented": self._index.file_count,
            "cross_cutting_concerns": len(self._index.cross_cutting_found),
            "last_updated": self._index.last_updated,
        }

    def clear_memory(self):
        """Clear all memory (use with caution)."""
        import shutil
        if self.memory_path.exists():
            shutil.rmtree(self.memory_path)
        self._ensure_directory_structure()
        self._index = MemoryIndex()
        self._save_index()
