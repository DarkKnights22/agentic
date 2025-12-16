"""
Documentation writer for Documentation Agent.

Generates markdown documentation and Mermaid diagrams from memory.
Works entirely from stored memory - no file reads.
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .doc_memory import MemoryManager, MemoryQuery


class DocWriter:
    """
    Generates documentation from memory.

    All documentation is generated solely from stored memory entries.
    No file reads are performed in this module.
    """

    def __init__(self, repo_path: Path, memory_manager: "MemoryManager"):
        self.repo_path = Path(repo_path)
        self.memory = memory_manager
        self.docs_path = self.repo_path / ".agentic" / "documentation"
        self.docs_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Main Generation Methods
    # -------------------------------------------------------------------------

    def generate_all(self) -> dict:
        """
        Generate all documentation from memory.

        Returns dict with paths to generated files and any errors.
        """
        results = {
            "generated": [],
            "errors": [],
        }

        # Generate ARCHITECTURE.md
        try:
            arch_path = self.generate_architecture_doc()
            results["generated"].append(str(arch_path))
        except Exception as e:
            results["errors"].append(f"ARCHITECTURE.md: {e}")

        # Generate component docs
        try:
            comp_paths = self.generate_component_docs()
            results["generated"].extend([str(p) for p in comp_paths])
        except Exception as e:
            results["errors"].append(f"Component docs: {e}")

        # Generate DATA_MODELS.md
        try:
            models_path = self.generate_data_models_doc()
            if models_path:
                results["generated"].append(str(models_path))
        except Exception as e:
            results["errors"].append(f"DATA_MODELS.md: {e}")

        # Generate architecture diagram
        try:
            diagram_path = self.generate_architecture_diagram()
            if diagram_path:
                results["generated"].append(str(diagram_path))
        except Exception as e:
            results["errors"].append(f"Architecture diagram: {e}")

        return results

    # -------------------------------------------------------------------------
    # Architecture Documentation
    # -------------------------------------------------------------------------

    def generate_architecture_doc(self) -> Path:
        """Generate ARCHITECTURE.md from memory."""
        from .doc_memory import MemoryQuery

        # Query architecture overview
        arch_results = self.memory.query_memory(MemoryQuery(
            query_type="architecture",
            max_entries=1,
            required_fields=[],  # Get all fields
            min_confidence=0.0,
        ))

        if not arch_results:
            raise ValueError("No architecture overview in memory")

        arch = arch_results[0]

        # Query components for listing
        components = self.memory.query_memory(MemoryQuery(
            query_type="component",
            max_entries=50,
            required_fields=["component_id", "component_name", "responsibility", "root_path"],
            min_confidence=0.0,
        ))

        # Query cross-cutting concerns
        concerns = self.memory.query_memory(MemoryQuery(
            query_type="cross_cutting",
            max_entries=20,
            required_fields=["concern_name", "concern_type", "description"],
            min_confidence=0.0,
        ))

        # Build markdown
        lines = [
            f"# {arch.get('system_name', 'System')}",
            "",
            "## Overview",
            "",
            arch.get('purpose', 'No purpose documented.'),
            "",
            "## Tech Stack",
            "",
        ]

        for tech in arch.get('tech_stack', []):
            lines.append(f"- {tech}")

        lines.extend([
            "",
            "## Architecture",
            "",
            f"**Style**: {arch.get('architectural_style', 'Not documented')}",
            "",
        ])

        if arch.get('key_abstractions'):
            lines.append("**Key Abstractions**:")
            for abstraction in arch['key_abstractions']:
                lines.append(f"- {abstraction}")
            lines.append("")

        lines.extend([
            "## Components",
            "",
        ])

        for comp in components:
            lines.append(f"### {comp.get('component_name', comp.get('component_id', 'Unknown'))}")
            lines.append("")
            lines.append(f"**Path**: `{comp.get('root_path', 'N/A')}`")
            lines.append("")
            lines.append(comp.get('responsibility', 'No description.'))
            lines.append("")

        if arch.get('entry_points'):
            lines.extend([
                "## Entry Points",
                "",
            ])
            for entry in arch['entry_points']:
                lines.append(f"- `{entry}`")
            lines.append("")

        if concerns:
            lines.extend([
                "## Cross-Cutting Concerns",
                "",
            ])
            for concern in concerns:
                lines.append(f"### {concern.get('concern_name', 'Unknown')}")
                lines.append("")
                lines.append(concern.get('description', 'No description.'))
                lines.append("")

        lines.extend([
            "---",
            f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            f"*Confidence: {arch.get('confidence', 0.0):.0%}*",
        ])

        # Write file
        doc_path = self.docs_path / "ARCHITECTURE.md"
        doc_path.write_text("\n".join(lines), encoding='utf-8')

        return doc_path

    # -------------------------------------------------------------------------
    # Component Documentation
    # -------------------------------------------------------------------------

    def generate_component_docs(self) -> list[Path]:
        """Generate documentation for each component."""
        from .doc_memory import MemoryQuery

        # Query all components
        components = self.memory.query_memory(MemoryQuery(
            query_type="component",
            max_entries=50,
            required_fields=[],  # Get all fields
            min_confidence=0.0,
        ))

        comp_docs_path = self.docs_path / "components"
        comp_docs_path.mkdir(exist_ok=True)

        generated = []

        for comp in components:
            try:
                doc_path = self._generate_single_component_doc(comp, comp_docs_path)
                generated.append(doc_path)
            except Exception:
                continue

        return generated

    def _generate_single_component_doc(self, comp: dict, output_dir: Path) -> Path:
        """Generate documentation for a single component."""
        from .doc_memory import MemoryQuery

        component_id = comp.get('component_id', 'unknown')

        # Query file roles for this component
        files = self.memory.query_memory(MemoryQuery(
            query_type="file",
            max_entries=50,
            filter_by={"component": component_id},
            required_fields=["file_path", "responsibility", "layer", "key_exports"],
            min_confidence=0.0,
        ))

        lines = [
            f"# {comp.get('component_name', component_id)}",
            "",
            "## Overview",
            "",
            comp.get('responsibility', 'No description.'),
            "",
            f"**Root Path**: `{comp.get('root_path', 'N/A')}`",
            "",
        ]

        # Key files
        if comp.get('key_files'):
            lines.extend([
                "## Key Files",
                "",
            ])
            for f in comp['key_files']:
                lines.append(f"- `{f}`")
            lines.append("")

        # Public interfaces
        if comp.get('public_interfaces'):
            lines.extend([
                "## Public Interface",
                "",
                "| Name | Type | File |",
                "|------|------|------|",
            ])
            for iface in comp['public_interfaces']:
                name = iface.get('name', 'N/A')
                itype = iface.get('type', 'N/A')
                ifile = iface.get('file', 'N/A')
                lines.append(f"| `{name}` | {itype} | `{ifile}` |")
            lines.append("")

        # Dependencies
        if comp.get('dependencies'):
            lines.extend([
                "## Dependencies",
                "",
                "This component depends on:",
                "",
            ])
            for dep in comp['dependencies']:
                lines.append(f"- `{dep}`")
            lines.append("")

        # Dependents
        if comp.get('dependents'):
            lines.extend([
                "## Used By",
                "",
                "Components that depend on this:",
                "",
            ])
            for dep in comp['dependents']:
                lines.append(f"- `{dep}`")
            lines.append("")

        # Design patterns
        if comp.get('design_patterns_used'):
            lines.extend([
                "## Design Patterns",
                "",
            ])
            for pattern in comp['design_patterns_used']:
                lines.append(f"- {pattern}")
            lines.append("")

        # File breakdown
        if files:
            lines.extend([
                "## Files",
                "",
            ])

            # Group by layer
            by_layer: dict[str, list] = {}
            for f in files:
                layer = f.get('layer', 'other')
                if layer not in by_layer:
                    by_layer[layer] = []
                by_layer[layer].append(f)

            for layer, layer_files in sorted(by_layer.items()):
                lines.append(f"### {layer.title()} Layer")
                lines.append("")
                for f in layer_files:
                    lines.append(f"**`{f.get('file_path', 'N/A')}`**")
                    lines.append(f.get('responsibility', 'No description.'))
                    if f.get('key_exports'):
                        lines.append(f"Exports: {', '.join(f['key_exports'][:5])}")
                    lines.append("")

        lines.extend([
            "---",
            f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
            f"*Confidence: {comp.get('confidence', 0.0):.0%}*",
        ])

        # Write file
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in component_id)
        doc_path = output_dir / f"{safe_name}.md"
        doc_path.write_text("\n".join(lines), encoding='utf-8')

        return doc_path

    # -------------------------------------------------------------------------
    # Data Models Documentation
    # -------------------------------------------------------------------------

    def generate_data_models_doc(self) -> Optional[Path]:
        """Generate DATA_MODELS.md from memory."""
        from .doc_memory import MemoryQuery

        # Query data models
        models = self.memory.query_memory(MemoryQuery(
            query_type="data_model",
            max_entries=100,
            required_fields=[],
            min_confidence=0.0,
        ))

        if not models:
            return None

        lines = [
            "# Data Models",
            "",
            "This document describes the data models used in the system.",
            "",
        ]

        # Group by type
        by_type: dict[str, list] = {}
        for model in models:
            mtype = model.get('model_type', 'other')
            if mtype not in by_type:
                by_type[mtype] = []
            by_type[mtype].append(model)

        for mtype, type_models in sorted(by_type.items()):
            lines.append(f"## {mtype.replace('_', ' ').title()}")
            lines.append("")

            for model in type_models:
                lines.append(f"### {model.get('model_name', 'Unknown')}")
                lines.append("")
                lines.append(f"**File**: `{model.get('file_path', 'N/A')}`")
                lines.append("")

                if model.get('fields'):
                    lines.append("| Field | Type | Description |")
                    lines.append("|-------|------|-------------|")
                    for field in model['fields']:
                        fname = field.get('name', 'N/A')
                        ftype = field.get('type', 'N/A')
                        fdesc = field.get('description', '')
                        lines.append(f"| `{fname}` | `{ftype}` | {fdesc} |")
                    lines.append("")

                if model.get('relationships'):
                    lines.append("**Relationships**:")
                    for rel in model['relationships']:
                        target = rel.get('target_model', 'N/A')
                        rtype = rel.get('relation_type', 'N/A')
                        lines.append(f"- {rtype} `{target}`")
                    lines.append("")

        lines.extend([
            "---",
            f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
        ])

        doc_path = self.docs_path / "DATA_MODELS.md"
        doc_path.write_text("\n".join(lines), encoding='utf-8')

        return doc_path

    # -------------------------------------------------------------------------
    # Mermaid Diagrams
    # -------------------------------------------------------------------------

    def generate_architecture_diagram(self) -> Optional[Path]:
        """Generate a Mermaid architecture diagram."""
        from .doc_memory import MemoryQuery

        # Query architecture
        arch_results = self.memory.query_memory(MemoryQuery(
            query_type="architecture",
            max_entries=1,
            required_fields=["system_name", "major_components"],
            min_confidence=0.0,
        ))

        # Query components
        components = self.memory.query_memory(MemoryQuery(
            query_type="component",
            max_entries=50,
            required_fields=["component_id", "component_name", "dependencies"],
            min_confidence=0.0,
        ))

        if not components:
            return None

        system_name = "System"
        if arch_results:
            system_name = arch_results[0].get('system_name', 'System')

        lines = [
            "```mermaid",
            "graph TD",
            f"    subgraph {self._mermaid_safe(system_name)}",
        ]

        # Add components as nodes
        for comp in components:
            comp_id = self._mermaid_safe(comp.get('component_id', 'unknown'))
            comp_name = comp.get('component_name', comp_id)
            lines.append(f"        {comp_id}[{comp_name}]")

        # Add dependency edges
        for comp in components:
            comp_id = self._mermaid_safe(comp.get('component_id', 'unknown'))
            for dep in comp.get('dependencies', []):
                dep_id = self._mermaid_safe(dep)
                lines.append(f"        {comp_id} --> {dep_id}")

        lines.extend([
            "    end",
            "```",
        ])

        diagrams_path = self.docs_path / "diagrams"
        diagrams_path.mkdir(exist_ok=True)

        doc_path = diagrams_path / "architecture.mmd"
        doc_path.write_text("\n".join(lines), encoding='utf-8')

        return doc_path

    def generate_flow_diagram(self, flow_id: str) -> Optional[Path]:
        """Generate a Mermaid sequence diagram for a flow."""
        from .doc_memory import MemoryQuery

        # Query the specific flow
        flows = self.memory.query_memory(MemoryQuery(
            query_type="flow",
            max_entries=1,
            filter_by={"flow_id": flow_id},
            required_fields=[],
            min_confidence=0.0,
        ))

        if not flows:
            return None

        flow = flows[0]

        lines = [
            "```mermaid",
            "sequenceDiagram",
            f"    Note over All: {flow.get('flow_name', 'Flow')}",
        ]

        # Add participants
        participants = set()
        for step in flow.get('steps', []):
            comp = step.get('component', 'Unknown')
            if comp not in participants:
                participants.add(comp)
                lines.append(f"    participant {self._mermaid_safe(comp)}")

        # Add steps
        prev_comp = None
        for step in flow.get('steps', []):
            comp = self._mermaid_safe(step.get('component', 'Unknown'))
            action = step.get('action', 'process')
            next_comp = step.get('next')

            if prev_comp and prev_comp != comp:
                lines.append(f"    {prev_comp}->>+{comp}: {action}")
            else:
                lines.append(f"    Note over {comp}: {action}")

            if next_comp:
                lines.append(f"    {comp}->>+{self._mermaid_safe(next_comp)}: next")

            prev_comp = comp

        lines.append("```")

        diagrams_path = self.docs_path / "diagrams"
        diagrams_path.mkdir(exist_ok=True)

        safe_id = "".join(c if c.isalnum() or c in '-_' else '_' for c in flow_id)
        doc_path = diagrams_path / f"flow_{safe_id}.mmd"
        doc_path.write_text("\n".join(lines), encoding='utf-8')

        return doc_path

    def _mermaid_safe(self, text: str) -> str:
        """Make text safe for Mermaid diagram IDs."""
        # Remove or replace unsafe characters
        safe = "".join(c if c.isalnum() or c == '_' else '_' for c in text)
        # Ensure it starts with a letter
        if safe and safe[0].isdigit():
            safe = 'n' + safe
        return safe or 'unknown'

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def get_documentation_summary(self) -> dict:
        """Get summary of generated documentation."""
        files = list(self.docs_path.rglob('*'))
        docs = [f for f in files if f.is_file()]

        return {
            "output_directory": str(self.docs_path),
            "total_files": len(docs),
            "files": [str(d.relative_to(self.docs_path)) for d in docs],
        }
