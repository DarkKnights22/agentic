"""
Documentation Agent with Long-Term Memory.

Main agent class that orchestrates phased codebase exploration and
documentation generation using persistent, schema-driven memory.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator
import uuid
import json

from .llm_client import LLMClient, Message, ChatResponse
from .doc_memory import MemoryManager, MemoryQuery
from .doc_phases import PhaseOrchestrator, Phase, build_architecture_context, build_component_context, build_cross_cutting_context, build_docgen_context
from .doc_tools import DocTools, ToolResult
from .doc_prompts import build_system_prompt, build_user_message, parse_response
from .doc_writer import DocWriter


# =============================================================================
# Debug Logger
# =============================================================================

class DocDebugLogger:
    """Debug logging for documentation agent."""

    def __init__(self, repo_path: Path, session_id: str):
        self.log_dir = repo_path / ".agentic" / "doc_sessions"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{session_id}_debug.txt"
        self.context_file = self.log_dir / f"{session_id}_context.txt"

    def log(self, message: str):
        """Append to incremental log."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    def log_round(self, round_num: int, phase: str, messages: list[Message], response: str):
        """Log a complete round including context snapshot."""
        self.log(f"=== Round {round_num} ({phase}) ===")

        # Full context snapshot (overwrite)
        with open(self.context_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Context Snapshot: Round {round_num} ({phase}) ===\n\n")
            for msg in messages:
                f.write(f"--- {msg.role.upper()} ---\n")
                f.write(msg.content[:5000])  # Truncate for readability
                if len(msg.content) > 5000:
                    f.write(f"\n... (truncated, {len(msg.content)} chars total)\n")
                f.write("\n\n")
            f.write("--- RESPONSE ---\n")
            f.write(response[:5000])
            if len(response) > 5000:
                f.write(f"\n... (truncated, {len(response)} chars total)\n")

    def log_tool_result(self, tool_name: str, result: ToolResult):
        """Log a tool execution result."""
        status = "SUCCESS" if result.success else "FAILED"
        self.log(f"Tool {tool_name}: {status}")
        if result.budget_warning:
            self.log(f"  WARNING: {result.budget_warning}")


# =============================================================================
# Documentation Agent
# =============================================================================

@dataclass
class DocAgentConfig:
    """Configuration for the documentation agent."""
    max_rounds_per_phase: int = 30
    max_tool_calls_per_round: int = 5
    debug_logging: bool = True
    on_status: Optional[callable] = None  # Callback for progress updates


class DocumentationAgent:
    """
    Documentation Agent with Long-Term Memory.

    Orchestrates phased codebase exploration:
    - Phase A: Architecture Discovery
    - Phase B: Component Deep Dives
    - Phase C: Cross-Cutting Concerns
    - Phase D: Documentation Generation

    All discoveries are stored to persistent memory.
    Documentation is generated from memory only.
    """

    def __init__(
        self,
        repo_path: Path,
        llm: LLMClient,
        config: Optional[DocAgentConfig] = None,
        session_id: Optional[str] = None,
    ):
        self.repo_path = Path(repo_path)
        self.llm = llm
        self.config = config or DocAgentConfig()
        self.session_id = session_id or f"doc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Initialize components
        self.memory = MemoryManager(self.repo_path)
        self.orchestrator = PhaseOrchestrator(self.repo_path, self.session_id)
        self.tools = DocTools(self.repo_path, self.memory, self.orchestrator)
        self.writer = DocWriter(self.repo_path, self.memory)

        # Debug logging
        self.logger = DocDebugLogger(self.repo_path, self.session_id) if self.config.debug_logging else None

        # Conversation state
        self._conversation: list[Message] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run_full_documentation(self) -> dict:
        """
        Run complete documentation workflow through all phases.

        Returns dict with status and generated documentation paths.
        """
        results = {
            "session_id": self.session_id,
            "phases_completed": [],
            "documentation": [],
            "errors": [],
        }

        try:
            # Phase A: Architecture Discovery
            if not self.orchestrator.is_phase_completed(Phase.ARCHITECTURE_DISCOVERY):
                await self._run_phase(Phase.ARCHITECTURE_DISCOVERY)
                results["phases_completed"].append("architecture_discovery")

            # Phase B: Component Deep Dives
            if not self.orchestrator.is_phase_completed(Phase.COMPONENT_DEEP_DIVE):
                await self._run_component_phase()
                results["phases_completed"].append("component_deep_dive")

            # Phase C: Cross-Cutting Concerns
            if not self.orchestrator.is_phase_completed(Phase.CROSS_CUTTING):
                await self._run_phase(Phase.CROSS_CUTTING)
                results["phases_completed"].append("cross_cutting")

            # Phase D: Documentation Generation
            if not self.orchestrator.is_phase_completed(Phase.DOCUMENTATION_GENERATION):
                await self._run_phase(Phase.DOCUMENTATION_GENERATION)
                results["phases_completed"].append("documentation_generation")

            # Generate final documentation
            doc_results = self.writer.generate_all()
            results["documentation"] = doc_results["generated"]
            results["errors"].extend(doc_results["errors"])

        except Exception as e:
            results["errors"].append(str(e))

        return results

    async def resume(self) -> dict:
        """
        Resume from last saved state.

        Returns dict with status and what was completed.
        """
        if not self.orchestrator.can_resume():
            return {"error": "No session to resume", "status": "nothing_to_resume"}

        return await self.run_full_documentation()

    async def update_documentation(self) -> dict:
        """
        Update documentation for changed files only.

        Returns dict with updated components and files.
        """
        changed_files = self.memory.get_changed_files()

        if not changed_files:
            return {"status": "no_changes", "changed_files": []}

        # TODO: Implement selective re-exploration
        # For now, just report what changed
        return {
            "status": "changes_detected",
            "changed_files": changed_files,
            "note": "Selective re-exploration not yet implemented. Run full documentation for updates.",
        }

    def get_memory_summary(self) -> dict:
        """Get summary of what's stored in memory."""
        return self.memory.get_memory_summary()

    def get_documentation_status(self) -> dict:
        """Get current documentation progress status."""
        return self.orchestrator.get_documentation_status()

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _run_phase(self, phase: Phase) -> str:
        """
        Run a single phase through the LLM reasoning loop.

        Returns the findings summary.
        """
        self._log(f"Starting phase: {phase.value}")
        self._status(f"[Phase: {phase.value}] Starting...")

        # Start the phase
        self.orchestrator.start_phase(phase)
        self._conversation = []  # Fresh conversation for each phase

        # Build initial context
        context = self._build_phase_context(phase)

        for round_num in range(self.config.max_rounds_per_phase):
            self._log(f"Phase {phase.value} - Round {round_num + 1}")
            self._status(f"[Phase: {phase.value}] Round {round_num + 1}/{self.config.max_rounds_per_phase} - Calling LLM...")

            # Build messages
            system_prompt = build_system_prompt(phase.value)
            user_message = build_user_message(
                phase.value,
                context,
                self._format_conversation_history()
            )

            messages = [
                Message("system", system_prompt),
                Message("user", user_message),
            ]

            # Add conversation history
            messages.extend(self._conversation)

            # Call LLM
            response = await self.llm.chat(messages, max_tokens=4096)

            if self.logger:
                self.logger.log_round(round_num, phase.value, messages, response.content)

            # Parse response
            parsed = parse_response(response.content)

            # Check for phase completion
            if parsed["phase_complete"] or parsed["doc_generation_complete"]:
                findings = parsed.get("findings_summary", "Phase completed")
                self.orchestrator.complete_phase(findings)
                self._log(f"Phase {phase.value} completed: {findings[:100]}...")
                self._status(f"[Phase: {phase.value}] Complete!")
                return findings

            # Execute tool calls
            if parsed["tool_calls"]:
                tool_calls = parsed["tool_calls"][:self.config.max_tool_calls_per_round]
                tool_names = [tc.get("tool", "?") for tc in tool_calls]
                self._status(f"[Phase: {phase.value}] Executing tools: {', '.join(tool_names)}")
                tool_results = await self._execute_tools(tool_calls)

                # Add to conversation
                self._conversation.append(Message("assistant", response.content))
                self._conversation.append(Message("user", f"Tool results:\n\n{tool_results}"))
                continue

            # No tool calls and no completion - add response and continue
            self._conversation.append(Message("assistant", response.content))
            self._conversation.append(Message("user", "Continue with exploration or signal phase completion."))

        # Max rounds reached
        self._log(f"Phase {phase.value} reached max rounds")
        self.orchestrator.complete_phase("Max rounds reached - partial exploration")
        return "Max rounds reached"

    async def _run_component_phase(self):
        """
        Run Phase B for all discovered components.

        Each component gets its own context and budget.
        """
        self._log("Starting component deep dives")
        self._status("[Phase: component_deep_dive] Starting...")
        self.orchestrator.start_phase(Phase.COMPONENT_DEEP_DIVE)

        # Get components to explore
        unexplored = self.memory.get_unexplored_components()

        if not unexplored:
            # Query architecture for component list
            arch_results = self.memory.query_memory(MemoryQuery(
                query_type="architecture",
                max_entries=1,
                required_fields=["major_components", "component_boundaries"],
                min_confidence=0.0,
            ))

            if arch_results:
                arch = arch_results[0]
                boundaries = arch.get("component_boundaries", {})
                for comp_name, comp_path in boundaries.items():
                    comp_id = comp_name.lower().replace(" ", "_")
                    if comp_id not in self.memory._index.components_explored:
                        unexplored.append(comp_id)

                        # Store placeholder component
                        self.memory.store_discovery("component", {
                            "component_id": comp_id,
                            "component_name": comp_name,
                            "root_path": comp_path[0] if isinstance(comp_path, list) else comp_path,
                            "responsibility": "To be discovered",
                            "confidence": 0.1,
                        })

        # Explore each component
        total_components = len(unexplored)
        for idx, component_id in enumerate(unexplored, 1):
            self._log(f"Exploring component: {component_id}")
            self._status(f"[Component {idx}/{total_components}] Exploring: {component_id}")
            self.orchestrator.start_component(component_id)

            # Get component info
            comp_info = self.memory.query_memory(MemoryQuery(
                query_type="component",
                max_entries=1,
                filter_by={"component_id": component_id},
                required_fields=["component_name", "root_path"],
                min_confidence=0.0,
            ))

            comp_path = "unknown"
            if comp_info:
                comp_path = comp_info[0].get("root_path", "unknown")

            # Run exploration loop for this component
            await self._run_component_exploration(component_id, comp_path)

            self.orchestrator.complete_component(component_id)
            self.memory.mark_component_explored(component_id)

        self.orchestrator.complete_phase("All components explored")

    async def _run_component_exploration(self, component_id: str, component_path: str):
        """Run exploration loop for a single component."""
        self._conversation = []  # Fresh conversation

        context = build_component_context(self.memory, self.orchestrator, component_id)

        for round_num in range(self.config.max_rounds_per_phase):
            system_prompt = build_system_prompt(
                "component_deep_dive",
                component_id=component_id,
                component_path=component_path,
            )
            user_message = build_user_message(
                "component_deep_dive",
                context,
                self._format_conversation_history()
            )

            messages = [
                Message("system", system_prompt),
                Message("user", user_message),
            ]
            messages.extend(self._conversation)

            response = await self.llm.chat(messages, max_tokens=4096)

            if self.logger:
                self.logger.log_round(round_num, f"component_{component_id}", messages, response.content)

            parsed = parse_response(response.content)

            if parsed["phase_complete"]:
                self._log(f"Component {component_id} exploration complete")
                return

            if parsed["tool_calls"]:
                tool_calls = parsed["tool_calls"][:self.config.max_tool_calls_per_round]
                tool_results = await self._execute_tools(tool_calls)

                self._conversation.append(Message("assistant", response.content))
                self._conversation.append(Message("user", f"Tool results:\n\n{tool_results}"))
                continue

            self._conversation.append(Message("assistant", response.content))
            self._conversation.append(Message("user", "Continue exploring or signal completion."))

    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------

    async def _execute_tools(self, tool_calls: list[dict]) -> str:
        """Execute a list of tool calls and return formatted results."""
        results = []

        for call in tool_calls:
            tool_name = call.get("tool", "")
            params = call.get("params", {})

            result = self._execute_single_tool(tool_name, params)

            if self.logger:
                self.logger.log_tool_result(tool_name, result)

            status = "SUCCESS" if result.success else "FAILED"
            result_text = f"### {tool_name} [{status}]\n{result.content}"

            if result.budget_warning:
                result_text += f"\n\n⚠️ {result.budget_warning}"

            results.append(result_text)

        return "\n\n".join(results)

    def _execute_single_tool(self, tool_name: str, params: dict) -> ToolResult:
        """Execute a single tool call."""
        try:
            if tool_name == "grep":
                return self.tools.grep(
                    pattern=params.get("pattern", ""),
                    file_extensions=params.get("file_extensions"),
                    context_lines=int(params.get("context_lines", 0)),
                    max_results=int(params.get("max_results", 20)),
                )
            elif tool_name == "read_file":
                return self.tools.read_file(
                    file_path=params.get("file_path", params.get("path", "")),
                    max_lines=int(params.get("max_lines", 200)),
                )
            elif tool_name == "list_files":
                return self.tools.list_files(
                    directory=params.get("directory", ""),
                    pattern=params.get("pattern", "*"),
                )
            elif tool_name == "get_symbols":
                return self.tools.get_symbols(
                    file_path=params.get("file_path", params.get("path", "")),
                )
            elif tool_name == "query_memory":
                return self.tools.query_memory(
                    query_type=params.get("query_type", ""),
                    max_entries=int(params.get("max_entries", 10)),
                    filter_by=params.get("filter_by"),
                    required_fields=params.get("required_fields"),
                    min_confidence=float(params.get("min_confidence", 0.0)),
                    sort_by=params.get("sort_by", "relevance"),
                )
            elif tool_name == "store_discovery":
                data = params.get("data", {})
                if isinstance(data, str):
                    data = json.loads(data)
                return self.tools.store_discovery(
                    entry_type=params.get("entry_type", ""),
                    data=data,
                )
            elif tool_name == "get_phase_context":
                return self.tools.get_phase_context()
            elif tool_name == "mark_explored":
                return self.tools.mark_explored(
                    target_type=params.get("target_type", ""),
                    target_id=params.get("target_id", ""),
                )
            elif tool_name == "estimate_token_usage":
                return self.tools.estimate_token_usage(
                    query_type=params.get("query_type", ""),
                    max_entries=int(params.get("max_entries", 10)),
                    filter_by=params.get("filter_by"),
                    required_fields=params.get("required_fields"),
                    min_confidence=float(params.get("min_confidence", 0.0)),
                )
            else:
                return ToolResult(success=False, content=f"Unknown tool: {tool_name}")
        except Exception as e:
            return ToolResult(success=False, content=f"Tool error: {e}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_phase_context(self, phase: Phase) -> str:
        """Build context string for a phase."""
        if phase == Phase.ARCHITECTURE_DISCOVERY:
            return build_architecture_context(self.memory, self.orchestrator)
        elif phase == Phase.CROSS_CUTTING:
            return build_cross_cutting_context(self.memory, self.orchestrator)
        elif phase == Phase.DOCUMENTATION_GENERATION:
            return build_docgen_context(self.memory, self.orchestrator)
        return ""

    def _format_conversation_history(self) -> str:
        """Format recent conversation for context."""
        if not self._conversation:
            return ""

        # Only include last few turns to manage context size
        recent = self._conversation[-6:]  # Last 3 exchanges
        parts = []
        for msg in recent:
            role = "Assistant" if msg.role == "assistant" else "Tool Results"
            content = msg.content[:1000]  # Truncate
            if len(msg.content) > 1000:
                content += "..."
            parts.append(f"[{role}]: {content}")

        return "\n\n".join(parts)

    def _log(self, message: str):
        """Log a message if debug logging is enabled."""
        if self.logger:
            self.logger.log(message)

    def _status(self, message: str):
        """Emit a status update if callback configured."""
        if self.config.on_status:
            self.config.on_status(message)


# =============================================================================
# Convenience Functions
# =============================================================================

async def document_repository(
    repo_path: Path,
    llm: LLMClient,
    config: Optional[DocAgentConfig] = None,
) -> dict:
    """
    Convenience function to document a repository.

    Usage:
        from agentic.doc_agent import document_repository
        from agentic.openrouter_client import OpenRouterClient

        llm = OpenRouterClient(api_key="...")
        result = await document_repository(Path("/path/to/repo"), llm)
    """
    agent = DocumentationAgent(repo_path, llm, config)
    return await agent.run_full_documentation()
