"""
Agentic CLI

Interactive command-line interface for the Agentic planning and execution system.
Uses Rich for beautiful terminal output.

Supports:
- Phase 1: Planning & Design
- Phase 2: Execution & Sub-Agent Orchestration
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.progress import Progress, TextColumn
from rich.theme import Theme
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich.table import Table

from .master_agent import MasterAgent
from .openrouter_client import DEFAULT_MODEL, OpenRouterClient
from .execution_engine import ExecutionEngine, ExecutionStatus
from .context_discovery_agent import ContextDiscoveryAgent, DiscoveredContext
from .config import AgenticConfig, get_config
from .doc_agent import DocumentationAgent, DocAgentConfig


# Custom theme for the CLI
AGENTIC_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
    "heading": "magenta bold",
    "muted": "dim white",
})

console = Console(theme=AGENTIC_THEME)


def print_banner(phase: int = 1):
    """Print the welcome banner."""
    phase_text = "Phase 1: Planning & Design" if phase == 1 else "Phase 2: Execution"
    banner = f"""
+-----------------------------------------------------------+
|                                                           |
|     AGENTIC CLI                                           |
|     {phase_text:<43}|
|                                                           |
+-----------------------------------------------------------+
"""
    console.print(banner, style="cyan")


def print_help():
    """Print help information."""
    help_text = """
[heading]Agentic CLI - Phase 1 Planning[/heading]

[info]Commands during conversation:[/info]
  • Type your responses naturally
  • [bold]/quit[/bold] or [bold]/exit[/bold] - Exit without saving
  • [bold]/status[/bold] - Show current session status
  • [bold]/tree[/bold] - Show repository file tree
  • [bold]/help[/bold] - Show this help

[info]Workflow:[/info]
  1. Enter your high-level request
  2. Answer any clarifying questions
  3. Review the generated plan
  4. Approve to lock the plan
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))


class AgenticCLI:
    """
    Interactive CLI for the Agentic planning and execution system.
    """
    
    def __init__(
        self,
        repo_path: str | Path,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_attempts: int = 3,
        skip_discovery: bool = False,
        config: Optional[AgenticConfig] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.api_key = api_key
        self.model = model
        self.max_attempts = max_attempts
        self.skip_discovery = skip_discovery
        
        # Load or use provided config
        self.config = config or get_config(self.repo_path)
        
        # Show config status
        if self.config.disable_semantic_search:
            console.print("[muted]Semantic search: DISABLED (enterprise mode)[/]")
        
        self.agent = MasterAgent(repo_path, api_key=api_key, model=model, config=self.config)
        self.running = True
    
    def get_user_input(self, prompt_text: str) -> str:
        """Get input from the user with command handling."""
        while True:
            try:
                console.print()
                response = Prompt.ask(f"[bold cyan]{prompt_text}[/]")
                
                # Handle commands
                if response.lower() in ("/quit", "/exit"):
                    if Confirm.ask("Exit without saving?", default=False):
                        self.running = False
                        return "/quit"
                    continue
                
                if response.lower() == "/help":
                    print_help()
                    continue
                
                if response.lower() == "/status":
                    self.show_status()
                    continue
                
                if response.lower() == "/tree":
                    self.show_tree()
                    continue
                
                return response
                
            except (KeyboardInterrupt, EOFError):
                console.print("\n[warning]Interrupted[/]")
                if Confirm.ask("Exit without saving?", default=False):
                    self.running = False
                    return "/quit"
    
    def display_message(self, message: str):
        """Display a message to the user."""
        # Handle special formatting
        if message.startswith("[") and "]" in message:
            console.print(message, style="muted")
        elif "Error" in message:
            console.print(message, style="error")
        elif message.startswith("TOOL:"):
            console.print(message, style="cyan")
        else:
            console.print(message)
    
    def display_questions(self, questions: list[str]):
        """Display clarifying questions from the agent."""
        console.print()
        console.print(Rule("Clarifying Questions", style="yellow"))
        console.print()
        
        for i, q in enumerate(questions, 1):
            console.print(f"  [bold yellow]{i}.[/] {q}")
        
        console.print()
        console.print("[muted]Answer the questions above, or type /help for options.[/]")
    
    def display_plan(self, plan_content: str):
        """Display the generated plan."""
        console.print()
        console.print(Rule("Generated Plan", style="green"))
        console.print()
        
        # Render as markdown
        md = Markdown(plan_content)
        console.print(Panel(md, border_style="green", padding=(1, 2)))
    
    def confirm_plan(self) -> bool:
        """Ask user to confirm the plan."""
        console.print()
        return Confirm.ask(
            "[bold]Approve this plan?[/] (This will lock it)",
            default=True
        )
    
    def show_status(self):
        """Show current session status."""
        if not self.agent.session:
            console.print("[warning]No active session[/]")
            return
        
        session = self.agent.session
        
        status_text = f"""
[bold]Session Status[/bold]

[info]Repository:[/info] {self.repo_path}
[info]User Prompt:[/info] {session.user_prompt[:100]}{'...' if len(session.user_prompt) > 100 else ''}
[info]Conversation Turns:[/info] {len(session.conversation)}
[info]Plan Ready:[/info] {'Yes' if session.plan_data else 'No'}
"""
        console.print(Panel(status_text, title="Status", border_style="blue"))
    
    def show_tree(self):
        """Show repository file tree."""
        if not self.agent.session or not self.agent.session.repo_structure:
            console.print("[warning]No repository scanned[/]")
            return
        
        tree = self.agent.session.repo_structure.file_tree
        tree_lines = tree.split("\n")[:50]  # Limit display
        
        if len(tree.split("\n")) > 50:
            tree_lines.append("... (truncated, full tree available in repo)")
        
        console.print(Panel(
            "\n".join(tree_lines),
            title="Repository Structure",
            border_style="blue"
        ))
    
    async def run(self, initial_prompt: Optional[str] = None):
        """
        Run the interactive CLI.
        
        Args:
            initial_prompt: Optional initial prompt to start with
        """
        print_banner()
        
        console.print(f"[info]Repository:[/] {self.repo_path}")
        console.print()
        
        # Get initial prompt
        if initial_prompt:
            prompt = initial_prompt
            console.print(f"[bold]Your request:[/] {prompt}")
        else:
            console.print("[info]Describe what you want to build or change.[/]")
            console.print("[muted]Be as detailed as possible. The agent will ask clarifying questions.[/]")
            console.print()
            prompt = self.get_user_input("Your request")
            
            if prompt == "/quit":
                console.print("[muted]Goodbye![/]")
                return
        
        # Phase 0: Context Discovery (unless skipped)
        discovered_context: Optional[DiscoveredContext] = None
        
        if not self.skip_discovery:
            console.print()
            console.print(Rule("Phase 0: Context Discovery", style="cyan"))
            console.print("[muted]Exploring codebase to find relevant context...[/]")
            console.print()
            
            discovery_agent = ContextDiscoveryAgent(
                self.repo_path,
                api_key=self.api_key,
                model=self.model,
            )
            
            try:
                def discovery_status(msg: str):
                    console.print(msg, style="muted")
                
                discovered_context = await discovery_agent.discover(
                    prompt,
                    on_status=discovery_status,
                )
                
                console.print()
                console.print(f"[success]Discovery complete![/] Explored {discovered_context.files_explored} files in {discovered_context.exploration_rounds} rounds")
                
                if discovered_context.frameworks_detected:
                    console.print(f"[info]Frameworks detected:[/] {', '.join(discovered_context.frameworks_detected)}")
                if discovered_context.relevant_files:
                    console.print(f"[info]Relevant files found:[/] {len(discovered_context.relevant_files)}")
                
                # Show debug log locations
                console.print()
                console.print("[muted]Discovery debug logs:[/]")
                console.print(f"[muted]  • Full log: .agentic/discovery_debug_log.txt[/]")
                console.print(f"[muted]  • LLM context: .agentic/discovery_context.txt[/]")
                console.print(f"[muted]  • Handoff to Master: .agentic/discovery_handoff.txt[/]")
                
            finally:
                await discovery_agent.close()
        else:
            console.print("[muted]Skipping context discovery (--skip-discovery)[/]")
        
        # Start Master Agent session with discovered context
        console.print()
        console.print(Rule("Phase 1: Planning", style="magenta"))
        console.print("[muted]Starting planning session...[/]")
        
        session_id = await self.agent.start_session(prompt, discovered_context=discovered_context)
        console.print(f"[success]Session started:[/] {session_id[:8]}...")
        
        rs = self.agent.session.repo_structure
        console.print(f"[muted]Found {rs.total_files} files in {rs.total_dirs} directories[/]")
        
        # Run planning loop
        try:
            success = await self.agent.run_planning_loop(
                get_user_input=self.get_user_input,
                display_message=self.display_message,
                display_questions=self.display_questions,
                display_plan=self.display_plan,
                confirm_plan=self.confirm_plan,
            )
            
            if success:
                console.print()
                console.print(Panel(
                    "[success]Plan approved and locked![/]\n\n"
                    f"Plan saved to: [bold]{self.agent.state.plan_path}[/]\n\n"
                    "Phase 1 complete.",
                    title="Success",
                    border_style="green"
                ))
                
                # Ask to proceed to Phase 2
                console.print()
                if Confirm.ask("[bold]Proceed to Phase 2 execution?[/]", default=True):
                    await self.run_execution()
                else:
                    console.print("[muted]Execution skipped. Run again to execute later.[/]")
                    
            elif self.running:
                console.print()
                console.print("[warning]Planning session ended without approval.[/]")
                
        except Exception as e:
            console.print(f"[error]Error: {e}[/]")
            raise
        finally:
            await self.agent.close()
        
        console.print()
        console.print("[muted]Goodbye![/]")
    
    async def run_execution(self):
        """
        Run Phase 2 execution.
        
        Executes tasks from the locked plan.
        """
        console.print()
        print_banner(phase=2)
        
        console.print(f"[info]Repository:[/] {self.repo_path}")
        console.print()
        
        # Create status callback for live updates
        def status_callback(message: str):
            if message.startswith("[ERROR]"):
                console.print(message, style="error")
            elif message.startswith("[Task:"):
                if "completed successfully" in message:
                    console.print(message, style="success")
                elif "failed" in message.lower():
                    console.print(message, style="warning")
                else:
                    console.print(message, style="info")
            elif "=" in message and len(message) > 50:
                console.print(Rule(style="dim"))
            else:
                console.print(message, style="muted")
        
        # Create and run execution engine
        engine = ExecutionEngine(
            repo_path=self.repo_path,
            api_key=self.api_key,
            model=self.model,
            max_attempts=self.max_attempts,
            on_status=status_callback,
        )
        
        try:
            summary = await engine.run()
            
            # Display summary
            console.print()
            console.print(Rule("Execution Summary", style="magenta"))
            console.print()
            
            # Build summary table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Task ID")
            table.add_column("Status")
            table.add_column("Attempts")
            table.add_column("Notes")
            
            for result in summary.task_results:
                status_style = "green" if result.success else "red"
                status_text = "Completed" if result.success else result.final_status
                notes = result.error_message[:50] if result.error_message else "-"
                
                table.add_row(
                    result.task_id,
                    f"[{status_style}]{status_text}[/{status_style}]",
                    str(result.attempts),
                    notes,
                )
            
            console.print(table)
            console.print()
            
            # Overall status
            if summary.status == ExecutionStatus.COMPLETED:
                console.print(Panel(
                    f"[success]All tasks completed successfully![/]\n\n"
                    f"Tasks: {summary.tasks_completed}/{summary.tasks_completed + summary.tasks_failed + summary.tasks_escalated}\n"
                    f"Total attempts: {summary.total_attempts}\n"
                    f"Duration: {summary.duration_seconds:.1f}s",
                    title="Execution Complete",
                    border_style="green"
                ))
            elif summary.status == ExecutionStatus.HALTED:
                console.print(Panel(
                    f"[warning]Execution halted - user intervention required[/]\n\n"
                    f"Completed: {summary.tasks_completed}\n"
                    f"Escalated: {summary.tasks_escalated}\n"
                    f"Failed: {summary.tasks_failed}\n\n"
                    f"Check .agentic/attempts/ for details.",
                    title="Execution Halted",
                    border_style="yellow"
                ))
            else:
                console.print(Panel(
                    f"[error]Execution failed[/]\n\n"
                    f"Completed: {summary.tasks_completed}\n"
                    f"Failed: {summary.tasks_failed}\n\n"
                    f"Check .agentic/attempts/ for details.",
                    title="Execution Failed",
                    border_style="red"
                ))
                
        except Exception as e:
            console.print(f"[error]Execution error: {e}[/]")
            raise


async def run_documentation(
    repo_path: Path,
    api_key: Optional[str],
    model: str,
    mode: str = "full",
):
    """
    Run the Documentation Agent.

    Args:
        repo_path: Path to the repository
        api_key: OpenRouter API key
        model: Model to use
        mode: "full", "resume", or "update"
    """
    console.print()
    console.print(Panel(
        "[bold]Documentation Agent[/]\n"
        "Phased codebase exploration with long-term memory",
        border_style="cyan"
    ))
    console.print()

    console.print(f"[info]Repository:[/] {repo_path}")
    console.print(f"[info]Mode:[/] {mode}")
    console.print()

    # Create LLM client
    llm = OpenRouterClient(api_key=api_key, model=model)

    # Status callback for live updates
    def status_callback(message: str):
        if "[Phase:" in message or "[Component" in message:
            console.print(message, style="cyan")
        else:
            console.print(message, style="muted")

    try:
        config = DocAgentConfig(debug_logging=True, on_status=status_callback)
        agent = DocumentationAgent(repo_path, llm, config)

        console.print(f"[info]Session:[/] {agent.session_id}")
        console.print()

        if mode == "full":
            console.print("[muted]Starting full documentation run...[/]")
            console.print("[muted]This will explore the codebase in phases and generate docs.[/]")
            console.print()

            result = await agent.run_full_documentation()

        elif mode == "resume":
            if not agent.orchestrator.can_resume():
                console.print("[warning]No session to resume. Starting fresh.[/]")
                result = await agent.run_full_documentation()
            else:
                console.print("[muted]Resuming from last saved state...[/]")
                result = await agent.resume()

        elif mode == "update":
            console.print("[muted]Checking for changed files...[/]")
            result = await agent.update_documentation()

        else:
            console.print(f"[error]Unknown mode: {mode}[/]")
            return

        # Display results
        console.print()
        console.print(Rule("Results", style="green"))
        console.print()

        if result.get("errors"):
            console.print("[warning]Errors encountered:[/]")
            for err in result["errors"]:
                console.print(f"  [error]• {err}[/]")
            console.print()

        if result.get("phases_completed"):
            console.print("[success]Phases completed:[/]")
            for phase in result["phases_completed"]:
                console.print(f"  [success]✓[/] {phase}")
            console.print()

        if result.get("documentation"):
            console.print("[success]Documentation generated:[/]")
            for doc in result["documentation"]:
                console.print(f"  [info]•[/] {doc}")
            console.print()

        if result.get("changed_files"):
            console.print("[info]Changed files detected:[/]")
            for f in result["changed_files"][:10]:
                console.print(f"  • {f}")
            if len(result["changed_files"]) > 10:
                console.print(f"  ... and {len(result['changed_files']) - 10} more")
            console.print()

        # Show memory summary
        memory_summary = agent.get_memory_summary()
        console.print("[info]Memory Summary:[/]")
        console.print(f"  • Phase: {memory_summary.get('current_phase', 'unknown')}")
        console.print(f"  • Components discovered: {memory_summary.get('components_discovered', 0)}")
        console.print(f"  • Components explored: {memory_summary.get('components_explored', 0)}")
        console.print(f"  • Files documented: {memory_summary.get('files_documented', 0)}")
        console.print()

        # Show output location
        console.print(Panel(
            f"[info]Documentation output:[/]\n"
            f"  {repo_path / '.agentic' / 'documentation'}\n\n"
            f"[info]Memory storage:[/]\n"
            f"  {repo_path / '.agentic' / 'memory'}\n\n"
            f"[info]Debug logs:[/]\n"
            f"  {repo_path / '.agentic' / 'doc_sessions'}",
            title="Output Locations",
            border_style="blue"
        ))

    finally:
        await llm.close()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Agentic CLI - Planning, Execution, and Documentation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agentic .                          # Start planning in current directory
  agentic /path/to/repo              # Start planning in specified repo
  agentic . --prompt "Add auth"      # Start with an initial prompt
  agentic . --model gpt-4o           # Use a different model
  agentic document .                 # Generate documentation for repo
  agentic document . --resume        # Resume interrupted documentation
        """
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Document subcommand
    doc_parser = subparsers.add_parser(
        "document",
        help="Generate documentation for a repository using phased exploration"
    )
    doc_parser.add_argument(
        "doc_repo_path",
        type=str,
        nargs="?",
        default=".",
        help="Path to the repository (default: current directory)"
    )
    doc_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved state"
    )
    doc_parser.add_argument(
        "--update",
        action="store_true",
        help="Update documentation for changed files only"
    )
    doc_parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    doc_parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )

    # Default planning command arguments (when no subcommand specified)
    parser.add_argument(
        "repo_path",
        type=str,
        nargs="?",
        default=".",
        help="Path to the repository (default: current directory)"
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Initial prompt to start with"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )

    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip the Context Discovery Agent phase (go straight to planning)"
    )

    parser.add_argument(
        "--no-semantic-search",
        action="store_true",
        help="Disable semantic search tool (for enterprises without embedding models)"
    )

    args = parser.parse_args()

    # Handle document command
    if args.command == "document":
        doc_repo_path = Path(args.doc_repo_path).resolve()
        if not doc_repo_path.exists():
            console.print(f"[error]Error: Path does not exist: {doc_repo_path}[/]")
            sys.exit(1)
        if not doc_repo_path.is_dir():
            console.print(f"[error]Error: Path is not a directory: {doc_repo_path}[/]")
            sys.exit(1)

        mode = "full"
        if args.resume:
            mode = "resume"
        elif args.update:
            mode = "update"

        try:
            asyncio.run(run_documentation(
                repo_path=doc_repo_path,
                api_key=args.api_key,
                model=args.model,
                mode=mode,
            ))
        except KeyboardInterrupt:
            console.print("\n[warning]Interrupted[/]")
            sys.exit(130)
        return
    
    # Build config from args
    config = get_config(Path(args.repo_path).resolve())
    
    # Override with CLI flags
    if args.no_semantic_search:
        config.disable_semantic_search = True
    
    # Validate repo path
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        console.print(f"[error]Error: Path does not exist: {repo_path}[/]")
        sys.exit(1)
    if not repo_path.is_dir():
        console.print(f"[error]Error: Path is not a directory: {repo_path}[/]")
        sys.exit(1)
    
    # Run CLI
    cli = AgenticCLI(
        repo_path=repo_path,
        api_key=args.api_key,
        model=args.model,
        skip_discovery=args.skip_discovery,
        config=config,
    )
    
    try:
        asyncio.run(cli.run(initial_prompt=args.prompt))
    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":
    main()

