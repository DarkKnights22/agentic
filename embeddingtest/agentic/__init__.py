"""
Agentic CLI Coding System

A highly agentic, long-running CLI-based coding system that separates
planning, execution, and verification through task-scoped sub-agents.

Phase 1: Planning & Design
Phase 2: Execution & Sub-Agent Orchestration

Enterprise Integration:
- Custom LLM clients via LLMClient protocol (for Azure, AWS Bedrock, internal APIs)
- Configurable features via AgenticConfig (disable semantic search, etc.)
"""

__version__ = "0.3.0"

# Enterprise integration
from .llm_client import LLMClient, Message, ChatResponse, TokenUsage, SimpleLLMClient
from .config import AgenticConfig, get_config, set_config

# Default OpenRouter client (implements LLMClient)
from .openrouter_client import OpenRouterClient

from .repo_scanner import RepoScanner, RepoStructure, scan_repo
from .state_manager import StateManager, PlanLockError, ContextViolationError
from .plan_generator import PlanGenerator, PlanDocument, create_plan
from .master_agent import MasterAgent

# Phase 2 modules
from .plan_parser import parse_plan, load_plan, ParsedPlan, ParsedTask
from .tools import apply_patch, run_tests, PatchResult, TestResult
from .sub_agent import SubAgent, TaskContext, SubAgentResult
from .execution_engine import ExecutionEngine, ExecutionSummary, run_execution

# Documentation Agent (Long-Term Memory)
from .doc_agent import DocumentationAgent, DocAgentConfig, document_repository
from .doc_memory import MemoryManager, MemoryQuery, MemoryQueryError
from .doc_phases import PhaseOrchestrator, Phase
from .doc_writer import DocWriter

__all__ = [
    # Enterprise Integration (NEW)
    "LLMClient",          # Abstract base for custom LLM implementations
    "SimpleLLMClient",    # Simple wrapper for quick custom implementations
    "AgenticConfig",      # Configuration (disable_semantic_search, etc.)
    "get_config",
    "set_config",
    
    # LLM Client (default)
    "OpenRouterClient",
    "Message", 
    "ChatResponse",
    "TokenUsage",
    
    # Scanner
    "RepoScanner",
    "RepoStructure",
    "scan_repo",
    
    # State
    "StateManager",
    "PlanLockError",
    "ContextViolationError",
    
    # Planning (Phase 1)
    "PlanGenerator",
    "PlanDocument",
    "create_plan",
    
    # Plan Parser (Phase 2)
    "parse_plan",
    "load_plan",
    "ParsedPlan",
    "ParsedTask",
    
    # Tools (Phase 2)
    "apply_patch",
    "run_tests",
    "PatchResult",
    "TestResult",
    
    # Sub-Agent (Phase 2)
    "SubAgent",
    "TaskContext",
    "SubAgentResult",
    
    # Execution (Phase 2)
    "ExecutionEngine",
    "ExecutionSummary",
    "run_execution",
    
    # Master Agent
    "MasterAgent",

    # Documentation Agent (Long-Term Memory)
    "DocumentationAgent",
    "DocAgentConfig",
    "document_repository",
    "MemoryManager",
    "MemoryQuery",
    "MemoryQueryError",
    "PhaseOrchestrator",
    "Phase",
    "DocWriter",
]

