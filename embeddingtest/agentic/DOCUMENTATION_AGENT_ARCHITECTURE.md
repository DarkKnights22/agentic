# Documentation Agent Architecture

A phased codebase exploration system with long-term memory that generates comprehensive documentation automatically.

## Overview

The Documentation Agent is designed to systematically explore codebases and produce high-quality documentation. Unlike traditional documentation tools that rely on static analysis alone, this agent uses an LLM to reason about code, make exploration decisions, and synthesize findings into human-readable documentation.

**Key Design Principles:**

1. **Phased Exploration** - Work progresses through distinct phases, each with specific goals
2. **Budget Enforcement** - Hard limits on file reads/greps prevent runaway token usage
3. **Persistent Memory** - All discoveries are stored to disk, enabling resume and incremental updates
4. **Memory-Only Generation** - Final documentation is generated purely from stored memory, ensuring consistency

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI (cli.py)                                   │
│                    Entry point, progress display, user I/O                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DocumentationAgent (doc_agent.py)                      │
│                                                                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ LLM Client   │  │ PhaseOrchestrator│  │ DocTools    │  │ DocWriter   │  │
│  │ (OpenRouter) │  │ (doc_phases.py)  │  │(doc_tools.py│  │(doc_writer) │  │
│  └──────────────┘  └──────────────────┘  └─────────────┘  └─────────────┘  │
│         │                   │                   │               │          │
│         │                   │                   │               │          │
│         ▼                   ▼                   ▼               ▼          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     MemoryManager (doc_memory.py)                    │  │
│  │               Schema-driven persistent storage layer                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          .agentic/ Directory                                │
│  ├── memory/           (persistent memory storage)                          │
│  │   ├── index.json                                                        │
│  │   ├── architecture/                                                     │
│  │   ├── components/                                                       │
│  │   ├── files/                                                            │
│  │   ├── data_models/                                                      │
│  │   ├── flows/                                                            │
│  │   └── cross_cutting/                                                    │
│  ├── documentation/    (generated output)                                   │
│  │   ├── ARCHITECTURE.md                                                   │
│  │   ├── DATA_MODELS.md                                                    │
│  │   ├── components/                                                       │
│  │   └── diagrams/                                                         │
│  └── doc_sessions/     (debug logs)                                         │
│       ├── {session_id}.json                                                │
│       ├── {session_id}_debug.txt                                           │
│       └── {session_id}_context.txt                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. DocumentationAgent (`doc_agent.py`)

The main orchestrator that coordinates all other components.

```python
class DocumentationAgent:
    def __init__(self, repo_path, llm, config, session_id):
        self.memory = MemoryManager(repo_path)
        self.orchestrator = PhaseOrchestrator(repo_path, session_id)
        self.tools = DocTools(repo_path, self.memory, self.orchestrator)
        self.writer = DocWriter(repo_path, self.memory)
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `run_full_documentation()` | Execute all 4 phases sequentially |
| `resume()` | Continue from last saved state |
| `update_documentation()` | Detect changed files and update docs |
| `_run_phase(phase)` | Execute a single phase's LLM reasoning loop |
| `_execute_tools(tool_calls)` | Execute tools requested by LLM |

**Conversation Loop:**

Each phase runs an LLM conversation loop:

```
1. Build system prompt + phase context
2. Call LLM with messages
3. Parse response for:
   - Tool calls → Execute and continue
   - Phase complete signal → Store findings and exit
4. Repeat until completion or max rounds (30)
```

### 2. MemoryManager (`doc_memory.py`)

Persistent, schema-driven storage with bounded retrieval.

**Memory Entry Types:**

| Type | Schema Class | Purpose |
|------|--------------|---------|
| `architecture` | `ArchitectureOverview` | System-level overview |
| `component` | `ComponentSummary` | Per-component details |
| `file` | `FileRole` | Individual file metadata |
| `data_model` | `DataModel` | Data structures/schemas |
| `flow` | `RuntimeFlow` | Request/data flows |
| `cross_cutting` | `CrossCuttingConcern` | Auth, logging, config, etc. |

**Schema Example - ComponentSummary:**

```python
@dataclass
class ComponentSummary:
    component_id: str            # Unique identifier
    component_name: str          # Human-readable name
    root_path: str               # Directory path
    responsibility: str          # What it does
    key_files: list[str]         # Important files
    public_interfaces: list[dict] # Exported APIs
    dependencies: list[str]      # Other components it uses
    dependents: list[str]        # Components that use it
    design_patterns_used: list[str]
    confidence: float            # 0.0-1.0 certainty level
    explored_files: list[str]    # Files actually read
```

**Bounded Retrieval Contract:**

All queries MUST specify `max_entries` - there is no default:

```python
@dataclass
class MemoryQuery:
    query_type: str      # Required
    max_entries: int     # REQUIRED - no default, max 100
    filter_by: dict      # Optional filters
    required_fields: list[str]  # Only return these fields
    min_confidence: float       # Filter low-confidence entries
    sort_by: str         # "relevance" | "recency" | "confidence"
```

This prevents accidental loading of unbounded data into LLM context.

### 3. PhaseOrchestrator (`doc_phases.py`)

Manages phase transitions and budget enforcement.

**The Four Phases:**

```
Phase A: Architecture Discovery
    ↓
Phase B: Component Deep Dives (loops per component)
    ↓
Phase C: Cross-Cutting Concerns
    ↓
Phase D: Documentation Generation
```

**Budget Configuration:**

Each phase has hard limits to prevent token explosion:

```python
PHASE_BUDGETS = {
    Phase.ARCHITECTURE_DISCOVERY: PhaseBudget(
        max_files_read=10,
        max_grep_calls=15,
        max_symbols_calls=5,
    ),
    Phase.COMPONENT_DEEP_DIVE: PhaseBudget(  # Per component!
        max_files_read=20,
        max_grep_calls=10,
        max_symbols_calls=10,
    ),
    Phase.CROSS_CUTTING: PhaseBudget(
        max_files_read=15,
        max_grep_calls=10,
        max_cross_component_queries=5,
    ),
    Phase.DOCUMENTATION_GENERATION: PhaseBudget(
        max_files_read=0,      # NO file reads allowed!
        max_grep_calls=0,
        max_memory_queries=20,
    ),
}
```

**Budget Enforcement:**

When a tool is called, the orchestrator checks budget first:

```python
def check_budget(self, operation: str) -> tuple[bool, str]:
    if usage.files_read >= budget.max_files_read:
        return False, "BUDGET_EXHAUSTED: File read limit reached. Summarize findings."
    return True, "OK"
```

### 4. DocTools (`doc_tools.py`)

Budget-aware tools available to the LLM.

**Exploration Tools (Budget-Checked):**

| Tool | Purpose | Budget Impact |
|------|---------|---------------|
| `grep(pattern, ...)` | Search codebase | +1 grep_call |
| `read_file(path, ...)` | Read file contents | +1 file_read |
| `get_symbols(path)` | Extract classes/functions | +1 symbols_call |
| `list_files(dir, ...)` | List directory | None (free) |

**Memory Tools:**

| Tool | Purpose |
|------|---------|
| `query_memory(...)` | Bounded retrieval from memory |
| `store_discovery(type, data)` | Save finding to memory |
| `get_phase_context()` | Get current phase + budget status |
| `mark_explored(type, id)` | Mark component as done |
| `estimate_token_usage(...)` | Preview query size before executing |

**Tool Result Format:**

```python
@dataclass
class ToolResult:
    success: bool
    content: str
    budget_warning: Optional[str]  # "WARNING: Only 2 file reads remaining"
```

### 5. DocPrompts (`doc_prompts.py`)

LLM prompts that guide behavior in each phase.

**Base System Prompt:**

```
You are a Documentation Agent specialized in understanding and documenting codebases.

CRITICAL RULES:
1. NEVER guess or assume - always verify with tools
2. RESPECT BUDGET LIMITS - when budget exhausted, summarize and store what you know
3. STORE DISCOVERIES - use store_discovery before moving on
4. BOUNDED QUERIES ONLY - always specify max_entries when querying memory
```

**Phase-Specific Prompts:**

- **Architecture Discovery**: Find system name, tech stack, entry points, components, architecture style
- **Component Deep Dive**: Fully understand one component - files, interfaces, dependencies
- **Cross-Cutting**: Find patterns spanning components - auth, config, logging, error handling
- **Doc Generation**: Generate markdown from memory ONLY (no file reads allowed)

**Response Parsing:**

The LLM responds in structured XML format:

```xml
<!-- Tool calls -->
<function_calls>
<invoke name="grep">
<parameter name="pattern">def main</parameter>
</invoke>
</function_calls>

<!-- Phase completion -->
<phase_complete>
{
  "findings_summary": "Discovered 5 major components...",
  "confidence": 0.85
}
</phase_complete>
```

### 6. DocWriter (`doc_writer.py`)

Generates documentation purely from memory.

**Output Files:**

```
.agentic/documentation/
├── ARCHITECTURE.md          # System overview
├── DATA_MODELS.md           # All data structures
├── components/
│   ├── auth.md              # Per-component docs
│   ├── api.md
│   └── ...
└── diagrams/
    ├── architecture.mmd     # Mermaid component diagram
    └── flow_*.mmd           # Sequence diagrams
```

**Key Principle:** DocWriter performs ZERO file reads. All content comes from memory queries. This ensures documentation reflects what the agent actually understood, not live code.

## Execution Flow

### Full Documentation Run

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         run_full_documentation()                           │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Phase A:       │      │  Phase B:       │      │  Phase C:       │
│  Architecture   │ ───► │  Components     │ ───► │  Cross-Cutting  │
│  Discovery      │      │  (loop each)    │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
   ┌───────────┐           ┌───────────┐           ┌───────────┐
   │ Store:    │           │ Store:    │           │ Store:    │
   │ Arch      │           │ Component │           │ Cross-    │
   │ Overview  │           │ Summaries │           │ Cutting   │
   └───────────┘           │ FileRoles │           │ Concerns  │
                           └───────────┘           └───────────┘
                                    │
                                    ▼
                        ┌─────────────────┐
                        │  Phase D:       │
                        │  Doc Generation │
                        │  (memory only)  │
                        └─────────────────┘
                                    │
                                    ▼
                        ┌─────────────────┐
                        │  DocWriter      │
                        │  generate_all() │
                        └─────────────────┘
                                    │
                                    ▼
                        ┌─────────────────┐
                        │  Output:        │
                        │  .agentic/      │
                        │  documentation/ │
                        └─────────────────┘
```

### Single Phase Loop

```
for round in range(max_rounds_per_phase):  # Default: 30

    1. Build messages:
       - System prompt (base + phase-specific)
       - User message (context + previous results)
       - Conversation history (last 3 exchanges)

    2. Call LLM (await llm.chat(messages))

    3. Parse response:
       - Tool calls? → Execute tools, add results to conversation, continue
       - Phase complete? → Store findings, exit loop
       - Neither? → Add "Continue or signal completion", continue

    4. Log round to debug file
```

## Memory System Deep Dive

### Why Schema-Driven Memory?

1. **Consistency**: Every entry has the same structure
2. **Queryability**: Filter by component, layer, confidence, etc.
3. **Bounded Retrieval**: Prevents loading entire memory into context
4. **Validation**: Invalid entries are rejected at storage time
5. **Change Detection**: File hashes enable incremental updates

### Storage Layout

```
.agentic/memory/
├── index.json              # Phase tracking, component lists
├── architecture/
│   └── overview.json       # Single architecture overview
├── components/
│   ├── auth.json           # One file per component
│   ├── api.json
│   └── database.json
├── files/
│   ├── src_auth_login.json # One file per documented file
│   └── ...
├── data_models/
│   ├── user.json
│   └── session.json
├── flows/
│   └── authentication.json
└── cross_cutting/
    ├── logging.json
    └── config.json
```

### Memory Index

Tracks global state for fast lookups:

```json
{
  "version": "1.0",
  "current_phase": "component_deep_dive",
  "phase_progress": {},
  "components_discovered": ["auth", "api", "database"],
  "components_explored": ["auth"],
  "cross_cutting_found": ["logging"],
  "file_count": 15,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### Change Detection

Each FileRole stores a hash of the file contents:

```python
def store_file_role(self, file_role: FileRole):
    file_role.last_read_hash = self._hash_file(file_role.file_path)
    # ...

def get_changed_files(self) -> list[str]:
    # Compare stored hashes with current file hashes
    # Return files where hash differs
```

This enables `update` mode to selectively re-explore changed code.

## Budget System Deep Dive

### Why Budgets?

Without budgets, an LLM could:
- Read every file in a large codebase (100k+ tokens)
- Run 50 grep searches loading massive context
- Load entire memory into a single prompt

Budgets force the agent to:
1. **Prioritize** - Read the most important files first
2. **Summarize** - Store findings before budget exhaustion
3. **Be strategic** - Use grep to find targets, then read selectively

### Budget Exhaustion Flow

```
1. LLM requests: read_file("src/utils/helpers.py")

2. DocTools.read_file() calls:
   orchestrator.check_budget("file_read")

3. If budget exhausted:
   return ToolResult(
       success=False,
       content="BUDGET_EXHAUSTED: File read limit (10) reached.
                Summarize findings and store to memory."
   )

4. LLM receives this message and must:
   - Stop trying to read more files
   - Call store_discovery() with current findings
   - Signal phase_complete
```

### Budget Warnings

When budget is running low (2 remaining), tools return warnings:

```python
def _get_budget_warning(self, operation: str) -> Optional[str]:
    if left <= 2:
        return f"WARNING: Only {left} file reads remaining. Consider summarizing."
```

## CLI Integration

### Running Documentation

```bash
# Full documentation run
agentic document /path/to/repo

# Resume interrupted session
agentic document /path/to/repo --resume

# Update for changed files only
agentic document /path/to/repo --update
```

### Progress Output

The CLI receives status callbacks during execution:

```
[Phase: architecture_discovery] Starting...
[Phase: architecture_discovery] Round 1/30 - Calling LLM...
[Phase: architecture_discovery] Executing tools: list_files, grep
[Phase: architecture_discovery] Round 2/30 - Calling LLM...
[Phase: architecture_discovery] Executing tools: read_file
[Phase: architecture_discovery] Complete!
[Component 1/5] Exploring: auth
...
```

### Debug Logs

Each session produces debug files:

| File | Contents |
|------|----------|
| `{session_id}.json` | Orchestrator state (phase, progress, budgets) |
| `{session_id}_debug.txt` | Incremental log of all operations |
| `{session_id}_context.txt` | Full LLM context snapshot (last round) |

## Configuration

### DocAgentConfig

```python
@dataclass
class DocAgentConfig:
    max_rounds_per_phase: int = 30    # Max LLM calls per phase
    max_tool_calls_per_round: int = 5 # Max tools per LLM response
    debug_logging: bool = True        # Write debug files
    on_status: callable = None        # Progress callback
```

### Customizing Budgets

To adjust budgets, modify `PHASE_BUDGETS` in `doc_phases.py`:

```python
PHASE_BUDGETS = {
    Phase.ARCHITECTURE_DISCOVERY: PhaseBudget(
        max_files_read=20,   # Increase for larger codebases
        max_grep_calls=25,
        max_symbols_calls=10,
    ),
    # ...
}
```

## Extending the System

### Adding a New Memory Type

1. Add enum value to `MemoryEntryType`
2. Create dataclass schema in `doc_memory.py`
3. Add to `SCHEMA_MAP` and `TYPE_DIRS`
4. Create `store_*` method
5. Update prompts to reference new type

### Adding a New Tool

1. Add method to `DocTools` class
2. Add to `TOOL_DEFINITIONS` string
3. Add case in `_execute_single_tool()` in `doc_agent.py`
4. Update prompts if tool should be used in specific phases

### Adding a New Phase

1. Add enum value to `Phase`
2. Create budget in `PHASE_BUDGETS`
3. Add phase prompt in `doc_prompts.py`
4. Add context builder function
5. Add execution logic in `doc_agent.py`

## Limitations and Future Work

### Current Limitations

1. **No Semantic Search**: Memory queries use exact matching, not embeddings
2. **Sequential Components**: Components are explored one at a time
3. **Fixed Budgets**: No dynamic adjustment based on codebase size
4. **Single LLM**: No multi-agent parallelization

### Potential Improvements

1. **Embedding-based Memory**: Semantic similarity for memory queries
2. **Parallel Component Exploration**: Multiple sub-agents for Phase B
3. **Adaptive Budgets**: Scale limits based on repo size
4. **Incremental Updates**: Smarter change detection and partial re-exploration
5. **Custom Output Formats**: Support for other doc formats (RST, AsciiDoc)
