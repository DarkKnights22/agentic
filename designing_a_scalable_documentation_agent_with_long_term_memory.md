# Designing a Scalable Documentation Agent with Long‑Term Memory

This document explains how to redesign the Documentation Agent so it can handle **large codebases**, avoid context exhaustion, and produce **accurate, deep, maintainable documentation** using agentic memory principles.

---

## 1. Problem Summary (Why the Current Design Fails)

The current Documentation Agent follows a common but fragile pattern:

- A directory tree is dumped into the prompt upfront (even if truncated)
- A single agent is responsible for exploration, understanding, memory, and documentation
- Knowledge is retained implicitly by keeping information in the context window
- Once the run ends, all understanding is lost

This causes:

- Rapid context exhaustion
- Missed files and incomplete understanding
- Recency bias (earlier discoveries are forgotten)
- Hallucinated architectural relationships

**Root cause:** context is being used as memory.

---

## 2. Core Mental Shift

> Documentation is not a single task.
> It is a long‑running discovery → extraction → synthesis process.

Key implications:

- Context ≠ Memory
- Bigger context windows do not solve understanding
- Understanding must be **earned**, **structured**, and **stored durably**

The Documentation Agent must behave like a **compiler**, not a reader.

---

## 3. Target Architecture (Tiered Memory Model)

The system should follow a tiered model similar to modern computer architecture:

- **Working Context** – minimal, per‑LLM‑call view
- **Session Logs** – structured event history of actions
- **Long‑Term Memory** – durable, searchable extracted knowledge
- **Artifacts** – large outputs (docs, diagrams, summaries)

Only the **working context** is passed to the LLM. Everything else is retrieved on demand.

---

## 4. Step‑by‑Step Redesign

### Step 1: Eliminate the Repository Tree Dump

The agent must never receive the full directory tree in context.

Initial context should include only:

- Repository root
- Package manager / build files (e.g. package.json, pyproject.toml, go.mod)
- README (if present)

These files act as architectural entry points.

---

### Step 2: Introduce a Repo Index (Non‑LLM)

Before the agent runs, generate a **machine‑built index** of the repository:

- File paths
- File sizes
- Languages
- Import relationships
- Directory metadata

This index:

- Is **not** placed into LLM context
- Is accessed only through tools
- Allows the agent to decide *what* to inspect

This alone reduces context usage dramatically.

---

### Step 3: Replace “Explore Everything” with Discovery Phases

The agent operates in structured phases, not a single run.

---

#### Phase A: Architecture Discovery

Goal: Identify the high‑level structure of the system.

Agent tasks:

- Identify entry points
- Identify major components/services
- Identify high‑level data and control flow

Allowed tools:

- Open specific files
- Search imports
- Query directory metadata

**Output → Long‑Term Memory (structured):**

- Architecture overview
- Component list
- Responsibility boundaries

---

#### Phase B: Component Deep Dives

Each component is explored in **complete isolation**.

For each component:

- Run a scoped agent pass
- Load only files relevant to that component
- Never mix components in the same context

**Output → Long‑Term Memory:**

- Component summary
- Key files
- Public interfaces
- Dependencies
- Runtime behavior

---

#### Phase C: Cross‑Cutting Concerns

Handled separately from components:

- Authentication
- Configuration
- Logging
- Data models
- Error handling

Discovered via targeted searches and dependency fan‑out.

Stored as structured long‑term memory entries.

---

## 5. Long‑Term Memory Design (Critical)

### Schema‑Driven Memory

Memory must never be stored as raw text blobs.

Each memory entry uses an explicit schema, for example:

- architecture_overview
- component_summary
- file_role
- data_model
- runtime_flow
- config_concern

Example:

- File role
- Path
- Responsibility
- Used‑by relationships

This makes memory:

- Searchable
- Auditable
- Reversible
- Composable

---

### Explicit Retrieval

Before any reasoning or generation step, the agent must ask:

> What memory do I need *right now*?

Memory is retrieved deliberately, never passively carried forward.

---

## 6. Documentation Generation Phase

Documentation is generated **only after discovery is complete**.

At this stage:

- No raw code is in context
- No logs are in context
- Only structured memory is loaded

Artifacts produced:

- ARCHITECTURE.md
- Component documentation
- Data model documentation
- Diagrams (e.g. Mermaid)

Each artifact is traceable back to memory entries.

---

## 7. Optional: Sub‑Agent Decomposition

For larger systems, responsibilities can be split:

- Explorer Agent – finds candidate files
- Summarizer Agent – extracts structured memory
- Verifier Agent – checks consistency
- Documentation Writer – consumes memory only

Agents communicate via memory and artifacts, not shared transcripts.

---

## 8. Immediate Next Actions

To move from the current design to this architecture:

1. Remove directory tree dumps from all prompts
2. Implement schema‑based long‑term memory storage
3. Enforce component‑scoped discovery runs

Even these three changes will significantly improve accuracy and scalability.

---

## 9. Litmus Test for Correct Design

The system is correctly designed if:

- The working context can be wiped between steps
- The agent can restart
- And it still understands the architecture

If not, context is still being misused as memory.

---

This architecture turns the Documentation Agent from a brittle demo into a **production‑ready, long‑running, self‑consistent system** capable of handling large and evolving codebases.

