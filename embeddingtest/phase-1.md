# Agentic CLI Coding System — Phase 1 (Design & Planning)

## Purpose

Build a highly agentic, long-running CLI-based coding system that exceeds Cursor’s current capabilities by:

* Separating **planning, execution, and verification**
* Minimising context pollution via **task-scoped sub-agents**
* Persisting state, plans, attempts, and results to disk
* Supporting iterative retries informed by past failures

Phase 1 focuses on **architecture, planning workflow, and artefacts**, not full automation.

---

## Core Design Principles

1. **Master–Sub-Agent Architecture**

   * One persistent **Master Agent** controls flow and state
   * Multiple short-lived **Sub-Agents**, each scoped to a single task

2. **Context Isolation**

   * Each sub-agent only receives:

     * Its specific task
     * Minimal required code context
     * Relevant prior attempts / failures
   * Prevents needle-in-haystack degradation

3. **Disk as Source of Truth**

   * Plans, task lists, progress, and attempt logs are written to files
   * System can resume after interruption

4. **Explicit Planning Before Execution**

   * No code edits until requirements are clarified and frozen

---

## High-Level Workflow (Phase 1)

1. User provides a **high-level prompt**
2. Master Agent:

   * Clarifies requirements
   * Asks follow-up questions
   * Reads repository structure (no edits)
3. Master Agent generates a **Planning Document**
4. User reviews and approves the plan
5. Plan is locked and written to disk

> Phase 1 ends here. No sub-agent execution yet.

---

## Master Agent Responsibilities (Phase 1)

* Interpret and reframe user intent
* Ask clarification questions
* Perform repo reconnaissance:

  * File tree
  * Key entry points
  * Test layout
* Produce a **structured task plan**
* Persist artefacts to disk

The Master Agent **never edits code directly**.

---

## Planning Document (Primary Output)

Saved as:

```
.agentic/plan.md
```

### Required Sections

```markdown
# Project Goal
Concise restatement of what is being built or changed.

# Non-Goals
Explicitly excluded behaviour or scope.

# Assumptions
Any assumptions made due to missing info.

# Constraints
Performance, compatibility, style, or architectural constraints.

# Task Breakdown
## Task 1: <title>
- Description
- Expected files touched
- Risks / unknowns

## Task 2: <title>
...

# Test Strategy
- Existing tests to run
- New tests required

# Success Criteria
Objective definition of “done”.
```

---

## Clarification Loop

Before writing `plan.md`, the Master Agent may:

* Ask questions interactively
* Propose assumptions and request confirmation

Only when the user explicitly approves does the plan become **immutable**.

---

## Repo Access Rules (Phase 1)

Allowed:

* Read-only file access
* Directory listing
* Test discovery

Disallowed:

* Writing files (except `.agentic/*`)
* Running tests
* Applying patches

---

## Disk State Layout (Initial)

```
.agentic/
├── plan.md          # Approved task plan
├── status.md        # Overall progress (empty in Phase 1)
└── meta.json        # Timestamps, model versions, config
```

---

## Why This Improves Quality

* Forces **thinking before acting**
* Eliminates premature code edits
* Produces an auditable, reviewable plan
* Enables deterministic execution in later phases

---

## Phase 1 Exit Criteria

Phase 1 is complete when:

* A reviewed and approved `plan.md` exists
* Tasks are clearly separable and scoped
* No ambiguity remains that would block execution

---

## What Comes Next (Phase 2 Preview)

* Sub-agent execution per task
* Context window synthesis
* Attempt tracking and retries
* Test-driven verification

Phase 2 will consume `plan.md` as its sole input.
