# Agentic CLI Coding System — Phase 2 (Execution & Sub‑Agent Orchestration)

## Purpose

Phase 2 introduces **actual code execution** through task‑scoped sub‑agents, while preserving the planning guarantees established in Phase 1.

The system now:

* Executes tasks defined in `plan.md`
* Spins up **isolated sub‑agents per task**
* Tracks attempts, failures, and successes
* Runs tests and reports results back to the Master Agent
* Persists all progress to disk

---

## Core Objectives (Phase 2)

1. Deterministic execution of the approved plan
2. Context‑minimal sub‑agents for higher accuracy
3. Robust retry and recovery loop
4. Full observability via disk‑backed state

---

## High‑Level Workflow (Phase 2)

1. Master Agent loads `.agentic/plan.md`
2. Tasks are processed **sequentially** (parallelism deferred)
3. For each task:

   * A scoped sub‑agent is spawned
   * Minimal context is assembled
   * Code changes are proposed and applied
   * Tests are run
   * Results are logged
4. Master Agent updates global status
5. Execution proceeds to next task or halts on hard failure

---

## Sub‑Agent Model

Each sub‑agent is:

* Stateless across tasks
* Context‑scoped to a single task
* Fully disposable after completion

### Sub‑Agent Receives

* Task definition (from `plan.md`)
* Relevant file contents only
* Relevant previous attempts (if retries)
* System constraints and coding standards

### Sub‑Agent Does NOT Receive

* Full repository
* Unrelated tasks
* Long chat history

---

## Sub‑Agent Responsibilities

For a single task:

1. Read required files
2. Propose code changes via **unified diffs**
3. Apply patches (through tool calls)
4. Request test execution
5. Analyse failures and retry if needed
6. Produce a structured report

Sub‑agents never modify planning documents.

---

## Attempt & Retry Strategy

Each task maintains an attempt log:

```
Task N
Attempt 1: Failed – test_x failed
Attempt 2: Failed – regression in test_y
Attempt 3: Success
```

Rules:

* Max attempts configurable (default: 3)
* Each retry receives:

  * Previous diffs
  * Test output
  * Failure analysis

---

## Master Agent Responsibilities (Phase 2)

* Load and interpret `plan.md`
* Spawn and supervise sub‑agents
* Decide retry vs abort
* Merge successful outcomes into global state
* Maintain execution order and integrity

The Master Agent **still does not write code directly**.

---

## Tooling (Phase 2)

### Allowed Tools

* `read_file(path)`
* `search_files(query)`
* `apply_patch(path, diff)`
* `run_tests(command)`

### Hard Rules

* All file edits must be diffs
* All edits must follow approved plan
* Tests must be run before task completion

---

## Disk State Layout (Expanded)

```
.agentic/
├── plan.md
├── status.md
├── meta.json
├── tasks/
│   ├── task_01.md
│   ├── task_02.md
│   └── ...
├── attempts/
│   ├── task_01_attempt_1.md
│   ├── task_01_attempt_2.md
│   └── ...
└── logs/
    ├── task_01_tests.log
    └── ...
```

---

## Task Status Tracking

`status.md` is continuously updated:

```markdown
- Task 1: completed (3 attempts)
- Task 2: in progress
- Task 3: pending
```

This file is the **single source of truth** for progress.

---

## Failure Handling

Hard stop conditions:

* Repeated test failures beyond retry limit
* Violation of plan constraints
* Unsafe file operations

On halt:

* System preserves all state
* User intervention required to continue

---

## Why This Works Better Than Cursor

* No shared polluted context
* Explicit retry memory
* Deterministic task boundaries
* Resume‑safe execution
* Auditable decision history

---

## Phase 2 Exit Criteria

Phase 2 is complete when:

* All tasks in `plan.md` are completed
* All tests pass
* `status.md` reflects full completion

---

## Phase 3 Preview

* Parallel sub‑agents
* Automatic task re‑ordering
* Heuristic context synthesis
* Cost‑aware execution planning

Phase 2 lays the foundation for true long‑running, high‑accuracy agentic coding.
