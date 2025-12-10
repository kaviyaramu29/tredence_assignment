# Mini Agent Workflow Engine (FastAPI, Single-File Implementation)

This repository contains a minimal yet functional **agent workflow engine** implemented in a
single FastAPI file: `workflow_engine_full.py`.

The engine allows you to define a simple directed graph of nodes (steps), run them with a
shared state dictionary, and support branching + looping. It includes a complete sample
workflow implementing **Option A: Code Review Mini-Agent** from the assignment.

The goal of this project is clarity, correctness, and structure — not completeness or scale.

---

##  How to Run the Project

### 1. Install dependencies
pip install fastapi uvicorn pydantic

2. Start the FastAPI server
uvicorn workflow_engine_full:app --reload --host 0.0.0.0 --port 8000

3. Open API documentation

Navigate to:

http://localhost:8000/docs


This gives you an interactive Swagger UI for testing all endpoints.

Project Structure

Since this is a single-file implementation, everything lives in:

workflow_engine_full.py


It contains:

API routes

Graph creation logic

Graph execution engine

Tool registry

Built-in code review workflow

Looping + branching system

In-memory graph + run storage

Execution logs

This keeps the submission small and easy to review.

 Example Workflow (Included)

The included workflow is Option A: Code Review Mini-Agent, which performs:

Extract functions

Compute complexity

Detect issues

Suggest improvements

Compute quality score

Loop until quality_score >= threshold

This workflow is pre-registered with the ID: code_review_A

Run the Sample Workflow

Use this curl command (Windows PowerShell or Linux/macOS):

curl -X POST "http://localhost:8000/graph/run" \
  -H "Content-Type: application/json" \
  -d "{\"graph_id\":\"code_review_A\",\"initial_state\":{\"code\":\"def foo(x):\n    print(x)\n    return x+1\n\",\"threshold\":0.85},\"options\":{}}"


This returns:

run_id

final state (functions extracted, issues found, quality score)

execution log

Check run state

Replace <RUN_ID> with the returned ID:

curl "http://localhost:8000/graph/state/<RUN_ID>"

🧠 Engine Capabilities
✔ Nodes

Each node is a Python “tool” function that reads/modifies a shared state dict.

✔ Edges

Structured as source → target.
Optional condition enables branching.

✔ Branching

Conditional routing using expressions like:

state.get("quality_score", 0) < 0.85

✔ Looping

Edges may point backwards.
Loop is controlled by:

node-level iteration limit

global step limit

✔ Tool Registry

Simple dictionary mapping tool names → Python callables.

✔ In-Memory Storage

Graphs and run states stored in Python dictionaries.

✔ Execution Logging

Every node run is logged for observability.

Example Features in the Code Review Workflow
Tools implemented:

extract_functions

check_complexity

detect_issues

suggest_improvements

compute_quality

Loop condition:
state.get('quality_score', 0) < state.get('threshold', 0.8)


If true → workflow loops back to suggest
If false → workflow ends
