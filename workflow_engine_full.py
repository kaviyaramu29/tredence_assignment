"""
Simple Agent Workflow Engine - Full implementation (Workflow A: Code Review Mini-Agent)

Single-file FastAPI app. Features:
- Create graphs (POST /graph/create)
- Run graphs synchronously (POST /graph/run) -> returns run_id, final state, and execution log
- Query run state (GET  /graph/state/{run_id}) while or after execution
- In-memory persistence for graphs, runs, and a tool registry
- Supports branching (conditional edges) and looping with loop-guard
- Pre-registered sample graph 'code_review_A' implementing Option A

How to run:
    pip install fastapi uvicorn pydantic
    uvicorn workflow_engine_full:app --reload --port 8000

API quick examples (using the pre-registered sample):
1) Start the sample run:
   curl -s -X POST "http://127.0.0.1:8000/graph/run" -H "Content-Type: application/json" -d \
     '{"graph_id":"code_review_A","initial_state":{"code":"def foo(x):\n    return x+1\n"},"options":{}}' | jq

2) Query run state (replace RUN_ID with response):
   curl "http://127.0.0.1:8000/graph/state/RUN_ID" | jq

Design notes:
- Nodes are executed by looking up a registered 'tool' function (or built-in runner)
- Edges support optional `condition` field which is a small expression evaluated against the state
  Example condition: "state.get('quality_score',0) >= 0.8"
- Looping is achieved by edges that point back to earlier nodes and conditions that keep the loop
- To avoid infinite loops, runs include a `max_iterations` guard per node and global step limit

This implementation emphasizes clarity over performance or security. It's suitable for the assignment
and local testing.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4
import asyncio
import time

app = FastAPI(title="Mini Agent Workflow Engine")

# -------------------------------
# In-memory stores
# -------------------------------
GRAPHS: Dict[str, Dict[str, Any]] = {}    # graph_id -> graph
RUNS: Dict[str, Dict[str, Any]] = {}      # run_id -> run metadata + state
TOOLS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}  # tool registry

# -------------------------------
# Pydantic models for API
# -------------------------------
class NodeDef(BaseModel):
    id: str
    type: str = Field(..., description="tool | builtin")
    tool: Optional[str] = None
    params: Optional[Dict[str, Any]] = {}

class EdgeDef(BaseModel):
    source: str
    target: str
    condition: Optional[str] = None  # python expression evaluated with `state` in locals

class GraphCreateRequest(BaseModel):
    id: Optional[str] = None
    nodes: List[NodeDef]
    edges: List[EdgeDef]

class RunRequest(BaseModel):
    graph_id: str
    initial_state: Dict[str, Any] = {}
    options: Optional[Dict[str, Any]] = {}

class RunStateResponse(BaseModel):
    run_id: str
    status: str
    state: Dict[str, Any]
    log: List[str]

# -------------------------------
# Utility helpers
# -------------------------------

def safe_eval_condition(cond: str, state: Dict[str, Any]) -> bool:
    """Evaluate a small condition expression safely-ish.
    Allowed names: state, True, False, None
    This is intentionally minimal — do NOT use with untrusted inputs in production.
    """
    allowed_globals = {"__builtins__": {}}
    allowed_locals = {"state": state}
    try:
        return bool(eval(cond, allowed_globals, allowed_locals))
    except Exception:
        return False

# -------------------------------
# Tool implementations (sample tools for Code Review)
# -------------------------------

def tool_extract_functions(state: Dict[str, Any]) -> Dict[str, Any]:
    code = state.get("code", "")
    # Very naive function extractor: lines starting with 'def '
    funcs = []
    cur = []
    in_func = False
    for line in code.splitlines():
        if line.strip().startswith("def "):
            if in_func:
                funcs.append("\n".join(cur))
                cur = []
            in_func = True
            cur.append(line)
        elif in_func:
            # stop on blank line with less indent or next def; simple heuristic
            if line.strip() == "":
                in_func = False
                funcs.append("\n".join(cur))
                cur = []
            else:
                cur.append(line)
    if cur:
        funcs.append("\n".join(cur))
    state.setdefault("functions", funcs)
    return {"functions": funcs}


def tool_check_complexity(state: Dict[str, Any]) -> Dict[str, Any]:
    funcs = state.get("functions", [])
    complexities = {}
    for i, f in enumerate(funcs):
        # heuristic: complexity = number of branches ('if','for','while') + lines
        lines = f.count("\n") + 1
        branches = sum(f.count(k) for k in ("if ", "for ", "while ", "elif "))
        score = lines + branches * 2
        complexities[f"func_{i}"] = {"lines": lines, "branches": branches, "complexity_score": score}
    state.setdefault("complexities", complexities)
    return {"complexities": complexities}


def tool_detect_issues(state: Dict[str, Any]) -> Dict[str, Any]:
    funcs = state.get("functions", [])
    issues = []
    for i, f in enumerate(funcs):
        if 'print(' in f:
            issues.append({"func": f"func_{i}", "issue": "debug_prints"})
        if 'TODO' in f or 'FIXME' in f:
            issues.append({"func": f"func_{i}", "issue": "todo_comment"})
        # naive: functions with > 50 chars in a single line flagged
        for ln in f.splitlines():
            if len(ln) > 120:
                issues.append({"func": f"func_{i}", "issue": "long_line"})
                break
    state.setdefault("issues", issues)
    return {"issues": issues}


def tool_suggest_improvements(state: Dict[str, Any]) -> Dict[str, Any]:
    issues = state.get("issues", [])
    suggestions = []
    for it in issues:
        if it["issue"] == "debug_prints":
            suggestions.append({"fix": "remove_prints_or_use_logger", "desc": "Replace print statements with logging"})
        elif it["issue"] == "todo_comment":
            suggestions.append({"fix": "address_todo", "desc": "Resolve TODO/FIXME comments or add a test"})
        elif it["issue"] == "long_line":
            suggestions.append({"fix": "wrap_long_line", "desc": "Break long lines to < 120 chars"})
    # simple heuristic quality delta
    state.setdefault("suggestions", suggestions)
    return {"suggestions": suggestions}


def tool_compute_quality(state: Dict[str, Any]) -> Dict[str, Any]:
    # Produce a quality_score between 0 and 1 based on number of issues and complexity
    complexities = state.get("complexities", {})
    issues = state.get("issues", [])
    avg_complexity = 0
    if complexities:
        vals = [v.get("complexity_score", 0) for v in complexities.values()]
        avg_complexity = sum(vals) / len(vals)
    issue_penalty = min(len(issues) * 0.1, 1.0)
    complexity_penalty = min(avg_complexity * 0.01, 0.5)
    quality = max(0.0, 1.0 - issue_penalty - complexity_penalty)
    state["quality_score"] = quality
    return {"quality_score": quality}

# Register tools
TOOLS["extract_functions"] = tool_extract_functions
TOOLS["check_complexity"] = tool_check_complexity
TOOLS["detect_issues"] = tool_detect_issues
TOOLS["suggest_improvements"] = tool_suggest_improvements
TOOLS["compute_quality"] = tool_compute_quality

# -------------------------------
# Graph execution engine
# -------------------------------

def execute_graph(graph: Dict[str, Any], initial_state: Dict[str, Any], run_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous execution engine for a graph.
    - graph: contains nodes (id->nodedef) and edges list
    - initial_state: starting shared state
    - run_id: id for storing progress in RUNS
    Returns final state and log
    """
    # Prepare run metadata
    run = RUNS[run_id]
    state = dict(initial_state)  # shared state
    log: List[str] = []

    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    # compute adjacency
    adj: Dict[str, List[EdgeDef]] = {}
    for e in edges:
        adj.setdefault(e["source"], []).append(e)

    # choose a start node(s): nodes with no incoming edges, or graph.start if provided
    incoming = {n: 0 for n in nodes}
    for e in edges:
        incoming[e["target"]] = incoming.get(e["target"], 0) + 1
    start_nodes = [nid for nid, deg in incoming.items() if deg == 0]
    if not start_nodes:
        # fallback to first node
        start_nodes = [list(nodes.keys())[0]]

    # Execution queue: list of node ids to execute
    queue: List[str] = start_nodes.copy()

    # loop guards
    node_iters: Dict[str, int] = {nid: 0 for nid in nodes}
    max_iters_per_node = options.get("max_iters_per_node", 10)
    global_step_limit = options.get("global_step_limit", 500)
    steps = 0

    while queue:
        if steps >= global_step_limit:
            log.append(f"Global step limit {global_step_limit} reached, aborting")
            run["status"] = "aborted"
            break
        nid = queue.pop(0)
        node = nodes[nid]
        node_iters[nid] = node_iters.get(nid, 0) + 1
        if node_iters[nid] > max_iters_per_node:
            log.append(f"Node {nid} exceeded max iterations ({max_iters_per_node}), skipping")
            continue

        # execute node
        log.append(f"Executing node {nid} (tool={node.get('tool')})")
        run["last_executed"] = nid
        try:
            if node["type"] == "tool":
                tool_name = node.get("tool")
                tool_fn = TOOLS.get(tool_name)
                if not tool_fn:
                    log.append(f"Missing tool: {tool_name}")
                else:
                    out = tool_fn(state)
                    log.append(f"Node {nid} output: {out}")
            elif node["type"] == "builtin":
                # reserved for future builtin node types
                log.append(f"Builtin node {nid} (noop)")
            else:
                log.append(f"Unknown node type for {nid}: {node.get('type')}")
        except Exception as exc:
            log.append(f"Error executing node {nid}: {exc}")

        # persist intermediate state
        run["state"] = dict(state)
        run["log"] = log.copy()

        # determine next nodes based on edges and conditions
        next_candidates = adj.get(nid, [])
        taken_any = False
        for e in next_candidates:
            cond = e.get("condition")
            if not cond:
                queue.append(e["target"])
                taken_any = True
            else:
                if safe_eval_condition(cond, state):
                    queue.append(e["target"])
                    taken_any = True
        # If no outgoing edges matched (dead end), continue
        steps += 1

    # finalize run
    if run.get("status") != "aborted":
        run["status"] = "finished"
    run["state"] = state
    run["log"] = log
    return {"state": state, "log": log}

# -------------------------------
# API Endpoints
# -------------------------------

@app.post("/graph/create")
async def create_graph(req: GraphCreateRequest):
    gid = req.id or f"graph_{uuid4().hex[:8]}"
    if gid in GRAPHS:
        raise HTTPException(status_code=400, detail="graph id already exists")
    # store minimal graph representation
    graph = {"id": gid, "nodes": [n.dict() for n in req.nodes], "edges": [e.dict() for e in req.edges]}
    GRAPHS[gid] = graph
    return {"graph_id": gid}

@app.post("/graph/run")
async def run_graph(req: RunRequest):
    graph = GRAPHS.get(req.graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail="graph not found")
    run_id = f"run_{uuid4().hex[:8]}"
    RUNS[run_id] = {"graph_id": req.graph_id, "status": "running", "state": dict(req.initial_state), "log": [], "last_executed": None}

    # execute synchronously (keeps API simple). For larger workloads this could be backgrounded.
    result = execute_graph(graph, req.initial_state, run_id, req.options or {})
    return {"run_id": run_id, "status": RUNS[run_id]["status"], "state": result["state"], "log": result["log"]}

@app.get("/graph/state/{run_id}")
async def get_run_state(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run id not found")
    return {"run_id": run_id, "status": run.get("status"), "state": run.get("state"), "log": run.get("log", [])}

@app.get("/graphs")
async def list_graphs():
    return {"graphs": list(GRAPHS.keys())}

@app.get("/runs")
async def list_runs():
    return {"runs": list(RUNS.keys())}

# -------------------------------
# Pre-register a sample graph for Workflow A (Code Review Mini-Agent)
# -------------------------------

def make_sample_code_review_graph():
    gid = "code_review_A"
    nodes = [
        {"id": "extract", "type": "tool", "tool": "extract_functions"},
        {"id": "complexity", "type": "tool", "tool": "check_complexity"},
        {"id": "detect", "type": "tool", "tool": "detect_issues"},
        {"id": "suggest", "type": "tool", "tool": "suggest_improvements"},
        {"id": "score", "type": "tool", "tool": "compute_quality"},
    ]
    # edges: extract -> complexity -> detect -> suggest -> score
    # branching: after score, if quality_score >= threshold -> END, else -> suggest (loop)
    edges = [
        {"source": "extract", "target": "complexity"},
        {"source": "complexity", "target": "detect"},
        {"source": "detect", "target": "suggest"},
        {"source": "suggest", "target": "score"},
        # loop back to suggest if score < threshold
        {"source": "score", "target": "suggest", "condition": "state.get('quality_score',0) < state.get('threshold', 0.8)"},
    ]
    graph = {"id": gid, "nodes": nodes, "edges": edges}
    GRAPHS[gid] = graph

# create sample graph at startup
make_sample_code_review_graph()

# -------------------------------
# If run as script, print quick instructions
# -------------------------------
if __name__ == "__main__":
    print("This module implements a FastAPI app. Run with: uvicorn workflow_engine_full:app --reload --port 8000")
