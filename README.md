# qprompt-a2a

Renders a [qprompt](../qprompt-dsl) graph into a Docker Compose project where
every agent is its own service, speaking the
[A2A protocol](https://a2a-protocol.org/) to each other, with LLM-kind agents
wired to [Docker Model Runner](https://docs.docker.com/ai/compose/models-and-compose/)
via Compose's `models:` key.

```
qprompt-dsl (TS/Langium)  --generate--> qprompt-ir.schema.json JSON
                                              |
                                              v
                              qprompt-a2a (this repo, pure Python)
                                              |
                                              v
                        docker-compose.yml + one A2A service per agent
                        + one generated orchestrator service
```

## Why a separate repo from qprompt-langgraph

Both consume the same versioned IR (see
[`../qprompt-dsl/schema/README.md`](../qprompt-dsl/schema/README.md)), but
render fundamentally different shapes: qprompt-langgraph emits one Python
process running a single `StateGraph`; this emits N independent, separately
deployable services. It's not "LangGraph with extra steps" -- it's a
different codegen target against the same source of truth, so it gets its
own repo, the same way qprompt-langgraph did.

## Who owns routing

A workflow's `routes:`/`on_complete:` are owned by one generated
**orchestrator** service, not spread across agents -- each agent stays a
simple, single-purpose A2A service with no knowledge of what calls it or
what gets called next; only the orchestrator knows the graph shape. This is
the same role qprompt-langgraph's generated `build_graph()` plays, just
distributed over the network instead of in-process. The orchestrator is
deliberately a plain HTTP trigger (`POST /run`), not an A2A agent itself --
A2A models communication *between* agents, and the orchestrator is
codegen-synthesized infrastructure, not one of the graph's declared agents.

## Reuse, not duplication

The per-agent-kind execution logic (`evaluate_rules`, `run_container` + its
adapter registry, `resolve_state_assignments`) is imported directly from
`qprompt_langgraph.runtime` inside every generated agent service -- none of
it is LangGraph-specific. The only genuinely new runtime logic here is
`call_llm` (backed by whatever model Compose injects for that service, since
this target commits to a concrete model runtime unlike qprompt-langgraph's
deliberate stub) and the A2A server/client boilerplate, both in
`qprompt_a2a/runtime.py`.

## Layout

```
src/qprompt_a2a/
  codegen/
    view.py        Builds a template-friendly ProjectView on top of
                    qprompt_langgraph's WorkflowView (state merging, route
                    translation) -- reused as-is, not re-implemented.
    env.py          Jinja Environment + filters (reuses qprompt_langgraph's
                    py_literal/py_triple_quoted/screaming_snake filters).
    generator.py    render_project() / write_project()
  templates/
    agent_service.py.jinja        One A2A service per agent, dispatch by
                                   agent kind mirrors qprompt-langgraph's
                                   graph.py.jinja node bodies.
    orchestrator.py.jinja          Owns routing; calls each agent via A2A.
    Dockerfile.agent.jinja         Shared by every agent service (parameterized
                                   by a SERVICE_NAME build arg); includes the
                                   Docker CLI since container-kind agents'
                                   run_container shells out to the host
                                   daemon via a mounted socket ("sibling
                                   containers", not Docker-in-Docker). A
                                   second build arg, NEEDS_RAG, gates
                                   qprompt-langgraph's [rag] extra
                                   (LangChain/FAISS/sentence-transformers) --
                                   confirmed against a real build this isn't
                                   docker-cli-sized (~5.5GB per image), so
                                   only services whose agent has non-empty
                                   context: pay for it.
    Dockerfile.orchestrator.jinja
    docker-compose.yml.jinja       For an llm-kind service with non-empty
                                   context:, also sets QPROMPT_RAG_ROOT and
                                   bind-mounts this project's own directory
                                   read-only at /rag -- same rag_root
                                   resolution `qprompt-langgraph index` uses
                                   when run directly against this directory
                                   on the host, no container-only special
                                   case.
  runtime.py         call_llm (Docker Model Runner-backed), render_prompt,
                     build_agent_app, call_agent -- see each's own
                     docstring. render_prompt is regex-based, not
                     str.format -- confirmed against a real run, a
                     RAG-retrieved chunk's stray brace crashed the old
                     str.format_map-based version outright.
  cli.py             `qprompt-a2a render <ir.json> <workflow-name> -o <dir>`
```

## Usage

```sh
pip install -e ../qprompt-langgraph  # not published; install the sibling first
pip install -e ".[dev,services]"

# from qprompt-dsl:
#   node packages/cli/bin/cli.js generate <file>.qprompt -d <dest>

qprompt-a2a render <file>.ir.json <workflow-name> -o generated/<workflow-name>
docker compose -f generated/<workflow-name>/docker-compose.yml up --build
curl -X POST localhost:8000/run -d '{"inputs": {...}}'
```

## Status

First vertical slice, mirroring qprompt-langgraph's own "Status" note: one
service per agent for every kind (`llm`, `container`, `rules_engine`, `db`,
`http`, `selector`), one generated orchestrator, Dockerfiles + compose file
wiring it all together. `db`/`http`/`selector` reuse qprompt-langgraph's
stubs as-is (still `NotImplementedError` until a project wires them up, same
as that target). Not yet handled: per-agent resource limits, healthchecks,
non-Compose deployment targets (e.g. Kubernetes) -- Compose is this target's
fixed delivery mechanism for now, not a configurable option.
