"""Support library imported by *generated* agent-service and orchestrator
modules (not by codegen itself) -- installed inside every generated
container via the `qprompt-a2a[services]` extra (see pyproject.toml).

`evaluate_rules`, `resolve_state_assignments`, `run_container` (with its
adapter registry), `run_query`, `call_http`, and `select_candidates` are
reused directly from `qprompt_langgraph.runtime` -- none of that logic is
LangGraph-specific, it's the same per-agent-kind behavior regardless of
whether the caller is a StateGraph node function or an A2A executor.

What's new here is specific to this target:
- `call_llm`: unlike qprompt_langgraph's stub (which deliberately doesn't
  know how to reach a model), this target commits to a concrete model
  runtime -- Docker Compose's `models:` key / Docker Model Runner -- so a
  real implementation belongs here rather than staying a stub. Reads the
  `<ENV_PREFIX>_URL` / `<ENV_PREFIX>_MODEL` variables Compose injects into
  any service listing a model (see docs.docker.com/ai/compose/models-and-
  compose) and calls its OpenAI-compatible endpoint.
- `build_agent_app` / `call_agent`: the A2A server/client boilerplate every
  generated agent service and the orchestrator would otherwise repeat
  verbatim (AgentCard, DefaultRequestHandler, FastAPI route wiring on the
  server side; card resolution, client creation, data-message round-trip on
  the client side).
"""

from __future__ import annotations

from typing import Any

import a2a.helpers as h
import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, SendMessageRequest
from fastapi import FastAPI


class _SafeFormatDict(dict):
    """Used with str.format_map so an unresolved top-level placeholder
    (e.g. a prompt's leftover {output_format} that nothing in `inputs` ever
    populates) renders back as literal text instead of raising -- prompts
    are free-form STRING literals in the DSL, not a validated template
    language, so a strict .format() would crash on any prompt authored
    before this substitution convention existed."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_prompt(template: str, inputs: dict[str, Any]) -> str:
    """Fills `{channel_name}` and `{state.field}` placeholders in a prompt
    from the current inputs/state dict. `state.<field>` access is supported
    by exposing the same dict as an attribute-accessible namespace; a
    missing field there still raises inside str.format's own machinery
    (unlike a missing top-level key), so agent_state/validation_state's
    conventional fields (rule, reason, details, error) are defaulted to
    None -- the fields cydebugger's prompt is documented to read."""

    class _State:
        def __init__(self, data: dict[str, Any]):
            for key in ("rule", "reason", "details", "error"):
                setattr(self, key, data.get(key))
            for key, value in data.items():
                setattr(self, key, value)

    context = _SafeFormatDict(inputs)
    context["state"] = _State(inputs)
    try:
        return template.format_map(context)
    except (KeyError, IndexError):
        return template


def call_llm(*, env_prefix: str, model: str, prompt: str, inputs: dict[str, Any]) -> str:
    """Real `call_llm`, backed by whatever model Compose injected for this
    service. `env_prefix` is the *Compose model name*'s env var prefix
    (e.g. `QWEN3_FINETUNED`), not the agent's name -- one model can back
    several agents, so it's looked up from `ir.models`, not derived from
    the calling agent."""
    import os

    from openai import OpenAI

    base_url = os.environ[f"{env_prefix}_URL"]
    model_name = os.environ.get(f"{env_prefix}_MODEL", model)

    client = OpenAI(base_url=base_url, api_key="unused")
    rendered_prompt = render_prompt(prompt, inputs)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": rendered_prompt},
            {"role": "user", "content": "\n".join(f"{k}: {v}" for k, v in inputs.items())},
        ],
    )
    return response.choices[0].message.content or ""


def build_agent_app(*, name: str, description: str, tags: list[str], executor: AgentExecutor, port: int) -> FastAPI:
    """Wires `executor` into a runnable FastAPI app exposing the A2A
    endpoints (agent card + JSON-RPC) -- the same shape for every generated
    agent service regardless of the wrapped agent's `kind`.

    The Agent Card's own `url` has to be where *other* services can reach
    this one, not where this one binds -- confirmed against a real running
    service: hardcoding `0.0.0.0` here makes card resolution work (a plain
    GET against whatever base_url the caller already has) but every actual
    JSON-RPC call fail, since the A2A client connects to the URL embedded
    in the resolved card, not the base_url used to fetch it. `SELF_URL` is
    set per service in docker-compose.yml to that service's own compose DNS
    name (see docker-compose.yml.jinja); the 0.0.0.0 fallback only works
    for resolving this service's own card locally, not for being called by
    another one.
    """
    import os

    self_url = os.environ.get("SELF_URL", f"http://0.0.0.0:{port}/")
    agent_card = AgentCard(
        name=name,
        description=description,
        version="0.1.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[AgentSkill(id=name, name=name, description=description, tags=tags)],
        supported_interfaces=[AgentInterface(url=self_url, protocol_binding="JSONRPC")],
    )
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=agent_card)
    app = FastAPI()
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
    )
    return app


async def call_agent(base_url: str, state: dict[str, Any]) -> dict[str, Any]:
    """Client side of the same convention: send the whole current state as
    one A2A data message, get the callee's updated view of it back as one
    data message. Used by the generated orchestrator to call each agent in
    turn -- never by agents calling each other directly, since routing is
    centralized in the orchestrator (see qprompt_a2a's README for why)."""
    async with httpx.AsyncClient(timeout=120.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client, base_url=base_url)
        card = await resolver.get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=httpx_client)).create(card)
        request = SendMessageRequest(message=h.new_data_message(state))
        async for event in client.send_message(request):
            parts = h.get_data_parts(event.message.parts)
            return parts[0] if parts else {}
    return {}
