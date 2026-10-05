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

import re
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


_PLACEHOLDER_RE = re.compile(r"\{(state\.)?([A-Za-z_][A-Za-z0-9_]*)\}")

# Fields cydebugger's prompt is documented to read via {state.<field>} --
# resolved even if this exact call never set them, same as render_prompt's
# old _State object defaulted them to None rather than raising.
_STATE_DEFAULTS = ("rule", "reason", "details", "error")


def render_prompt(template: str, inputs: dict[str, Any]) -> str:
    """Fills `{channel_name}` and `{state.field}` placeholders in a prompt
    from the current inputs/state dict.

    Regex-based, not str.format -- a prompt is a free-form STRING literal in
    the DSL, not a validated template language, and str.format_map would
    treat *any* other `{`/`}` in the template as format syntax too, which
    real content downstream can easily contain: a state.details value that's
    a dict's own repr (literal braces), or RAG-retrieved reference material
    full of real code snippets (see augment_prompt) -- confirmed against a
    real run: a retrieved Cypress-docs chunk containing a stray `}` crashed
    str.format_map with "Single '}' encountered in format string", which
    the old implementation's `except (KeyError, IndexError)` didn't even
    catch. A placeholder naming a key `inputs`/state doesn't have (e.g. a
    prompt's leftover {output_format} nothing ever populates) renders back
    as literal text, same as before; anything that isn't `{name}` or
    `{state.name}` shaped is never touched at all, so this can't crash or
    partially-substitute regardless of what the rest of the template or any
    appended content contains."""

    def _replace(match: re.Match[str]) -> str:
        is_state, name = match.group(1), match.group(2)
        if is_state and (name in _STATE_DEFAULTS or name in inputs):
            return str(inputs.get(name))
        if not is_state and name in inputs:
            return str(inputs[name])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)


def call_llm(
    *,
    env_prefix: str,
    model: str,
    prompt: str,
    inputs: dict[str, Any],
    subscribed: list[str],
    context_chunks: list[str] = (),
) -> str:
    """Real `call_llm`, backed by whatever model Compose injected for this
    service. `env_prefix` is the *Compose model name*'s env var prefix
    (e.g. `QWEN3_FINETUNED`), not the agent's name -- one model can back
    several agents, so it's looked up from `ir.models`, not derived from
    the calling agent.

    `inputs` is the orchestrator's *entire* accumulated pipeline state
    (every channel and control field set by every step so far, not just
    this agent's own) -- `render_prompt` needs that full picture, since a
    prompt's `{state.rule}`-style placeholders can reference a field a
    different agent wrote. The user turn is not the same thing: confirmed
    against a real run, dumping all of `inputs` into it (rather than just
    `subscribed`, the channels this agent's own `subscribe:` declares) hits
    a real model's context limit a few steps into the workflow -- lint's
    emit_state alone puts a JSON report under `bug_details` *and* a subset
    of the same data under `details`, and every subsequent LLM call was
    re-sending both, verbatim, in every turn.

    `context_chunks` (RAG-retrieved text, see qprompt_langgraph.rag) is
    appended via `augment_prompt` *after* `render_prompt`, not folded into
    `prompt` by the caller first -- render_prompt must only ever run once,
    on the author-written static prompt, never on text that already
    contains retrieved content or a rendered `{state.details}` dict repr;
    either could coincidentally contain something shaped like a placeholder
    and get mangled by a second pass.
    """
    import os

    from openai import OpenAI
    from qprompt_langgraph.runtime import apply_output_filters, augment_prompt

    base_url = os.environ[f"{env_prefix}_URL"]
    model_name = os.environ.get(f"{env_prefix}_MODEL", model)

    client = OpenAI(base_url=base_url, api_key="unused")
    rendered_prompt = augment_prompt(render_prompt(prompt, inputs), list(context_chunks))
    user_content = "\n".join(f"{name}: {inputs.get(name)}" for name in subscribed)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": rendered_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    content = response.choices[0].message.content or ""
    # Fence-stripping is the only universal step: wrapping an answer in
    # ```lang is a habit of instruction-tuned chat models generally, not
    # of any one domain. Everything past that -- where a domain's real
    # output starts and stops -- is an opt-in filter, see
    # qprompt_langgraph.runtime.register_output_filter.
    return apply_output_filters(_strip_markdown_fence(content))


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n(.*?)\n?```$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    """Strips a wrapping ```lang ... ``` fence, if present. A prompt telling
    a code-generating agent not to do this is not reliable enough on its own
    to depend on -- confirmed against a real 0.6B-parameter model: the
    instruction was explicit in cygenerator's prompt and the model still
    fenced its output. Wrapping code in a fence is such a strong habit from
    an instruction-tuned model's training data that stripping it
    programmatically is the actually-robust fix, not a bigger model or a
    more insistent prompt."""
    match = _CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text


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
    centralized in the orchestrator (see qprompt_a2a's README for why).

    timeout=600 -- confirmed against two real runs: 120s was too tight for
    a container-kind agent's real Cypress run (retries against a real page
    can legitimately take a while) or a just-restarted LLM service's
    cold-start inference, and 300s was in turn too tight for an llm-kind
    agent reasoning over a real, multi-item tool result (e.g. filterJobs
    over several real scraped job postings) on CPU-only inference with no
    GPU -- raised A2AClientTimeoutError both times."""
    async with httpx.AsyncClient(timeout=600.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client, base_url=base_url)
        card = await resolver.get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=httpx_client)).create(card)
        request = SendMessageRequest(message=h.new_data_message(state))
        async for event in client.send_message(request):
            parts = h.get_data_parts(event.message.parts)
            return parts[0] if parts else {}
    return {}
