"""Turns a (QpromptIR, Workflow) pair into the view this codegen target
needs, on top of qprompt_langgraph's existing WorkflowView. Reused as-is,
not re-implemented: state-field merging and route-condition translation
don't change just because each node is now its own A2A service instead of
a StateGraph node function -- see qprompt_langgraph.codegen.view for that
logic. What's new here is purely about the *deployment* shape: which port
each service listens on, and which models (from `ir.models`) any workflow's
`llm`-kind agents actually reference, so the compose file only declares the
ones in use.
"""

from __future__ import annotations

from dataclasses import dataclass

from qprompt_langgraph.codegen.view import NodeView, WorkflowView, build_workflow_view
from qprompt_langgraph.ir.models import Model, QpromptIR, Workflow

# Services listen on this port *inside* their own container -- compose's
# internal DNS (service name -> container) makes host-side port allocation
# irrelevant; only the orchestrator is published to the host.
AGENT_SERVICE_PORT = 8000
ORCHESTRATOR_PORT = 8000
ORCHESTRATOR_HOST_PORT = 8000


@dataclass
class ServiceView:
    node: NodeView
    env_prefix: str  # e.g. "SYNTAXVALIDATOR" -- used for this service's own env var names
    # Compose service names double as container DNS names and (absent an
    # explicit `image:`) get used to derive an image tag, and Docker
    # requires both to be lowercase -- step names in the DSL are free-form
    # (e.g. "validatePlan", "debugError"), so this is never assumed equal
    # to node.name. Only used for Docker-facing identifiers (the compose
    # service key, depends_on, URL hostnames); Python-side code keeps using
    # node.name (e.g. as the AGENT_URLS dict key) since case doesn't matter
    # there.
    compose_name: str


@dataclass
class ProjectView:
    workflow: WorkflowView
    services: list[ServiceView]
    models: list[Model]
    # Deduped, project-wide -- every distinct secret *name* any node
    # references (see NodeView.secret_env_names), for the compose file's
    # one top-level secrets: block. Each service's own secrets: list (which
    # of these it actually needs) comes straight from its own
    # node.secret_env_names in the template -- this field is only for the
    # top-level declarations, which must be unique even if several
    # services reference the same secret.
    secret_names: list[str]


def _env_prefix(name: str) -> str:
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def build_project_view(ir: QpromptIR, workflow: Workflow) -> ProjectView:
    workflow_view = build_workflow_view(ir, workflow)
    services = [
        ServiceView(node=node, env_prefix=_env_prefix(node.name), compose_name=node.name.lower())
        for node in workflow_view.nodes
    ]

    used_model_names = {
        node.agent.model for node in workflow_view.nodes if node.agent.kind == "llm" and node.agent.model
    }
    models = [m for m in ir.models if m.name in used_model_names]

    secret_names = sorted({name for node in workflow_view.nodes for name in node.secret_env_names})

    return ProjectView(workflow=workflow_view, services=services, models=models, secret_names=secret_names)
