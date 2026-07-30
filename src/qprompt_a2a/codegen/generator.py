"""render_project()/write_project(): the qprompt-a2a analogue of
qprompt_langgraph.codegen.generator, except the unit of output is a whole
directory (one file per agent service, an orchestrator, two Dockerfiles,
and docker-compose.yml) rather than a single Python module.
"""

from __future__ import annotations

import os
from pathlib import Path

import qprompt_a2a
import qprompt_langgraph
from qprompt_langgraph.ir.models import QpromptIR

from qprompt_a2a.codegen.env import make_environment
from qprompt_a2a.codegen.view import (
    AGENT_SERVICE_PORT,
    ORCHESTRATOR_HOST_PORT,
    ORCHESTRATOR_PORT,
    ProjectView,
    _env_prefix,
    build_project_view,
)

_ENV = make_environment()


def _repo_root(package_module) -> Path:
    """Given an installed-editable package's module object, walks up from
    its `src/<pkg>/` location to that package's own repo root (e.g.
    .../qprompt-langgraph/src/qprompt_langgraph/__init__.py -> .../qprompt-langgraph)."""
    return Path(package_module.__file__).resolve().parents[2]


def render_project(ir: QpromptIR, workflow_name: str, out_dir: Path) -> dict[str, str]:
    """Returns {relative_path: file_content} for the whole generated project.
    `out_dir` is needed even just to *render* (not only to write) since the
    Dockerfiles' COPY paths and docker-compose.yml's build.context are
    computed relative to where the project will actually live on disk."""
    workflow = next(w for w in ir.workflows if w.name == workflow_name)
    view: ProjectView = build_project_view(ir, workflow)

    langgraph_root = _repo_root(qprompt_langgraph)
    a2a_root = _repo_root(qprompt_a2a)
    monorepo_root = os.path.commonpath([langgraph_root, a2a_root, out_dir.resolve()])
    build_context = os.path.relpath(monorepo_root, start=out_dir.resolve())
    out_rel = os.path.relpath(out_dir.resolve(), start=monorepo_root)
    qprompt_langgraph_rel = os.path.relpath(langgraph_root, start=monorepo_root)
    qprompt_a2a_rel = os.path.relpath(a2a_root, start=monorepo_root)

    # A real host-visible directory per container-kind service, bind-mounted
    # into it -- see workspace_dir()'s docstring in qprompt_langgraph.runtime
    # for why this has to be a fixed host path, not the generic tempfile-based
    # scratch space run_docker's other callers use.
    host_workspace_dirs = {
        svc.compose_name: str(out_dir.resolve() / ".workspace" / svc.compose_name)
        for svc in view.services
        if svc.node.agent.kind == "container"
    }

    files: dict[str, str] = {}

    agent_tpl = _ENV.get_template("agent_service.py.jinja")
    for svc in view.services:
        model_env_prefix = _env_prefix(svc.node.agent.model) if svc.node.agent.model else ""
        files[f"services/{svc.node.name}.py"] = agent_tpl.render(
            ir=ir,
            node=svc.node,
            port=AGENT_SERVICE_PORT,
            model_env_prefix=model_env_prefix,
            workflow_name=workflow_name,
        )

    orch_tpl = _ENV.get_template("orchestrator.py.jinja")
    files["orchestrator.py"] = orch_tpl.render(
        ir=ir,
        view=view.workflow,
        services=view.services,
        max_loops=workflow.max_loops or 10,
        port=ORCHESTRATOR_PORT,
    )

    files["Dockerfile.agent"] = _ENV.get_template("Dockerfile.agent.jinja").render(
        qprompt_langgraph_rel=qprompt_langgraph_rel,
        qprompt_a2a_rel=qprompt_a2a_rel,
        services_rel=f"{out_rel}/services",
    )
    files["Dockerfile.orchestrator"] = _ENV.get_template("Dockerfile.orchestrator.jinja").render(
        qprompt_langgraph_rel=qprompt_langgraph_rel,
        qprompt_a2a_rel=qprompt_a2a_rel,
        orchestrator_rel=f"{out_rel}/orchestrator.py",
    )
    files["docker-compose.yml"] = _ENV.get_template("docker-compose.yml.jinja").render(
        ir=ir,
        view=view.workflow,
        services=view.services,
        models=view.models,
        orchestrator_host_port=ORCHESTRATOR_HOST_PORT,
        build_context=build_context,
        out_rel=out_rel,
        host_workspace_dirs=host_workspace_dirs,
    )

    container_services = [svc for svc in view.services if svc.node.agent.kind == "container"]
    if container_services:
        files["docker-compose.override.yml.example"] = _ENV.get_template(
            "docker-compose.override.yml.example.jinja"
        ).render(container_services=container_services)

    return files


def write_project(ir: QpromptIR, workflow_name: str, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    (out_dir / "services").mkdir(parents=True, exist_ok=True)

    written = []
    for rel_path, content in render_project(ir, workflow_name, out_dir).items():
        path = out_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(path)

    workflow = next(w for w in ir.workflows if w.name == workflow_name)
    view = build_project_view(ir, workflow)
    for svc in view.services:
        if svc.node.agent.kind == "container":
            (out_dir / ".workspace" / svc.compose_name).mkdir(parents=True, exist_ok=True)

    return written
