import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from qprompt_langgraph.codegen.env import py_literal, py_triple_quoted, screaming_snake

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def yaml_str(value: str) -> str:
    """JSON is a valid subset of YAML flow scalars, so `json.dumps` doubles
    as a YAML-safe quoted string -- avoids hand-rolling YAML string escaping
    for env var values, image names, etc. in docker-compose.yml.jinja."""
    return json.dumps(value)


def make_environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["pyrepr"] = repr
    env.filters["py_literal"] = py_literal
    env.filters["py_triple_quoted"] = py_triple_quoted
    env.filters["screaming_snake"] = screaming_snake
    env.filters["yaml_str"] = yaml_str
    return env
