from pathlib import Path

import pytest

from qprompt_langgraph import QpromptIR, load_ir

# Shared with qprompt-langgraph's own test suite -- not duplicated, this
# repo already depends on qprompt-langgraph and the two live as siblings in
# the same monorepo checkout (see qprompt_a2a.codegen.generator._repo_root,
# which makes the same assumption for Dockerfile COPY paths).
FIXTURES = Path(__file__).parent.parent.parent / "qprompt-langgraph" / "tests" / "fixtures"


@pytest.fixture
def cypress_ir() -> QpromptIR:
    return load_ir(FIXTURES / "cypress-workflow.ir.json")
