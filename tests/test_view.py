from qprompt_a2a.codegen.view import build_project_view


def test_build_project_view_resolves_context_resources(cypress_ir):
    workflow = cypress_ir.workflows[0]
    view = build_project_view(cypress_ir, workflow)
    by_name = {svc.node.name: svc for svc in view.services}

    gen_code_resources = [r.name for r in by_name["genCode"].node.context_resources]
    assert gen_code_resources == ["cypressDocs", "cypressBestPractices"]

    assert by_name["validatePlan"].node.context_resources == []
