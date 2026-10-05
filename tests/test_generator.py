from qprompt_a2a.codegen.generator import render_project, write_project


def test_render_project_llm_service_includes_rag_wiring(cypress_ir, tmp_path):
    files = render_project(cypress_ir, "cypressFlow", tmp_path)

    gen_code_src = files["services/genCode.py"]
    assert "CONTEXT_RESOURCES" in gen_code_src
    assert "retrieve(" in gen_code_src
    assert "context_chunks=rag_chunks" in gen_code_src
    compile(gen_code_src, "<genCode.py>", "exec")  # raises SyntaxError if the template ever emits bad Python

    lint_src = files["services/lint.py"]
    assert "CONTEXT_RESOURCES" not in lint_src
    compile(lint_src, "<lint.py>", "exec")


def test_docker_compose_mounts_rag_volume_for_llm_services_with_context(cypress_ir, tmp_path):
    files = render_project(cypress_ir, "cypressFlow", tmp_path)
    compose = files["docker-compose.yml"]

    gencode_block = compose.split("  gencode:")[1].split("\n\n")[0]
    assert "QPROMPT_RAG_ROOT=/rag" in gencode_block
    assert ":/rag:ro" in gencode_block
    assert 'NEEDS_RAG: "true"' in gencode_block

    validateplan_block = compose.split("  validateplan:")[1].split("\n\n")[0]
    assert "QPROMPT_RAG_ROOT" not in validateplan_block
    assert ":/rag:ro" not in validateplan_block
    assert "NEEDS_RAG" not in validateplan_block


def test_write_project_creates_rag_dir_when_needed(cypress_ir, tmp_path):
    write_project(cypress_ir, "cypressFlow", tmp_path)
    assert (tmp_path / ".rag").is_dir()
