from qprompt_a2a.runtime import render_prompt


def test_render_prompt_substitutes_state_and_channel_placeholders():
    template = "Failure rule: {state.rule} Reason: {state.reason} Plan: {test_plan}"
    result = render_prompt(template, {"rule": "SYNTAX", "reason": None, "test_plan": "Title: t"})
    assert result == "Failure rule: SYNTAX Reason: None Plan: Title: t"


def test_render_prompt_leaves_unknown_placeholder_literal():
    # cygenerator's real prompt has {output_format}, which nothing in inputs
    # ever populates -- must stay literal, not raise.
    assert render_prompt("code in {output_format}", {}) == "code in {output_format}"


def test_render_prompt_defaults_state_control_fields_to_none():
    result = render_prompt("Error: {state.error}", {})
    assert result == "Error: None"


def test_render_prompt_never_crashes_on_stray_braces():
    # Confirmed against a real run: a RAG-retrieved Cypress-docs chunk
    # containing a stray '}' crashed the old str.format_map-based
    # implementation with "Single '}' encountered in format string".
    template = "Reference material:\nfunction f() { return x; }\nunmatched: }"
    assert render_prompt(template, {}) == template


def test_render_prompt_does_not_mangle_dict_repr_in_state_details():
    # state.details can itself be a dict whose str() contains literal
    # braces -- a second render pass over that text must not treat it as
    # more placeholder syntax. render_prompt only ever runs once per
    # call_llm call (see call_llm's docstring), but this guards the
    # invariant directly: the *output* of a render must be inert if fed
    # back in.
    inputs = {"details": {"typescript": [{"code": "TS2304"}]}}
    once = render_prompt("Details: {state.details}", inputs)
    assert render_prompt(once, inputs) == once
