"""strip_trailing_prose has to be safe by construction: it deletes text, so
every case where it can't positively identify the end of the describe() block
must leave the input untouched.

Lives in qprompt_langgraph.adapters.cypress, not the shared runtime: the
describe()-block shape it keys off is Cypress/Jest-specific, and a graph
generating anything else must not inherit it (see register_output_filter).
"""

from qprompt_langgraph.adapters.cypress import strip_trailing_prose as strip

CLEAN = '''/// <reference types="cypress" />
describe('Login', () => {
  it('logs in', () => {
    cy.visit('/login');
    cy.get('#user').type('bob');
  });
});'''


def test_drops_fence_and_explanation_after_the_block():
    # The real cydebugger output that burned all four retries: valid code,
    # then a stray closing fence and an "### Explanation:" section. Every
    # linter error landed on the trailing lines, none on the code.
    text = CLEAN + '''
```

### Explanation:
- The test correctly uses `cy.visit()` to navigate to the login form page.
- The final URL assertion checks the expected URL.

This version follows all official Cypress best practices.'''
    assert strip(text) == CLEAN


def test_clean_code_is_untouched():
    assert strip(CLEAN) == CLEAN


def test_keeps_the_reference_line_before_describe():
    assert strip(CLEAN + "\n\ntrailing prose").startswith('/// <reference types="cypress" />')


def test_braces_inside_strings_do_not_close_the_block():
    code = '''describe('x', () => {
  it('y', () => {
    cy.get('#a').should('have.text', '} not a real brace {');
    cy.get("#b").type("also } fake");
  });
});'''
    assert strip(code + "\n\n### Notes:\nblah") == code


def test_braces_inside_template_literal_do_not_close_the_block():
    code = '''describe('x', () => {
  it('y', () => {
    cy.get(`#item-${1 + 1}`).click();
  });
});'''
    assert strip(code + "\n\nExplanation: done") == code


def test_braces_inside_comments_do_not_close_the_block():
    code = '''describe('x', () => {
  it('y', () => {
    // a stray } in a line comment
    /* and a { in a block comment */
    cy.visit('/a');
  });
});'''
    assert strip(code + "\n\n**Important Notes:**\n- nope") == code


def test_no_describe_is_left_alone():
    # Not every llm agent emits Cypress -- testplanner's output is prose by
    # design and must survive this untouched.
    plan = "Title: Log in\nObjective: Verify login\nSteps:\n  1. Visit /login"
    assert strip(plan) == plan


def test_unbalanced_braces_are_left_alone():
    truncated = "describe('x', () => {\n  it('y', () => {\n    cy.visit('/a');"
    assert strip(truncated) == truncated
