describe("Test Login Page with Form Authentication", () => {
  beforeEach(() => {
    cy.visit('https://the-internet.herokuapp.com/login');
    cy.get('#username').type('valid_user').should('have.#username', 'valid_user');
    cy.get('#password').type('valid_password').should('have.#password', 'valid_password');
    cy.get('button[type="submit"]').click();
    cy.url().should('contain', 'secure');
    cy.get('button[type="submit"]').click();
    cy.url().should('contain', 'secure');
  });
});