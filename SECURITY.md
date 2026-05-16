# Security Policy

## Supported Versions

Only the latest `main` branch is currently supported.

## Reporting a Vulnerability

Please report security issues privately rather than opening a public issue.

Include:

- a concise description
- affected component
- reproduction steps
- impact assessment
- any suggested mitigation

Until a dedicated security contact is published for the project, use the repository owner's private contact channel for disclosure.

## Scope Notes

- The chatbot only permits read-only SQL over an allowlisted schema.
- The dashboard operations page can trigger local data and model jobs; protect deployed instances with appropriate network and host access controls.
- Secrets belong in `.env`, never in committed files.
