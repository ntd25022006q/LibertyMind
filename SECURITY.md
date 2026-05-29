# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

This project is in early development (pre-alpha). Only the latest version receives security updates.

## Reporting a Vulnerability

If you discover a security vulnerability in LibertyMind, please report it responsibly.

**Do not file a public GitHub issue for security vulnerabilities.**

Instead, please:

1. **Email the maintainer** at the contact listed in the GitHub repository, or
2. **Use GitHub's private vulnerability reporting** feature: go to the [Security tab](https://github.com/ntd25022006q/LibertyMind/security) and click "Report a vulnerability."

Please include the following information in your report:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- The version or commit hash where you observed the issue
- Any suggested mitigations or fixes (optional)

## Response Timeline

- **Acknowledgment**: We aim to acknowledge receipt within 48 hours.
- **Initial assessment**: We aim to provide an initial assessment within 5 business days.
- **Resolution**: We aim to resolve confirmed vulnerabilities within 30 days. Critical issues will be prioritized.

## Security Considerations

### API Keys

LibertyMind interacts with third-party AI providers (OpenAI, Anthropic, Google, etc.) that require API keys. These keys must be stored in environment variables -- **never hardcode API keys in source code.**

- All provider credentials are read from environment variables (see `.env.example`).
- The `.gitignore` file excludes `.env` files from version control.
- If you accidentally commit an API key, rotate it immediately through the provider's dashboard.

### Neural Module Outputs

The PyTorch neural modules in this project are **untrained scaffolding**. Their outputs are random and should not be used for:

- Content moderation or safety decisions
- Reward signal computation in production systems
- Any automated decision-making

### Proxy Server

The FastAPI proxy server forwards requests to upstream AI providers. Be aware of:

- **Upstream API key exposure**: The proxy requires an upstream API key. Ensure the server is not exposed to untrusted networks.
- **CORS configuration**: The default configuration allows all origins (`allow_origins=["*"]`). Restrict this in production.
- **No authentication**: The proxy does not require authentication by default. Add authentication middleware before deploying to production.

## Scope

This security policy covers:

- Hardcoded credentials or secrets in source code
- Injection vulnerabilities in the proxy server
- Unsafe deserialization or code execution
- Path traversal or file access issues

This security policy does **not** cover:

- Outputs of untrained neural modules (these are explicitly documented as non-functional)
- Third-party provider outages or API changes
- General AI safety concerns beyond the scope of this codebase
