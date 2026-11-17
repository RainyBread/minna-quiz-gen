# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
vulnerability reporting (Security tab → "Report a vulnerability") when it is
available, or contact the repository owner privately.

## In scope

- API key handling, `.env` / `.env.bak` exposure and provider/base URL matching
- Prompt/request logs and locally stored textbook or answer data
- Export path handling and batch identifiers
- Dependency vulnerabilities and CI secret scanning

## Local hardening

- `.env` files are created with mode `0600` when possible (POSIX; on Windows
  restrict access with file properties/`icacls`).
- `data/db/` and `data/logs/` are restricted to `0700` when possible.
- Run `python tools/check_repo_hygiene.py --history` before pushing.
- CI runs pytest, the hygiene/history scan, and `pip-audit`.
