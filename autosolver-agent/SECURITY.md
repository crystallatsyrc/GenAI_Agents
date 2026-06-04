# Security Policy

This repository intentionally excludes platform credentials, user email
addresses, private submission tokens, and hidden benchmark data.

If you adapt the submission workflow:

- keep tokens in environment variables,
- do not commit platform result files containing private team metadata,
- do not publish hidden or organizer-provided datasets unless explicitly allowed,
- review `solver.py` before uploading to ensure the final file performs no network calls.
