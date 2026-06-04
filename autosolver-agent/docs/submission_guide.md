# Submission Guide

The contest upload artifact is:

```text
solver.py
```

Do not upload the full repository unless the platform explicitly accepts it.

## Required Interface

```python
def solve(input_text: str) -> list:
    ...
```

The returned value must be:

```python
[(task_id_list_str, [courier_id, ...]), ...]
```

## Safety Checklist

Before upload:

```powershell
python -m py_compile solver.py
python tests\test_solver_synthetic.py
```

Confirm that:

- `solver.py` contains no network calls,
- no platform credentials are embedded,
- the file is below the upload limit,
- `solve()` imports without side effects,
- hidden-data logs and result JSONs are not bundled.

## If You Need a Submit Script

Keep submit scripts outside the repository or parameterize them with
environment variables:

```powershell
$env:AUTOSOLVER_TEAM = "your-team"
$env:AUTOSOLVER_EMAIL = "your-email"
$env:AUTOSOLVER_TOKEN = "token-from-login"
```

Never commit raw credentials.
