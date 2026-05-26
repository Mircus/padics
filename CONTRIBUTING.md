# Contributing to padic-ds

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/Mircus/padics.git
cd padics
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v --cov=src/padic
```

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Submitting changes

1. Fork the repository and create a branch (`git checkout -b feature/my-feature`).
2. Write tests for your changes.
3. Ensure `pytest` and `ruff check` both pass.
4. Open a pull request against `main`.

## Reporting bugs

Please open an issue with a minimal reproducible example, your Python version,
and the `padic-ds` version (`pip show padic-ds`).

## Mathematical conventions

- Elements are stored as `u_mod * p^v` where `gcd(u_mod, p) = 1` and `u_mod` is taken mod `p^prec`.
- Zero is represented by `u_mod = 0` with a sentinel valuation of `10**9`.
- All arithmetic is truncated to `ctx.prec` significant p-adic digits.
