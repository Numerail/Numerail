# Contributing to Numerail

Thank you for your interest in contributing to Numerail. This document explains how to set up the development environment, run tests, and submit changes.

## Development setup

```bash
git clone https://github.com/Numerail/Numerail.git
cd Numerail

# Install core
cd packages/numerail
pip install -e ".[dev]"

# Install extension
cd ../numerail_ext
pip install -e ".[dev]"
```

## Running tests

```bash
# Core — 153 tests
cd packages/numerail && pytest tests/ -v

# Guarantee certification — 45 tests
cd packages/numerail && pytest tests/test_guarantee.py -v

# Proof checker — 3,732 assertions
cd packages/numerail && python proof/verify_proof.py

# Extension — 207 tests
cd packages/numerail_ext && pytest tests/ -v
```

All tests must pass before submitting a pull request.

## The enforcement guarantee

The central design constraint is Theorem 1: if `enforce()` returns APPROVE or PROJECT, the output satisfies every active constraint. Any change that weakens this guarantee will not be accepted. The proof checker (`proof/verify_proof.py`) and guarantee certification suite (`tests/test_guarantee.py`) exist to catch such regressions.

## Pull request process

1. Fork the repository and create a branch from `main`.
2. Make your changes.
3. Ensure all tests pass, including the proof checker.
4. Update documentation if your change affects the public API or guarantee.
5. Submit a pull request with a clear description of the change.

## Reporting issues

Open an issue on GitHub. For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## Code style

- The core engine is a single file (`engine.py`) by design. This is deliberate — it makes the proof checker's job tractable and the enforcement path auditable.
- No additional runtime dependencies beyond numpy and scipy for the core.
- Type annotations on all public APIs.
- Docstrings on all public classes and functions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
