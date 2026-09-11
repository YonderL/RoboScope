# Contributing

Start with a bounded issue: an observed failure, reproducible bug, or one new measured recipe. Include config, dependency versions, checkpoint identity and task/init IDs where relevant.

```bash
python -m pip install -e '.[report,dev]'
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest tests/test_results.py tests/test_cli.py
```

Model tests require the training environment (`tests/test_models.py`). Simulator smoke tests are opt-in and bounded. Never launch a long training run as part of unit tests or CI.

Use absolute package imports, preserve episode boundaries and action units, and document changes to checkpoint or evaluation semantics. Add a regression for observable behavior, not an assertion that repeats the implementation. Keep large data, weights, private paths and credentials out of commits. Reports must identify training seeds, episode denominators, checkpoint selection and limits of comparison.
