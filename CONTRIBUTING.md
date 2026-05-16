# Contributing

Thanks for contributing to NBA OSS Game Predictor.

## Development Setup

```bash
cp .env.example .env
make install
docker compose up -d postgres
make test
```

## Contribution Guidelines

- Keep the project self-hosted and open-source only.
- Preserve leakage-safe feature construction. Features for a game must not read the same game or future games.
- Keep TimesFM as a numeric forecaster, not as the win/loss classifier.
- Prefer focused changes with tests for behavior changes.
- Do not commit generated database dumps, model artifacts, or processed outputs.
- Respect NBA.com terms when working with data. This repository distributes code, not NBA data snapshots.

## Pull Requests

1. Create a focused branch.
2. Add or update tests.
3. Run:

   ```bash
   make test
   ```

4. Update documentation when behavior or setup changes.
5. Describe any migration, data-refresh, or retraining steps in the PR.

## Reporting Issues

Please include:

- expected behavior
- actual behavior
- reproduction steps
- logs or stack traces
- operating system and Python / Docker versions
- whether the issue concerns API, dashboard, ingestion, forecasting, or training
