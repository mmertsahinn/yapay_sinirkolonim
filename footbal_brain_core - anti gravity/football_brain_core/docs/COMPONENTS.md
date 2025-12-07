# Component Reference

## `src/ingestion`
Handles all external data fetching.
- **`api_client.py`**: Wrapper for API-Football. Adds headers, handles response codes.
- **`normalizer.py`**: Cleans team names (e.g., "Man Utd" -> "Manchester United") to match database records.

## `src/db`
Database abstraction layer.
- **`connection.py`**: Manages SQLite connection and session scope.
- **`schema.py`**: SQLAlchemy ORM definitions for `Match`, `Team`, `League`, `Odds`.
- **`repositories.py`**: CRUD operations (e.g., `MatchRepository.get_by_date(...)`).

## `src/features`
- **`engineer.py`**: Main entry point for transforming raw DB objects into `X` (features) and `y` (targets).
- **`market_targets.py`**: Defines what we are predicting (enums for `HOME_WIN`, `DRAW`, `BTTS_YES`).

## `src/models`
- **`multi_task_model.py`**: PyTorch `nn.Module` definition. Input size -> Shared Layers -> [Head1, Head2, ...].
- **`team_profile.py`**: `TeamProfile` class logic. Calculates form, streaks, and stats.
- **`train_offline.py`**: Script to train on historical data without APIs.
- **`train_incremental.py`**: Online learning script for continuous updates.

## `src/inference`
- **`predictor.py`**: Loads a saved model `.pth` and a `TeamProfileManager` to generate probabilities for upcoming fixtures.

## `src/explanations`
- **`narrative_generator.py`**: Uses templates (and optionally LLMs) to convert probabilities into human-readable text (e.g., "Arsenal is likely to win because...").

## `src/cli`
- **`main.py`**: The CLI entry point. Uses `argparse` to route commands like `train`, `predict`, `init-db`.
