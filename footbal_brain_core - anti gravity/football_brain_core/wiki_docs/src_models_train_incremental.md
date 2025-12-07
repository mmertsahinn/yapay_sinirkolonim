# Module: src\models\train_incremental.py

## Classes

### IncrementalTrainer
#### Methods
- **__init__**(self, market_types, config)

- **get_new_matches_since_last_training**(self, last_training_date, league_ids)

- **retrain**(self, base_model, new_matches, train_seasons, val_seasons, league_ids, epochs)

- **should_update_model**(self, new_model, current_model, val_loader)

