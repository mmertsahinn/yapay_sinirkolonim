# Module: src\models\train_offline.py

## Classes

### OfflineTrainer
#### Methods
- **__init__**(self, market_types, config, model_config)

- **prepare_data**(self, train_seasons, val_seasons, league_ids)

- **train_epoch**(self, model, train_loader, optimizer, criterion)

- **validate**(self, model, val_loader, criterion)

- **train**(self, train_seasons, val_seasons, league_ids, epochs)

