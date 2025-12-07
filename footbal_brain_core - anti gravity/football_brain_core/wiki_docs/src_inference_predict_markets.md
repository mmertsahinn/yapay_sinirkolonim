# Module: src\inference\predict_markets.py

## Classes

### MarketPredictor
#### Methods
- **__init__**(self, model, market_types, feature_builder)

- **predict_match**(self, match_id, session)

- **predict_upcoming_matches**(self, date_from, date_to, league_ids)

- **save_predictions**(self, match_id, predictions, model_version_id)

## Functions

### load_model_and_predict(model_path, market_types, input_size, model_config)
