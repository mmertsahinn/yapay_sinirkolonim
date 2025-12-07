# Module: db_to_json.py

Veritabanını JSON'a çevir - TÜM VERİLER

## Functions

### datetime_to_str(obj)
Datetime objelerini string'e çevir

### serialize_match(match)
Match objesini dict'e çevir - None değerleri None olarak bırak

### serialize_league(league)
League objesini dict'e çevir

### serialize_team(team)
Team objesini dict'e çevir

### serialize_match_odds(odds)
MatchOdds objesini dict'e çevir

### serialize_stat(stat)
Stat objesini dict'e çevir

### serialize_prediction(pred)
Prediction objesini dict'e çevir

### serialize_result(result)
Result objesini dict'e çevir

### serialize_explanation(expl)
Explanation objesini dict'e çevir

### serialize_experiment(exp)
Experiment objesini dict'e çevir

### serialize_model_version(mv)
ModelVersion objesini dict'e çevir

### serialize_market(market)
Market objesini dict'e çevir

### export_database_to_json(output_file)
Tüm veritabanını JSON'a export et

