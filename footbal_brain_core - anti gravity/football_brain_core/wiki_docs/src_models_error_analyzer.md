# Module: src\models\error_analyzer.py

Hata Analizi ve Öğrenme Mekanizması
- Her hatayı kategorize eder (bias, variance, feature eksikliği, vb.)
- Sapma payı, bias, feature importance öğrenir
- Formülüne/feature'larına eklemeler yapar
- Kendini evrimleştirir

## Classes

### ErrorAnalyzer
Hataları analiz eder, kategorize eder ve öğrenme için bilgi çıkarır.

#### Methods
- **__init__**(self, feature_builder)

- **analyze_error**(self, match_id, market_type, predicted, actual, predicted_proba, session)
  - Bir hatayı detaylı analiz eder ve kategorize eder.

Returns:
    Hata analizi sonuçları (hata türü, nedeni, çözüm önerisi)

- **_detect_team_bias**(self, home_team_id, away_team_id, market_type, session)
  - Takım bazlı bias tespit eder

- **_analyze_feature_contribution**(self, features, predicted, actual, market_type)
  - Feature'ların hataya katkısını analiz eder

- **_detect_missing_features**(self, match, features, predicted, actual, session)
  - Eksik feature'ları tespit eder

- **_analyze_pattern_error**(self, match, predicted, actual, market_type, session)
  - Pattern hatası analizi

- **_calculate_deviation**(self, predicted_proba, predicted, actual)
  - Sapma payı hesaplar

- **_get_h2h_matches**(self, home_id, away_id, before_date, session)
  - Head-to-head maçları getirir

- **_calculate_form_trend**(self, recent_matches)
  - Form trendini hesaplar (pozitif = yükseliş, negatif = düşüş)

- **collect_errors**(self, errors)
  - Toplu hata analizi ve öğrenme

