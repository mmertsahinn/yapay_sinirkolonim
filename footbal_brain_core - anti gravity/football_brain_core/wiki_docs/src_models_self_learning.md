# Module: src\models\self_learning.py

Beynin Kendini Test Etme ve Öğrenme Mekanizması
- Eski maçları bugün yapılıyormuş gibi tahmin eder
- Hataları analiz eder ve öğrenir
- Tüm takımlar, ikili ilişkiler, pattern'ler hakkında öğrenir
- En başarılı olana kadar sürekli deneme-yanılma ile öğrenir

## Classes

### SelfLearningBrain
Beynin kendini test etme ve öğrenme mekanizması.
Eski maçları bugün yapılıyormuş gibi tahmin eder, hataları öğrenir,
tüm takımlar ve ilişkiler hakkında bilgi toplar.

#### Methods
- **__init__**(self, model, market_types, config)

- **learn_from_past_matches**(self, season, league_ids, max_iterations, target_accuracy)
  - Geçmiş maçları bugün yapılıyormuş gibi tahmin eder ve öğrenir.

Args:
    season: Hangi sezon üzerinde öğrenilecek
    league_ids: Hangi ligler (None ise tüm ligler)
    max_iterations: Maksimum öğrenme iterasyonu
    target_accuracy: Hedef doğruluk oranı

Returns:
    Öğrenme süreci metrikleri

- **analyze_team_relationships**(self, season, league_ids)
  - Takımlar arası ikili ilişkileri analiz eder ve öğrenir.
Hangi takımlar hangi takımlara karşı nasıl performans gösteriyor.

- **continuous_learning_loop**(self, seasons, league_ids, max_iterations_per_season)
  - Sürekli öğrenme döngüsü: Tüm sezonlar üzerinde öğrenir,
en başarılı olana kadar deneme-yanılma yapar.

