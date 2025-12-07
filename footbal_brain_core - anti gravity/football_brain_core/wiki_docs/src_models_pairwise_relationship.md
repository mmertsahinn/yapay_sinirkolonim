# Module: src\models\pairwise_relationship.py

Takım Çiftleri İlişki Modeli
Her takım çifti için ayrı algoritma/düşünce:
- H2H (head-to-head) analizi
- Ev sahibi avantajı
- Deplasman gücü
- İkili dinamikler

## Classes

### PairwiseRelationship
İki takım arasındaki özel ilişki modeli.
Her takım çifti için ayrı algoritma.

#### Methods
- **__init__**(self, team_a_id, team_b_id)

- **build_relationship_model**(self, matches, market_types)
  - İki takım arasındaki ilişki modelini oluştur.
Ayrı bir algoritma/düşünce sistemi.

- **_analyze_h2h_patterns**(self, matches, session)
  - Head-to-head pattern'leri

- **_analyze_home_advantage**(self, matches, session)
  - Ev sahibi avantajı analizi

- **_analyze_away_strength**(self, matches, session)
  - Deplasman gücü analizi

- **_analyze_market_relationship**(self, matches, market_type, session)
  - Bir market için ilişki analizi

- **_determine_dominance**(self)
  - Hangi takım dominant?

- **_classify_relationship**(self)
  - İlişki tipini sınıflandır

- **_create_prediction_algorithm**(self)
  - Bu takım çifti için özel tahmin algoritması oluştur.

### PairwiseRelationshipManager
Tüm takım çiftleri ilişkilerini yönetir.

#### Methods
- **__init__**(self)

- **get_or_create_relationship**(self, team_a_id, team_b_id)
  - İlişki al veya oluştur

- **build_all_relationships**(self, season, market_types)
  - Tüm takım çiftleri ilişkilerini oluştur

