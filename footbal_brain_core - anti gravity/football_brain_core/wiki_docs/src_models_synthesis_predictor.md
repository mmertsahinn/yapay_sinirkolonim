# Module: src\models\synthesis_predictor.py

Sentez Tahmin Sistemi
Takım profili + İkili ilişki → Sentezlenmiş tahmin
Her iki bilgiyi birleştirerek düşük hata oranı için optimize edilmiş tahmin.

## Classes

### SynthesisPredictor
Takım profili ve ikili ilişkiyi sentezleyerek tahmin yapar.
Düşük hata oranı için optimize edilmiş.

#### Methods
- **__init__**(self, team_profile_manager, pairwise_manager)

- **predict_with_synthesis**(self, match_id, market_type, base_prediction)
  - Sentezlenmiş tahmin yap:
1. Takım profillerinden bilgi al
2. İkili ilişkiden bilgi al
3. İkisini sentezle
4. Optimize edilmiş tahmin döndür

- **_extract_profile_info**(self, profile, market_type, venue)
  - Takım profilinden bilgi çıkar

- **_extract_relationship_info**(self, relationship, market_type)
  - İkili ilişkiden bilgi çıkar

- **_synthesize_predictions**(self, home_info, away_info, relationship_info, market_type)
  - İki bilgiyi sentezle:
- Takım profilleri: Genel davranış
- İkili ilişki: Özel dinamikler

- **_combine_with_base**(self, synthesis, base_prediction)
  - Base prediction ile birleştir (ensemble)

- **_calculate_confidence**(self, home_info, away_info, relationship_info)
  - Güven seviyesi hesapla

- **_generate_reasoning**(self, home_info, away_info, relationship_info)
  - Açıklama oluştur

