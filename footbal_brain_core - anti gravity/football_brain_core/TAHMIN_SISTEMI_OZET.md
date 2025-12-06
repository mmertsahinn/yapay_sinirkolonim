# 🎯 TAHMİN SİSTEMİ NASIL ÇALIŞIYOR?

## 📋 GENEL AKIŞ

### 1️⃣ **MODEL YÜKLEME**
```
model_prd_v1.0.pth → PyTorch Model → MultiTaskModel
```
- Eğitilmiş model dosyası yüklenir
- 6 farklı market için tahmin yapabilir:
  - Match Result (1-X-2)
  - BTTS (Both Teams To Score)
  - Over/Under 2.5
  - Goal Range
  - Correct Score
  - Double Chance

### 2️⃣ **FEATURE OLUŞTURMA**
Her maç için özellikler (features) oluşturulur:
- **Takım formu**: Son 10 maçtaki performans
- **Ev sahibi avantajı**: Ev sahibi takımın ev performansı
- **Deplasman performansı**: Deplasman takımın deplasman performansı
- **Karşılaşma geçmişi**: İki takım arasındaki geçmiş maçlar
- **Lig istatistikleri**: Lig ortalamaları, pozisyonlar
- **Zaman faktörü**: Sezon içindeki hafta, gün, saat

### 3️⃣ **TAHMIN YAPMA**
```
Features → Model → Probability Scores → Tahmin
```

**Adımlar:**
1. Maçın feature'ları hazırlanır
2. Model'e verilir (torch.no_grad() ile)
3. Model her market için olasılık skorları üretir
4. En yüksek olasılıklı sonuç seçilir

**Örnek:**
- Match Result: Home %45, Draw %25, Away %30 → **Home** tahmini
- BTTS: Yes %60, No %40 → **Yes** tahmini
- Over/Under 2.5: Over %55, Under %45 → **Over** tahmini

### 4️⃣ **LLM İLE AÇIKLAMA ÜRETME**
```
Tahmin + Maç Bilgileri → GPT/Grok → Açıklama Metni
```

**ScenarioBuilder** şunları yapar:
- Maç bilgilerini toplar (takımlar, form, geçmiş)
- GPT ve Grok'a sorar (hangi model daha iyi açıklama üretirse onu kullanır)
- Her market için açıklama metni üretir

**Örnek Açıklama:**
> "Manchester City ev sahibi avantajıyla güçlü. Son 5 maçta 4 galibiyet. 
> Arsenal deplasmanda zayıf, son 3 deplasman maçında 2 mağlubiyet. 
> City'nin ev sahibi formu ve Arsenal'in deplasman sorunları nedeniyle 
> **Home** tahmini yapıldı."

### 5️⃣ **KAYDETME**
Tahminler ve açıklamalar veritabanına kaydedilir:
- `predictions` tablosu: Tahmin sonuçları
- `explanations` tablosu: LLM açıklamaları
- `model_version_id`: Hangi model versiyonu kullanıldı

### 6️⃣ **EXCEL RAPORU**
Tüm tahminler Excel'e aktarılır:
- Maç bilgileri
- Her market için tahmin
- Olasılık skorları
- LLM açıklamaları
- Gerçek sonuçlar (maç oynandıktan sonra)

---

## 🔄 TAM WORKFLOW

```
1. Gelecek 7 gün içindeki maçları bul
   ↓
2. Her maç için:
   a) Feature'ları oluştur
   b) Model ile tahmin yap
   c) LLM ile açıklama üret
   d) Veritabanına kaydet
   ↓
3. Excel raporu oluştur
```

---

## 📊 ÖRNEK KULLANIM

```python
# 1. Model yükle
model = load_model("model_prd_v1.0.pth", market_types, input_size)

# 2. Predictor oluştur
predictor = MarketPredictor(model, market_types)

# 3. Maç için tahmin yap
predictions = predictor.predict_match(match_id, session)
# Sonuç: {
#   MarketType.MATCH_RESULT: {"outcome": "Home", "probability": 0.45},
#   MarketType.BTTS: {"outcome": "Yes", "probability": 0.60},
#   ...
# }

# 4. LLM ile açıklama üret
scenario_builder = ScenarioBuilder()
explanations = scenario_builder.generate_explanation(match, predictions, market_types)

# 5. Kaydet
predictor.save_predictions(match_id, predictions, model_version_id)
scenario_builder.save_explanations(match, explanations, {})
```

---

## 🎯 SONUÇ

**Tahmin Sistemi:**
- ✅ Model ile tahmin yapar (6 market)
- ✅ LLM ile açıklama üretir (GPT/Grok)
- ✅ Veritabanına kaydeder
- ✅ Excel raporu oluşturur

**Kullanım:**
```bash
python predict_with_explanations.py
```






