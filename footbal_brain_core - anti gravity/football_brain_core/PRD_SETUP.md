# 🧠 PRD'ye Uygun Tam Kurulum

## Hedef: PRD'de belirtilen tam özellikler

- ✅ Son 3-5 sezon veri
- ✅ Çoklu market tahminleri (MS, BTTS, Alt/Üst, Skor, vb.)
- ✅ LLM ile zekice senaryolar ve yorumlar
- ✅ Karşılaştırmalı analiz
- ✅ Excel öğrenme defteri çıktıları

---

## ADIM 1: Paketleri Yükle

```powershell
cd football_brain_core
pip install -r requirements.txt
```

**Gerekli paketler:**
- sqlalchemy (veritabanı)
- requests (API)
- torch (ML model)
- numpy, pandas (veri işleme)
- openpyxl (Excel)
- scikit-learn (metrikler)

---

## ADIM 2: API Key'leri Ayarla

```powershell
# API-FOOTBALL (Zorunlu)
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"

# OpenRouter (LLM açıklamaları için - PRD'de belirtilen modeller)
$env:OPENROUTER_API_KEY="your_openrouter_key_here"

# Alternatif: OpenAI direkt (opsiyonel)
$env:OPENAI_API_KEY="your_openai_key_here"

# Grok (Alternatif - OpenRouter üzerinden kullanılır)
$env:GROK_API_KEY="your_grok_key_here"
```

**Not:** 
- OpenRouter API key almak için: https://openrouter.ai/keys
- PRD'de belirtilen modeller:
  - `openai/gpt-oss-20b:free` (ana senaryo motoru)
  - `x-ai/grok-4.1-fast:free` (uzun bağlamlı özetler için)
- Her iki model de OpenRouter üzerinden ücretsiz kullanılabilir
- LLM key olmadan da çalışır ama açıklamalar olmaz

---

## ADIM 3: Veritabanını Oluştur

```powershell
python -m football_brain_core.src.cli.main init-db
```

**Oluşturulan tablolar:**
- leagues, teams, matches
- stats, markets, predictions
- results, experiments, model_versions
- explanations

---

## ADIM 4: Tarihsel Veri Yükle (3-5 Sezon)

### Seçenek A: Son 5 Sezon (Önerilen - PRD'ye uygun)

```powershell
python -m football_brain_core.src.cli.main load-historical
```

**Ne yapar:**
- 7 lig için (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Liga Portugal, Süper Lig)
- Son 5 sezonun TÜM maçlarını çeker
- ~5000-10000 maç verisi

**Süre:** 1-3 saat (API limitlerine bağlı)

### Seçenek B: Belirli Sezonlar

```powershell
# Sadece 3 sezon (daha hızlı)
python -m football_brain_core.src.cli.main load-historical --seasons 2021 2022 2023

# 5 sezon (tam PRD)
python -m football_brain_core.src.cli.main load-historical --seasons 2019 2020 2021 2022 2023
```

**Önerilen:** En az 3 sezon (PRD minimum)

---

## ADIM 5: Model Eğitimi (Çoklu Market)

### Tam Model Eğitimi

```powershell
python -m football_brain_core.src.cli.main train --train-seasons 2020 2021 2022 --val-seasons 2023
```

**Ne yapar:**
- 2020-2022 sezonlarını eğitim için kullanır
- 2023 sezonunu validasyon için kullanır
- **Çoklu marketler için eğitir:**
  - Maç Sonucu (1-X-2)
  - BTTS (Var/Yok)
  - Alt/Üst 2.5
  - Ve daha fazlası...

**Süre:** 1-3 saat (veri miktarına bağlı)

### Deney Çalıştırma (Farklı Konfigürasyonlar)

```powershell
python -m football_brain_core.src.cli.main experiment --train-seasons 2020 2021 2022 --val-seasons 2023
```

**Ne yapar:**
- Farklı model konfigürasyonlarını dener
- En iyi performansı veren konfigürasyonu seçer
- Metrikleri kaydeder (Brier score, log loss, accuracy)

---

## ADIM 6: Tahmin ve LLM Açıklamaları

### Günlük Tahminler + LLM Yorumları

```python
# predict_with_explanations.py
from football_brain_core.src.inference.predict_markets import MarketPredictor
from football_brain_core.src.explanations.scenario_builder import ScenarioBuilder
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, ModelVersionRepository
from datetime import date, timedelta

# Modeli yükle (eğitilmiş model)
# ... model yükleme kodu ...

predictor = MarketPredictor(model, [
    MarketType.MATCH_RESULT,
    MarketType.BTTS,
    MarketType.OVER_UNDER_25,
    MarketType.GOAL_RANGE,
    MarketType.CORRECT_SCORE,
])

scenario_builder = ScenarioBuilder()

session = get_session()
matches = MatchRepository.get_by_date_range(
    session, 
    date.today(), 
    date.today() + timedelta(days=7)
)

active_model = ModelVersionRepository.get_active(session)

for match in matches:
    # Tahmin yap
    predictions = predictor.predict_match(match.id, session)
    
    # LLM ile açıklama üret
    explanations = scenario_builder.generate_explanation(
        match, predictions, [MarketType.MATCH_RESULT, MarketType.BTTS]
    )
    
    # Kaydet
    predictor.save_predictions(match.id, predictions, active_model.id)
    scenario_builder.save_explanations(match, explanations, {})
    
    print(f"✅ {match.id} - Tahmin ve açıklama hazır")

session.close()
```

---

## ADIM 7: Excel Öğrenme Defteri Çıktısı

### Günlük Rapor (Tahminler + Açıklamalar)

```powershell
python -m football_brain_core.src.cli.main report-daily
```

**Oluşturulan:**
- `reports/predictions_YYYY-MM-DD_YYYY-MM-DD.xlsx`
- Her maç için:
  - Tahmin edilen outcome'lar (tüm marketler)
  - Gerçek sonuçlar (varsa)
  - Doğruluk işaretleri (yeşil/kırmızı)
  - LLM yorumları (2-3 cümlelik senaryolar)
  - Özet istatistikler

### Haftalık Rapor (Backtest + Analiz)

```powershell
python -m football_brain_core.src.cli.main report-weekly
```

**İçerik:**
- Haftalık doğruluk metrikleri
- Market bazlı performans
- LLM açıklamaları
- Karşılaştırmalı analiz

---

## ADIM 8: Günlük Kullanım (İsteğe Bağlı)

### Günlük Güncelleme

```powershell
python -m football_brain_core.src.cli.main daily-update
```

**Ne yapar:**
- Yeni fikstürleri çeker
- Oynanmış maç sonuçlarını günceller
- Market sonuçlarını hesaplar

---

## 🎯 Tam PRD Workflow

### İlk Kurulum (Bir Kere)

```powershell
# 1. Paketler
pip install -r requirements.txt

# 2. API Keys
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
$env:OPENAI_API_KEY="your_key"

# 3. DB
python -m football_brain_core.src.cli.main init-db

# 4. 3-5 sezon veri yükle
python -m football_brain_core.src.cli.main load-historical --seasons 2020 2021 2022 2023

# 5. Model eğit
python -m football_brain_core.src.cli.main train --train-seasons 2020 2021 2022 --val-seasons 2023
```

### Günlük Kullanım

```powershell
# 1. Güncelle
python -m football_brain_core.src.cli.main daily-update

# 2. Tahmin + Açıklama (Python script ile - yukarıdaki kod)
python predict_with_explanations.py

# 3. Excel raporu
python -m football_brain_core.src.cli.main report-daily
```

---

## 📊 PRD Gereksinimleri Karşılanıyor mu?

- ✅ **3-5 sezon veri:** `load-historical --seasons 2020 2021 2022 2023`
- ✅ **Çoklu marketler:** Model tüm marketleri öğreniyor
- ✅ **LLM açıklamaları:** `ScenarioBuilder` ile senaryo üretimi
- ✅ **Excel çıktısı:** `report-daily` ve `report-weekly`
- ✅ **Karşılaştırmalı analiz:** Backtest ve metrikler
- ✅ **Deney takibi:** `experiment` komutu ile

---

## 🔧 Özelleştirme

### Daha Fazla Market Eklemek

`quick_test.py` veya eğitim scriptinde:

```python
market_types = [
    MarketType.MATCH_RESULT,
    MarketType.BTTS,
    MarketType.OVER_UNDER_25,
    MarketType.GOAL_RANGE,
    MarketType.CORRECT_SCORE,
    MarketType.DOUBLE_CHANCE,
    # ... daha fazlası
]
```

### Model Parametrelerini Ayarlamak

`config.py` veya eğitim sırasında:

```python
config.MODEL_CONFIG.hidden_size = 128  # Daha büyük model
config.MODEL_CONFIG.num_layers = 2     # Daha derin
config.MODEL_CONFIG.epochs = 50        # Daha uzun eğitim
```

---

## ❓ Sorun Giderme

**Veri yok hatası:**
- En az 3 sezon veri yüklendiğinden emin ol
- `load-historical` komutunun başarılı olduğunu kontrol et

**LLM açıklama yok:**
- OpenAI API key'in doğru ayarlandığından emin ol
- API key olmadan da çalışır ama açıklama olmaz

**Model eğitimi uzun sürüyor:**
- Normal (3-5 sezon veri ile 1-3 saat)
- GPU varsa otomatik kullanılır

