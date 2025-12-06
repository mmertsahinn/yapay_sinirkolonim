# 🧠 Minimal Beyin (Model) Kurulumu

## Hedef: En az veriyle modeli eğitip çalıştırmak

---

## ADIM 1: Paketleri Yükle

```powershell
cd football_brain_core
pip install -r requirements.txt
```

---

## ADIM 2: API Key Ayarla

```powershell
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
```

---

## ADIM 3: Veritabanını Oluştur

```powershell
python -m football_brain_core.src.cli.main init-db
```

---

## ADIM 4: Minimal Veri Yükle (Sadece 1 Sezon)

**Hızlı test için sadece 1 sezon yükle:**

```powershell
python -m football_brain_core.src.cli.main load-historical --seasons 2023
```

**Veya sadece bugünün fikstürleri (en hızlı):**

```powershell
python -m football_brain_core.src.cli.main daily-update
```

**Ne kadar sürer:**
- 1 sezon: ~10-15 dakika
- Günlük güncelleme: ~2-5 dakika

---

## ADIM 5: Model Eğit (Minimal)

**Sadece 1 sezon veri varsa:**

```powershell
python -m football_brain_core.src.cli.main train --train-seasons 2023 --val-seasons 2023
```

**2-3 sezon veri varsa (daha iyi):**

```powershell
python -m football_brain_core.src.cli.main train --train-seasons 2022 2023 --val-seasons 2023
```

**Eğitim parametrelerini azaltmak için (daha hızlı):**

Config'i düzenle veya direkt Python'da:

```python
from football_brain_core.src.models.train_offline import OfflineTrainer
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.config import Config

config = Config()
config.MODEL_CONFIG.epochs = 10  # 50 yerine 10 (daha hızlı)
config.MODEL_CONFIG.batch_size = 16  # 32 yerine 16 (daha az RAM)

market_types = [
    MarketType.MATCH_RESULT,
    MarketType.BTTS,
    MarketType.OVER_UNDER_25,
]

trainer = OfflineTrainer(market_types, config, model_config={
    "hidden_size": 64,  # 128 yerine 64 (daha küçük model)
    "num_layers": 1,    # 2 yerine 1 (daha basit)
    "dropout": 0.2
})

# Eğit
from football_brain_core.src.db.repositories import LeagueRepository
from football_brain_core.src.db.connection import get_session

session = get_session()
league_ids = [
    LeagueRepository.get_or_create(session, league.name).id
    for league in config.TARGET_LEAGUES
]
session.close()

model = trainer.train([2023], [2023], league_ids)
print("✅ Model eğitimi tamamlandı!")
```

---

## ADIM 6: Modeli Test Et

**Tahmin yap:**

```python
from football_brain_core.src.inference.predict_markets import MarketPredictor
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository

# Modeli yükle (eğitimden sonra kaydedilmiş olmalı)
# Şimdilik eğitilmiş modeli direkt kullan

predictor = MarketPredictor(model, [MarketType.MATCH_RESULT, MarketType.BTTS])

session = get_session()
matches = MatchRepository.get_by_date_range(session, date.today(), date.today() + timedelta(days=7))

for match in matches[:5]:  # İlk 5 maç
    predictions = predictor.predict_match(match.id, session)
    print(f"Maç: {match.id}")
    print(f"Tahminler: {predictions}")
    print("---")
```

---

## ⚡ Hızlı Test Scripti

`quick_test.py` dosyası oluştur:

```python
# quick_test.py
from football_brain_core.src.config import Config
from football_brain_core.src.models.train_offline import OfflineTrainer
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import LeagueRepository

print("🚀 Minimal model eğitimi başlıyor...")

config = Config()
config.MODEL_CONFIG.epochs = 5  # Çok hızlı test için
config.MODEL_CONFIG.batch_size = 16

market_types = [MarketType.MATCH_RESULT, MarketType.BTTS]

trainer = OfflineTrainer(market_types, config, model_config={
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.2
})

session = get_session()
league_ids = [
    LeagueRepository.get_or_create(session, league.name).id
    for league in config.TARGET_LEAGUES[:2]  # Sadece 2 lig (daha hızlı)
]
session.close()

print(f"📊 {len(league_ids)} lig için eğitim başlıyor...")
model = trainer.train([2023], [2023], league_ids)
print("✅ Model hazır!")
```

Çalıştır:
```powershell
python quick_test.py
```

---

## 🎯 Özet: En Hızlı Yol

```powershell
# 1. Paketler
pip install -r requirements.txt

# 2. API Key
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"

# 3. DB
python -m football_brain_core.src.cli.main init-db

# 4. Minimal veri (sadece bugün)
python -m football_brain_core.src.cli.main daily-update

# 5. Hızlı model eğitimi
python quick_test.py
```

---

## 📝 Notlar

- **Minimal veri:** Sadece bugünün fikstürleri yeterli değil model eğitimi için
- **En az 1 sezon veri** gerekli (2023 sezonu)
- **Model küçük tutuldu** (hızlı eğitim için)
- **Performans düşük olabilir** (az veri + küçük model)
- **Asıl eğitim için:** 3-5 sezon veri + tam model gerekli







