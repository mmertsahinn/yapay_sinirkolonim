# 🚀 Projeyi Başlatma Rehberi

## ADIM 1: Paketleri Yükle

```powershell
cd football_brain_core
pip install -r requirements.txt
```

**Beklenen:** Tüm paketler başarıyla yüklenir (sqlalchemy, torch, requests, vb.)

---

## ADIM 2: API Key'leri Ayarla

PowerShell'de (her yeni terminal için tekrar yapman gerekir):

```powershell
# API-FOOTBALL (Zorunlu)
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"

# OpenRouter (LLM için - GPT ve Grok)
$env:OPENROUTER_API_KEY="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"
```

**Kontrol et:**
```powershell
echo $env:API_FOOTBALL_KEY
echo $env:OPENROUTER_API_KEY
```

---

## ADIM 3: Veritabanını Oluştur

```powershell
python -m football_brain_core.src.cli.main init-db
```

**Beklenen çıktı:**
```
INFO - Database initialized
```

**Kontrol:** `football_brain.db` dosyası oluşmuş olmalı.

---

## ADIM 4: Veri Yükle (3-5 Sezon)

### Seçenek A: 3 Sezon (Hızlı - Önerilen İlk Test)

```powershell
python -m football_brain_core.src.cli.main load-historical --seasons 2021 2022 2023
```

**Süre:** ~30-60 dakika (API limitlerine bağlı)

### Seçenek B: 5 Sezon (Tam PRD)

```powershell
python -m football_brain_core.src.cli.main load-historical --seasons 2019 2020 2021 2022 2023
```

**Süre:** ~1-3 saat

**İlerleme takibi:**
```
INFO - Loading leagues for season 2023
INFO - Loaded league: Premier League (ID: 39)
INFO - Loading data for Premier League
INFO - Loaded teams for Premier League
INFO - Loaded 380 matches for Premier League season 2023
...
```

---

## ADIM 5: Model Eğit (PRD'ye Uygun)

```powershell
python quick_test.py
```

**Ne yapar:**
- 3 sezon eğitim (2020-2022), 1 sezon validasyon (2023)
- 6 market için eğitim (MS, BTTS, Alt/Üst, Gol Aralığı, Skor, Çifte Şans)
- Tam model (128 hidden, 2 layer, 50 epoch)

**Süre:** 1-3 saat (veri miktarına bağlı)

**Beklenen çıktı:**
```
🚀 PRD'ye uygun model eğitimi başlıyor...
📊 Marketler: ['match_result', 'btts', 'over_under_25', ...]
📚 Veri hazırlanıyor...
INFO - Starting training for 50 epochs...
INFO - Epoch 1/50 - Train Loss: 2.3456, Val Loss: 2.1234
...
✅ Model eğitimi tamamlandı!
💾 Model 'model_prd_v1.0.pth' olarak kaydedildi!
```

---

## ADIM 6: Beyin Kendini Test Etsin ve Öğrensin

### Seçenek A: Tek Sezon Öğrenme

```powershell
python -m football_brain_core.src.cli.main self-learn --season 2023 --max-iterations 10 --target-accuracy 0.70
```

**Ne yapar:**
- 2023 sezonundaki maçları bugün yapılıyormuş gibi tahmin eder
- Hataları analiz eder (bias, variance, eksik feature)
- LLM ile neden yanlış olduğunu düşünür
- Mantıklı sebep bulamazsa sana sorar
- Hatalardan öğrenerek modeli günceller

### Seçenek B: Sürekli Öğrenme (Tüm Sezonlar)

```powershell
python -m football_brain_core.src.cli.main continuous-learn --seasons 2021 2022 2023 --max-iterations 10
```

**Ne yapar:**
- Tüm sezonlar üzerinde öğrenir
- Takım ilişkilerini analiz eder
- En başarılı olana kadar deneme-yanılma yapar

---

## ADIM 7: Tahmin Yap ve LLM Açıklamaları Üret

```powershell
python predict_with_explanations.py
```

**Ne yapar:**
- Gelecek maçlar için tahmin yapar
- Her tahmin için GPT ve Grok açıklama üretir
- En hızlı olanı seçer
- Veritabanına kaydeder

---

## ADIM 8: Excel Raporu Oluştur

```powershell
python -m football_brain_core.src.cli.main report-daily
```

**Oluşturulan:**
- `reports/predictions_YYYY-MM-DD_YYYY-MM-DD.xlsx`
- Her maç için:
  - Tahminler (tüm marketler)
  - Gerçek sonuçlar (varsa)
  - Doğruluk (yeşil/kırmızı)
  - GPT ve Grok açıklamaları
  - Hangi model daha hızlı (GPT/Grok)
  - Model süreleri

---

## 🎯 Hızlı Başlangıç (Tüm Adımlar Tek Seferde)

```powershell
# 1. Paketler
pip install -r requirements.txt

# 2. API Keys
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
$env:OPENROUTER_API_KEY="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"

# 3. DB
python -m football_brain_core.src.cli.main init-db

# 4. Veri (3 sezon - hızlı)
python -m football_brain_core.src.cli.main load-historical --seasons 2021 2022 2023

# 5. Model Eğit
python quick_test.py

# 6. Beyin Öğrensin
python -m football_brain_core.src.cli.main self-learn --season 2023 --max-iterations 5

# 7. Tahmin + Açıklama
python predict_with_explanations.py

# 8. Excel Raporu
python -m football_brain_core.src.cli.main report-daily
```

---

## ❓ Sorun Giderme

### "API key must be provided"
**Çözüm:** API key'leri ayarladığından emin ol
```powershell
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
```

### "ModuleNotFoundError"
**Çözüm:** Paketleri yükle
```powershell
pip install -r requirements.txt
```

### "No such table"
**Çözüm:** Veritabanını initialize et
```powershell
python -m football_brain_core.src.cli.main init-db
```

### Veri yükleme çok yavaş
**Normal:** API limitlerine bağlı. 3 sezon için 30-60 dakika normal.

### Model eğitimi uzun sürüyor
**Normal:** 3 sezon veri ile 1-3 saat normal. GPU varsa daha hızlı.

---

## 📊 İlerleme Kontrolü

### Veri var mı kontrol et:
```python
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match

session = get_session()
count = session.query(Match).count()
print(f"Toplam maç: {count}")
session.close()
```

### Model eğitildi mi kontrol et:
```python
from football_brain_core.src.db.repositories import ModelVersionRepository

session = get_session()
active = ModelVersionRepository.get_active(session)
if active:
    print(f"Aktif model: {active.version}")
else:
    print("Aktif model yok")
session.close()
```

---

## 🎉 Başarı!

Tüm adımlar tamamlandığında:
- ✅ Veritabanında 3-5 sezon veri var
- ✅ Model eğitildi ve kaydedildi
- ✅ Beyin kendini test etti ve öğrendi
- ✅ Excel raporu hazır

Artık günlük kullanım için hazırsın! 🚀







