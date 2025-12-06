# 🚀 Kolay Başlatma Rehberi

## ADIM 1: API Key'leri Ayarla ve Test Et

PowerShell'de:
```powershell
cd football_brain_core
.\BASLA.ps1
```

Veya manuel:
```powershell
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
$env:OPENROUTER_API_KEY="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"
python test_apis.py
```

---

## ADIM 2: Veritabanını Oluştur

```powershell
python init_db.py
```

**Beklenen:** `✅ Veritabani hazir!`

---

## ADIM 3: Veri Yükle

### 3 Sezon (Hızlı):
```powershell
python load_data.py --seasons 2021 2022 2023
```

### 5 Sezon (Tam PRD):
```powershell
python load_data.py --seasons 2019 2020 2021 2022 2023
```

**Süre:** 30-60 dakika (3 sezon) veya 1-3 saat (5 sezon)

---

## ADIM 4: Model Eğit

```powershell
python quick_test.py
```

**Süre:** 1-3 saat

---

## ADIM 5: Günlük Kullanım

### Günlük Güncelleme:
```powershell
python run_daily.py
```

### Excel Raporu:
```powershell
python -m src.reporting.export_excel
```

---

## 🎯 Tüm Adımlar Tek Seferde

```powershell
# 1. API Keys + Test
.\BASLA.ps1

# 2. DB
python init_db.py

# 3. Veri (3 sezon)
python load_data.py --seasons 2021 2022 2023

# 4. Model
python quick_test.py
```

---

## ❓ Sorun Giderme

### "ModuleNotFoundError: No module named 'football_brain_core'"
**Çözüm:** `football_brain_core` dizininde olduğundan emin ol:
```powershell
cd football_brain_core
python init_db.py  # Kolay script kullan
```

### Import hataları
**Çözüm:** Kolay script'leri kullan (`init_db.py`, `load_data.py` vb.) - bunlar path'i otomatik düzeltir.







