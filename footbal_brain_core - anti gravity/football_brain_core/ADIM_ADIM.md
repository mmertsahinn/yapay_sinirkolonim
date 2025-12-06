# 🚀 Adım Adım Kurulum ve Çalıştırma Rehberi

## ADIM 1: Python Ortamını Hazırla

### 1.1 Python Versiyonunu Kontrol Et
```powershell
python --version
```
**Beklenen:** Python 3.8 veya üzeri olmalı

### 1.2 Gerekli Paketleri Yükle
```powershell
cd football_brain_core
pip install -r requirements.txt
```

**Yüklenecek paketler:**
- sqlalchemy (veritabanı)
- requests (API çağrıları)
- torch (machine learning)
- numpy, pandas (veri işleme)
- openpyxl (Excel)
- scikit-learn (metrikler)
- pyyaml (config)

---

## ADIM 2: API Key'i Ayarla

### 2.1 PowerShell'de Ortam Değişkenini Ayarla
```powershell
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
```

### 2.2 Kontrol Et (Opsiyonel)
```powershell
echo $env:API_FOOTBALL_KEY
```
API key'in göründüğünü doğrula.

**Not:** Bu ayar sadece o PowerShell penceresi için geçerli. Kalıcı yapmak için:
- Sistem Özellikleri > Ortam Değişkenleri
- Veya `.env` dosyası kullan (ileride ekleyebiliriz)

---

## ADIM 3: Veritabanını Oluştur

### 3.1 Veritabanı Tablolarını Oluştur
```powershell
python -m football_brain_core.src.cli.main init-db
```

**Beklenen çıktı:**
```
INFO - Database initialized
```

**Oluşturulan:**
- `football_brain.db` dosyası (proje klasöründe)
- Tüm tablolar (leagues, teams, matches, predictions, vb.)

### 3.2 Kontrol Et (Opsiyonel)
```powershell
# Dosyanın oluştuğunu kontrol et
dir football_brain.db
```

---

## ADIM 4: API'yi Test Et (Opsiyonel ama Önerilen)

### 4.1 API Bağlantısını Test Et
```powershell
python test_api.py
```

**Beklenen çıktı:**
```
✅ API testi başarılı! Projeyi kullanmaya hazırsın.
```

**Hata alırsan:**
- API key'in doğru ayarlandığından emin ol
- İnternet bağlantını kontrol et

---

## ADIM 5: Veri Yükleme

### 5.1 Seçenek A: Tarihsel Veri (İlk Kurulum - Önerilen)

**Son 5 sezonun tüm maçlarını yükle:**
```powershell
python -m football_brain_core.src.cli.main load-historical
```

**Ne yapar:**
- 7 lig için (Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Liga Portugal, Süper Lig)
- Son 5 sezonun tüm maçlarını çeker
- Ligleri, takımları, maçları veritabanına yazar

**Süre:** API limitlerine bağlı (30 dakika - 2 saat arası)

**İlerleme takibi:**
```
INFO - Loading leagues for season 2024
INFO - Loaded league: Premier League (ID: 39)
INFO - Loading data for Premier League
INFO - Loaded teams for Premier League
INFO - Loaded 380 matches for Premier League season 2024
...
```

### 5.2 Seçenek B: Sadece Bugünün Fikstürleri (Hızlı Test)

**Sadece yakın tarihleri yükle:**
```powershell
python -m football_brain_core.src.cli.main daily-update
```

**Ne yapar:**
- Bugün ve önümüzdeki 7 günün fikstürlerini çeker
- Son 7 günün maç sonuçlarını günceller
- Daha hızlı (5-10 dakika)

**Hangi seçeneği seçmeliyim?**
- **İlk kurulum:** Seçenek A (tarihsel veri)
- **Hızlı test:** Seçenek B (günlük güncelleme)

---

## ADIM 6: Veri Yükleme Kontrolü

### 6.1 Veritabanında Veri Var mı Kontrol Et

**Python ile kontrol:**
```python
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository

session = get_session()
matches = session.query(MatchRepository).count()
print(f"Toplam maç sayısı: {matches}")
session.close()
```

**Veya basit script:**
```powershell
python -c "from football_brain_core.src.db.connection import get_session; from football_brain_core.src.db.schema import Match; s = get_session(); print(f'Maç sayısı: {s.query(Match).count()}'); s.close()"
```

**Beklenen:** En az birkaç yüz maç olmalı (tarihsel veri yüklediysen)

---

## ADIM 7: Model Eğitimi (Veri Yüklendikten Sonra)

### 7.1 Model Eğit
```powershell
python -m football_brain_core.src.cli.main train --train-seasons 2020 2021 2022 --val-seasons 2023
```

**Ne yapar:**
- 2020-2022 sezonlarını eğitim için kullanır
- 2023 sezonunu validasyon için kullanır
- Multi-task model eğitir (Maç Sonucu, BTTS, Alt/Üst marketleri için)

**Süre:** 30 dakika - 2 saat (veri miktarına ve bilgisayar hızına bağlı)

**İlerleme takibi:**
```
INFO - Preparing data...
INFO - Starting training for 50 epochs...
INFO - Epoch 1/50 - Train Loss: 2.3456, Val Loss: 2.1234
...
```

---

## ADIM 8: Günlük Kullanım

### 8.1 Günlük Güncelleme
```powershell
python -m football_brain_core.src.cli.main daily-update
```

**Ne zaman çalıştır:**
- Her gün (yeni fikstürler ve sonuçlar için)

### 8.2 Rapor Oluştur
```powershell
# Günlük rapor
python -m football_brain_core.src.cli.main report-daily

# Haftalık rapor
python -m football_brain_core.src.cli.main report-weekly
```

**Oluşturulan:**
- `reports/predictions_YYYY-MM-DD_YYYY-MM-DD.xlsx` dosyası
- Her maç için tahminler, sonuçlar, doğruluk işaretleri

---

## ❌ Sık Karşılaşılan Hatalar ve Çözümleri

### Hata 1: "API key must be provided"
**Çözüm:** API key'i ayarladığından emin ol
```powershell
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
```

### Hata 2: "ModuleNotFoundError"
**Çözüm:** Paketleri yükle
```powershell
pip install -r requirements.txt
```

### Hata 3: "No such table"
**Çözüm:** Veritabanını initialize et
```powershell
python -m football_brain_core.src.cli.main init-db
```

### Hata 4: API Rate Limit
**Çözüm:** Bekle ve tekrar dene (ücretsiz planda günlük limit var)

---

## ✅ Başarı Kontrol Listesi

- [ ] Python 3.8+ yüklü
- [ ] `pip install -r requirements.txt` başarılı
- [ ] API key ayarlandı
- [ ] `init-db` başarılı
- [ ] `load-historical` veya `daily-update` başarılı
- [ ] Veritabanında veri var
- [ ] Model eğitimi başarılı (opsiyonel)

---

## 🎯 Hızlı Başlangıç (Özet)

```powershell
# 1. Paketleri yükle
pip install -r requirements.txt

# 2. API key ayarla
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"

# 3. Veritabanı oluştur
python -m football_brain_core.src.cli.main init-db

# 4. Veri yükle (seçeneklerden biri)
python -m football_brain_core.src.cli.main load-historical
# VEYA
python -m football_brain_core.src.cli.main daily-update

# 5. Model eğit (veri yüklendikten sonra)
python -m football_brain_core.src.cli.main train --train-seasons 2020 2021 2022 --val-seasons 2023
```

---

## 📞 Yardım

Bir adımda takıldıysan, hangi adımda olduğunu ve aldığın hatayı söyle!







