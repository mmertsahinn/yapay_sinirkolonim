# 🐍 PYTHON 3.12 KURULUM REHBERİ

## 📥 ADIM 1: PYTHON 3.12 İNDİR VE KUR

### Windows için:

1. **Python 3.12 İndir**
   - https://www.python.org/downloads/release/python-3127/
   - "Windows installer (64-bit)" seç

2. **Kurulum**
   - İndirilen `.exe` dosyasını çalıştır
   - **ÖNEMLİ**: "Add Python 3.12 to PATH" seçeneğini işaretle ✅
   - "Install Now" tıkla

3. **Kurulumu Doğrula**
   ```powershell
   py -3.12 --version
   ```
   Çıktı: `Python 3.12.x` olmalı

---

## 🔧 ADIM 2: VIRTUAL ENVIRONMENT OLUŞTUR

```powershell
# Proje klasörüne git
cd C:\Users\muham\Desktop\footbal_brain_core

# Python 3.12 ile virtual environment oluştur
py -3.12 -m venv venv312

# Aktif et
.\venv312\Scripts\activate
```

**Not**: Artık `python` komutu Python 3.12'yi kullanacak.

---

## 📦 ADIM 3: PAKETLERİ YÜKLE

```powershell
# snscrape'i yükle
pip install snscrape

# Diğer paketleri yükle
pip install -r requirements.txt
```

---

## ✅ ADIM 4: TEST ET

```powershell
# snscrape test
python -c "import snscrape.modules.twitter as sntwitter; print('✅ snscrape çalışıyor!')"

# Hype test
python hype_ornek_analiz.py
```

---

## 🚀 ADIM 5: HYPE ÇEKMEYİ BAŞLAT

```powershell
# Virtual environment aktifken
python tum_maclar_hype_cek.py
```

---

## 📝 NOTLAR

- Python 3.12 ile snscrape sorunsuz çalışır
- API key'e gerek yok, snscrape direkt Twitter'dan çeker
- Virtual environment kullanmak önerilir (Python 3.13 ile karışmaz)

---

## 🔄 HIZLI KURULUM (Tek Komut)

```powershell
# Python 3.12 kurulduktan sonra:
cd C:\Users\muham\Desktop\footbal_brain_core
py -3.12 -m venv venv312
.\venv312\Scripts\activate
pip install snscrape
pip install -r requirements.txt
python tum_maclar_hype_cek.py
```






