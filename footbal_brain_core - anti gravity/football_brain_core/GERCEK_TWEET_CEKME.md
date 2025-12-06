# 🐦 GERÇEK TWEET ÇEKME REHBERİ

## ⚠️ PYTHON 3.13 SORUNU

snscrape Python 3.13 ile uyumlu değil. Gerçek tweet çekmek için:

## ✅ ÇÖZÜM 1: PYTHON 3.11/3.12'YE GEÇİŞ (ÖNERİLEN)

### Adımlar:

1. **Python 3.11 veya 3.12 Kur**
   - https://www.python.org/downloads/
   - Python 3.11.9 veya 3.12.7 önerilir

2. **Virtual Environment Oluştur**
   ```bash
   python3.11 -m venv venv
   # veya
   python3.12 -m venv venv
   ```

3. **Aktif Et**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Paketleri Yükle**
   ```bash
   pip install snscrape
   pip install -r requirements.txt
   ```

5. **Test Et**
   ```bash
   python hype_ornek_analiz.py
   ```

## ✅ ÇÖZÜM 2: TWITTER API V2 (ÜCRETLİ)

### Adımlar:

1. **Twitter Developer Account Oluştur**
   - https://developer.twitter.com/
   - API key al

2. **Bearer Token Ayarla**
   ```bash
   # Windows PowerShell
   $env:TWITTER_BEARER_TOKEN="your_bearer_token_here"
   
   # Linux/Mac
   export TWITTER_BEARER_TOKEN="your_bearer_token_here"
   ```

3. **Sistem Otomatik Kullanır**
   - Kod zaten Twitter API v2 desteği içeriyor
   - Bearer token varsa otomatik kullanılır

## 📊 ŞU ANKİ AYARLAR

- ✅ **300 tweet/hashtag** çekiliyor
- ✅ **Eğitim maçları (2020-2022)** öncelikli
- ✅ **En eskiden en yeniye** sıralama
- ✅ **Eğitim maçları bitince durur**

## 🚀 KULLANIM

```bash
python tum_maclar_hype_cek.py
```

## ⚙️ AYARLAR

### Tweet Sayısı
- Lig hashtag'leri: **300 tweet**
- Takım hashtag'leri: **300 tweet**
- Toplam: ~1500-2000 tweet/maç

### Rate Limiting
- Her maç arasında: **2 saniye bekleme**
- Twitter rate limit'lerine dikkat

### Tarih Aralığı
- Maç tarihinden **1 gün öncesine** kadar
- Örnek: 1 Aralık 2025 maçı → 30 Kasım - 1 Aralık tweet'leri

## 📝 NOTLAR

1. **Python 3.13**: snscrape çalışmaz, alternatif yöntemler denenir
2. **Twitter API v2**: Ücretli ama daha stabil
3. **Nitter**: Açık kaynak alternatif (deneysel)
4. **Rate Limits**: Twitter rate limit'lerine dikkat et

## 🔧 TROUBLESHOOTING

### snscrape çalışmıyor
- Python 3.11/3.12 kullan
- veya Twitter API v2 bearer token ayarla

### Tweet çekilemiyor
- Internet bağlantısını kontrol et
- Twitter rate limit'lerini kontrol et
- Alternatif yöntemler otomatik denenir

---

**Öneri**: Python 3.11/3.12'ye geçiş yap, böylece snscrape direkt çalışır ve gerçek tweet'ler çekilir.






