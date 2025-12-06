# 📱 HYPE ÇEKİCİ NASIL ÇALIŞIR?

## 🎯 ÖRNEK: FENERBAHÇE vs GALATASARAY DERBİSİ
**Tarih:** 1 Aralık 2025  
**Lig:** Süper Lig

---

## 🔄 ADIM ADIM SÜREÇ

### 1️⃣ **HASHTAG BELİRLEME**

#### Lig Hashtag'leri:
```
#SüperLig
#SuperLig
#TSL
```

#### Takım Hashtag'leri:
```
Fenerbahçe:
  #Fenerbahce
  #FB

Galatasaray:
  #Galatasaray
  #GS
```

### 2️⃣ **TWEET ÇEKME**

**snscrape** kullanarak Twitter'dan tweet'ler çekilir:

```bash
snscrape --jsonl --max-results 50 twitter-hashtag SüperLig
snscrape --jsonl --max-results 30 twitter-hashtag Fenerbahce
snscrape --jsonl --max-results 30 twitter-hashtag FB
snscrape --jsonl --max-results 30 twitter-hashtag Galatasaray
snscrape --jsonl --max-results 30 twitter-hashtag GS
```

**Tarih Filtresi:** Maç tarihinden 1 gün öncesine kadar (30 Kasım - 1 Aralık 2025)

### 3️⃣ **TWEET ANALİZİ**

Her tweet içeriği analiz edilir:

```python
# Örnek tweet'ler:
"Fenerbahçe bugün kazanacak! #FB #SüperLig"
"Galatasaray'a inanıyorum! #GS #Galatasaray"
"Derbi çok heyecanlı olacak! #Fenerbahce #Galatasaray"
```

**Mention Sayımı:**
- Fenerbahçe mention'ları: Tweet'lerde "Fenerbahçe", "#FB", "#Fenerbahce" geçenler
- Galatasaray mention'ları: Tweet'lerde "Galatasaray", "#GS", "#Galatasaray" geçenler

### 4️⃣ **HYPE HESAPLAMA**

```python
total_mentions = home_mentions + away_mentions

home_support = home_mentions / total_mentions  # 0.0 - 1.0
away_support = away_mentions / total_mentions  # 0.0 - 1.0

sentiment_score = home_support - away_support  # -1.0 to +1.0
```

### 5️⃣ **ÖRNEK SONUÇ**

Diyelim ki 100 tweet çekildi:
- 60 tweet'te Fenerbahçe geçiyor
- 40 tweet'te Galatasaray geçiyor

**Hesaplama:**
```
home_support = 60 / 100 = 0.60 (60%)
away_support = 40 / 100 = 0.40 (40%)
sentiment_score = 0.60 - 0.40 = 0.20 (Fenerbahçe lehine)
```

**Görsel Gösterim:**
```
Fenerbahçe:     ████████████████████████ 60.0%
Galatasaray:    ████████████████ 40.0%
```

### 6️⃣ **MODEL FEATURE'LARINA EKLEME**

Bu değerler otomatik olarak model feature vector'üne eklenir:

```python
feature_vector = [
    # ... diğer feature'lar ...
    home_support,        # 0.6000
    away_support,        # 0.4000
    sentiment_score,     # 0.2000
    total_tweets_norm    # 0.0100 (100 tweet / 100 = 1.0, ama max 1.0)
]
```

---

## 📊 GERÇEK ÖRNEK ÇIKTI

```
================================================================================
HYPE ANALİZİ SONUÇLARI
================================================================================

📊 TOPLAM TWEET SAYISI: 100

🏠 FENERBAHÇE DESTEĞİ:
   • Mention sayısı: 60
   • Destek oranı: 60.00%

🟡 GALATASARAY DESTEĞİ:
   • Mention sayısı: 40
   • Destek oranı: 40.00%

📈 SENTIMENT SCORE:
   • Değer: 0.20 (Fenerbahçe lehine hafif)

📊 GÖRSEL GÖSTERİM:
--------------------------------------------------------------------------------
Fenerbahçe:     ████████████████████████ 60.0%
Galatasaray:    ████████████████ 40.0%

🤖 MODEL İÇİN FEATURE DEĞERLERİ:
--------------------------------------------------------------------------------
   • home_support: 0.6000
   • away_support: 0.4000
   • sentiment_score: 0.2000
   • total_tweets_norm: 1.0000
```

---

## 🔧 KURULUM

### snscrape Kurulumu

```bash
pip install snscrape
```

**Not:** snscrape Python 3.8+ gerektirir ve `libxml2` ve `libxslt` kütüphanelerine ihtiyaç duyar.

### Windows'ta Kurulum

```powershell
# Python paket yöneticisi ile
pip install snscrape

# Veya conda ile
conda install -c conda-forge snscrape
```

---

## ⚙️ AYARLAR

### Tarih Aralığı

```python
# Maçtan kaç gün öncesine bakılacak?
days_before = 1  # Varsayılan: 1 gün
```

### Maksimum Tweet Sayısı

```python
# Her hashtag için maksimum tweet sayısı
max_results = 50  # Lig hashtag'leri için
max_results = 30  # Takım hashtag'leri için
```

### Rate Limiting

```python
# Her maç arasında bekleme süresi
time.sleep(1)  # 1 saniye
```

---

## 🎯 KULLANIM SENARYOLARI

### Senaryo 1: Yüksek Hype
```
Total tweets: 500
Home support: 70%
Away support: 30%
Sentiment: +0.40 (Home lehine güçlü)
```
**Yorum:** Ev sahibi takım için çok yüksek destek var, model bunu dikkate alır.

### Senaryo 2: Dengeli Hype
```
Total tweets: 200
Home support: 52%
Away support: 48%
Sentiment: +0.04 (Hafif home lehine)
```
**Yorum:** Dengeli bir maç, hype çok etkili olmayabilir.

### Senaryo 3: Düşük Hype
```
Total tweets: 10
Home support: 50%
Away support: 50%
Sentiment: 0.00 (Dengeli)
```
**Yorum:** Çok az tweet var, hype feature'ları default değerlere yakın kalır.

---

## 💡 MODEL İÇİN ÖNEMİ

Hype feature'ları modelin tahmin yaparken kullandığı ek bilgilerdir:

1. **home_support / away_support**: Hangi takım daha çok destekleniyor?
2. **sentiment_score**: Genel sentiment hangi yönde?
3. **total_tweets_norm**: Ne kadar konuşuluyor? (popülerlik göstergesi)

Bu bilgiler modelin daha iyi tahmin yapmasına yardımcı olur, özellikle:
- Büyük derbilerde
- Yüksek ilgi gören maçlarda
- Sosyal medyada çok konuşulan maçlarda

---

## ⚠️ NOTLAR

1. **snscrape Gereksinimi**: Sistem snscrape'in kurulu olmasını gerektirir
2. **Rate Limiting**: Twitter rate limit'leri nedeniyle çok fazla istek yapılmamalı
3. **Tarih Filtresi**: Geçmiş maçlar için tweet bulmak zor olabilir (Twitter API limitleri)
4. **Hata Durumları**: Hata durumunda default değerler (0.5, 0.5, 0.0, 0.0) kullanılır

---

**Kaynak**: [snscrape GitHub](https://github.com/JustAnotherArchivist/snscrape)






