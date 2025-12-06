# 📱 HASHTAG HYPE SİSTEMİ

## 🎯 AMAÇ

PRD: Twitter'dan lig hashtag'lerini çekerek maç öncesi hype ölçümü yapmak ve bu bilgiyi modele feature olarak eklemek.

## 🔧 KURULUM

### 1. snscrape Kurulumu

```bash
pip install snscrape
```

**Not**: snscrape Python 3.8+ gerektirir ve `libxml2` ve `libxslt` kütüphanelerine ihtiyaç duyar.

### 2. Sistem Entegrasyonu

Hashtag scraper otomatik olarak `FeatureBuilder`'a entegre edilmiştir.

## 📊 NASIL ÇALIŞIYOR?

### 1. **Lig Hashtag Mapping**

Her lig için önceden tanımlı hashtag'ler:

```python
LEAGUE_HASHTAGS = {
    "Premier League": ["#PremierLeague", "#EPL", "#PL"],
    "La Liga": ["#LaLiga", "#LaLigaSantander"],
    "Serie A": ["#SerieA", "#SerieATIM"],
    "Bundesliga": ["#Bundesliga"],
    "Ligue 1": ["#Ligue1", "#Ligue1UberEats"],
    "Liga Portugal": ["#LigaPortugal", "#PrimeiraLiga"],
    "Süper Lig": ["#SüperLig", "#SuperLig", "#TSL"],
}
```

### 2. **Takım Hashtag Mapping**

Önemli takımlar için özel hashtag'ler:

```python
TEAM_HASHTAGS = {
    "Manchester United": ["#MUFC", "#ManUnited"],
    "Real Madrid": ["#RealMadrid", "#HalaMadrid"],
    "Juventus": ["#Juve", "#ForzaJuve"],
    # ... daha fazlası
}
```

### 3. **Hype Analizi Süreci**

1. **Hashtag Çekme**: Maç tarihinden 1 gün öncesine kadar tweet'leri çeker
2. **Takım Mention Analizi**: Tweet'lerde hangi takım daha çok geçiyor?
3. **Sentiment Score**: -1 (tam away) to +1 (tam home) arası skor
4. **Feature Oluşturma**: Model için feature vector'e eklenir

### 4. **Feature Vector'e Eklenen Değerler**

```python
[
    home_support,      # 0-1 arası (ev takımı desteği)
    away_support,      # 0-1 arası (deplasman takımı desteği)
    sentiment_score,   # -1 to +1 (genel sentiment)
    total_tweets_norm  # 0-1 arası (normalize edilmiş tweet sayısı)
]
```

## 🚀 KULLANIM

### Otomatik Kullanım

`FeatureBuilder` otomatik olarak hype feature'larını ekler:

```python
from src.features.feature_builder import FeatureBuilder

feature_builder = FeatureBuilder(use_hashtag_hype=True)
features = feature_builder.build_match_features(
    home_team_id=1,
    away_team_id=2,
    match_date=datetime(2024, 12, 1),
    league_id=39,
    session=session
)
```

### Manuel Kullanım

```python
from src.ingestion.hashtag_scraper import HashtagScraper

scraper = HashtagScraper()
hype = scraper.get_match_hype(
    league_name="Premier League",
    home_team="Manchester United",
    away_team="Liverpool",
    match_date=datetime(2024, 12, 1),
    days_before=1
)

print(f"Home support: {hype['home_support']:.2%}")
print(f"Away support: {hype['away_support']:.2%}")
print(f"Sentiment: {hype['sentiment_score']:.2f}")
```

## 📈 HYPE ANALİZİ METRİKLERİ

### **home_support** / **away_support**
- 0.0 - 1.0 arası
- Takım mention'larının toplam mention'lara oranı

### **sentiment_score**
- -1.0 (tam away desteği) to +1.0 (tam home desteği)
- `home_support - away_support`

### **total_tweets**
- Toplam çekilen tweet sayısı
- Normalize edilmiş hali feature vector'de (0-1 arası)

## ⚙️ YAPILANDIRMA

### Hype Feature'larını Kapatma

```python
feature_builder = FeatureBuilder(use_hashtag_hype=False)
```

### Tarih Aralığı Ayarlama

`get_match_hype()` metodunda `days_before` parametresi ile ayarlanabilir (varsayılan: 1 gün).

### Hashtag Ekleme

`HashtagScraper` sınıfındaki `LEAGUE_HASHTAGS` ve `TEAM_HASHTAGS` dictionary'lerine yeni hashtag'ler eklenebilir.

## 🔍 ÖRNEK ÇIKTI

```
📱 Hashtag çekiliyor: #PremierLeague
✅ 45 tweet çekildi: #PremierLeague
📱 Hashtag çekiliyor: #MUFC
✅ 23 tweet çekildi: #MUFC
📊 Hype analizi: Manchester United vs Liverpool
   Home support: 65.00%
   Away support: 35.00%
   Total tweets: 68
```

## ⚠️ NOTLAR

1. **snscrape Gereksinimleri**: Sistem snscrape'in kurulu olmasını gerektirir
2. **Rate Limiting**: Twitter rate limit'leri nedeniyle çok fazla istek yapılmamalı
3. **Cache**: İleride veritabanında cache tablosu oluşturulabilir
4. **Hata Durumları**: Hata durumunda default değerler (0.5, 0.5, 0.0, 0.0) kullanılır

## 📝 GELECEK İYİLEŞTİRMELER

1. **Cache Sistemi**: Veritabanında hype cache tablosu
2. **Sentiment Analysis**: Daha gelişmiş sentiment analizi (LLM kullanarak)
3. **Real-time Updates**: Maç öncesi gerçek zamanlı güncellemeler
4. **Multi-platform**: Instagram, Reddit gibi diğer platformlar

---

**Kaynak**: [snscrape GitHub](https://github.com/JustAnotherArchivist/snscrape)






