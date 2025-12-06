# 🎯 ALTERNATİF HYPE ÖLÇÜM YÖNTEMLERİ

Twitter API'ler çalışmadığı için alternatif yöntemler:

## 📊 1. GOOGLE TRENDS (Önerilen ✅)

**Avantajlar:**
- ✅ Ücretsiz
- ✅ API key gerekmez
- ✅ Gerçek zamanlı arama trendleri
- ✅ Kolay kullanım

**Kurulum:**
```bash
pip install pytrends
```

**Kullanım:**
```python
from src.ingestion.alternative_hype_scraper import AlternativeHypeScraper

scraper = AlternativeHypeScraper()
hype = scraper.get_match_hype(
    league_name="Süper Lig",
    home_team="Fenerbahçe",
    away_team="Galatasaray",
    match_date=datetime(2025, 12, 1)
)
```

**Nasıl Çalışır:**
- Google Trends'de takım isimlerini arar
- Son 7 günlük arama trendlerini karşılaştırır
- Hangi takım daha çok aranıyorsa o daha fazla hype'a sahip

---

## 📱 2. REDDIT API

**Avantajlar:**
- ✅ Ücretsiz (public subreddit'ler için)
- ✅ Gerçek tartışmalar
- ✅ Takım/lig subreddit'leri

**Kurulum:**
```bash
pip install praw
```

**Kullanım:**
- Reddit API credentials opsiyonel (public subreddit'ler için)
- `superlig`, `soccer` gibi subreddit'lerden post çeker
- Takım mention sayılarını sayar

**Subreddit'ler:**
- Premier League: `r/soccer`
- Süper Lig: `r/superlig`
- La Liga: `r/soccer`
- Serie A: `r/soccer`

---

## 📰 3. NEWS API

**Avantajlar:**
- ✅ Haber sitelerinden mention sayıları
- ✅ Popülerlik skorları
- ✅ Çoklu dil desteği

**Kurulum:**
```bash
pip install newsapi-python
```

**API Key:**
1. https://newsapi.org/ adresinden ücretsiz API key al
2. `NEWS_API_KEY` environment variable olarak ayarla

**Kullanım:**
```bash
# PowerShell
$env:NEWS_API_KEY = "your_api_key_here"
```

**Limitler:**
- Ücretsiz tier: 100 request/gün
- Development: Sınırsız (localhost)

---

## 🌐 4. WEB SCRAPING (Haber Siteleri)

**Avantajlar:**
- ✅ API key gerekmez
- ✅ Direkt haber sitelerinden
- ✅ Gerçek zamanlı

**Kurulum:**
```bash
pip install requests beautifulsoup4
```

**Haber Siteleri:**
- Fanatik
- Hürriyet Spor
- Sözcü Spor

**Nasıl Çalışır:**
- Haber sitelerinin ana sayfalarını scrape eder
- Takım isimlerinin mention sayılarını sayar
- Rate limiting ile yavaş çalışır (1 saniye bekleme)

---

## 🎯 HANGİSİNİ KULLANMALI?

### Önerilen Sıralama:

1. **Google Trends** ⭐⭐⭐⭐⭐
   - En kolay, en güvenilir
   - API key gerekmez
   - Gerçek arama trendleri

2. **Reddit** ⭐⭐⭐⭐
   - Gerçek tartışmalar
   - Takım/lig subreddit'leri
   - API key opsiyonel

3. **News API** ⭐⭐⭐
   - Haber mention sayıları
   - API key gerekli (ücretsiz)
   - Günlük limit var

4. **Web Scraping** ⭐⭐
   - Yavaş
   - Rate limiting
   - Site yapısı değişebilir

---

## 🔧 ENTEGRASYON

`AlternativeHypeScraper` tüm yöntemleri otomatik dener ve sonuçları birleştirir:

```python
from src.ingestion.alternative_hype_scraper import AlternativeHypeScraper

scraper = AlternativeHypeScraper()
hype = scraper.get_match_hype(
    league_name="Süper Lig",
    home_team="Fenerbahçe",
    away_team="Galatasaray",
    match_date=datetime(2025, 12, 1)
)

print(f"Home Support: {hype['home_support']:.2%}")
print(f"Away Support: {hype['away_support']:.2%}")
print(f"Sources: {hype['sources']}")
```

**Sonuç:**
- Tüm kaynaklardan veri toplar
- Ortalama alır
- Hangi kaynakların kullanıldığını gösterir
- Hiçbiri çalışmazsa default değerler döner

---

## 📦 KURULUM (Tüm Yöntemler)

```bash
# Python 3.11 venv'inde
.\venv311\Scripts\activate

# Tüm kütüphaneleri yükle
pip install pytrends praw newsapi-python requests beautifulsoup4
```

---

## ⚠️ NOTLAR

1. **Google Trends**: En güvenilir, önerilen
2. **Reddit**: Public subreddit'ler için API key gerekmez
3. **News API**: Ücretsiz API key al (100 request/gün)
4. **Web Scraping**: Yavaş, site yapısı değişebilir

**Öneri:** Google Trends + Reddit kombinasyonu en iyi sonucu verir! 🎯






