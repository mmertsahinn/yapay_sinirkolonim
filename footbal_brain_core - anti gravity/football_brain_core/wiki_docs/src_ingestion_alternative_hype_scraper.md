# Module: src\ingestion\alternative_hype_scraper.py

Alternatif Hype Scraper - HIZLI KAYNAKLAR
1. Google Trends - Arama trendleri (HIZLI)
2. Web Scraping - Haber siteleri (HIZLI)
News API KALDIRILDI - Zaman kazanmak için

## Classes

### AlternativeHypeScraper
Twitter API çalışmadığında alternatif yöntemlerle hype ölçer

#### Methods
- **__init__**(self)

- **get_match_hype**(self, league_name, home_team, away_team, match_date)
  - Maç için hype verilerini toplar (tüm alternatif yöntemlerden)

Returns:
    {
        "home_support": float,  # 0-1 arası
        "away_support": float,  # 0-1 arası
        "sentiment_score": float,  # -1 to +1
        "total_mentions": int,
        "hype_score": float,  # 0-1 arası genel hype
        "sources": List[str]  # Hangi kaynaklar kullanıldı
    }

- **_get_google_trends**(self, home_team, away_team, match_date)
  - Google Trends'den DERİNLEMESİNE arama trendlerini çeker
- Farklı ülkeler için ayrı ayrı
- Farklı zaman aralıkları
- İlgili terimler

- **_get_news_mentions**(self, home_team, away_team, match_date)
  - News API'den haber mention sayılarını çeker

- **_get_web_scraping_hype**(self, home_team, away_team, match_date, league_name)
  - Haber sitelerinden web scraping ile mention sayıları
- Ligin ülkesine göre filtreli siteler
- Maç tarihine yakın haberler

- **_get_team_name_variations**(self, team_name)
  - Takım ismi için farklı varyasyonlar döndürür

- **_default_hype**(self)
  - Varsayılan hype değerleri (hiç veri yoksa)

