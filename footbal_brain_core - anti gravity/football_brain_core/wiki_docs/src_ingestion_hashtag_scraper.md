# Module: src\ingestion\hashtag_scraper.py

Hashtag Scraper - Twitter'dan lig hashtag'lerini çeker ve hype analizi yapar
snscrape kullanarak: https://github.com/JustAnotherArchivist/snscrape

## Classes

### HashtagScraper
PRD: Hashtag çekici ve hype analizi
- Lig hashtag'lerini çeker
- Maç tarihinde hype ölçer
- Hangi takım daha çok destekleniyor analiz eder

#### Methods
- **__init__**(self)

- **get_league_hashtags**(self, league_name)
  - Lig için hashtag listesi döndür

- **get_team_hashtags**(self, team_name)
  - Takım için hashtag listesi döndür

- **scrape_hashtag**(self, hashtag, date_from, date_to, max_results)
  - snscrape kullanarak hashtag'den tweet'leri çeker (library olarak)

Args:
    hashtag: Hashtag (örn: #PremierLeague)
    date_from: Başlangıç tarihi
    date_to: Bitiş tarihi
    max_results: Maksimum sonuç sayısı
    
Returns:
    Tweet listesi (JSON formatında)

- **_scrape_with_alternative**(self, hashtag, date_from, date_to, max_results)
  - Alternatif yöntemlerle gerçek tweet çek

- **analyze_hype**(self, tweets, home_team, away_team)
  - Tweet'lerden hype analizi yapar
Hangi takım daha çok destekleniyor?

Returns:
    {
        "home_support": float,  # 0-1 arası
        "away_support": float,  # 0-1 arası
        "total_tweets": int,
        "home_mentions": int,
        "away_mentions": int,
        "sentiment_score": float  # -1 (away) to +1 (home)
    }

- **_normalize_team_name**(self, team_name)
  - Takım ismini normalize et (arama için)

- **get_match_hype**(self, league_name, home_team, away_team, match_date, days_before)
  - Maç için hype analizi yapar

Args:
    league_name: Lig adı
    home_team: Ev sahibi takım
    away_team: Deplasman takımı
    match_date: Maç tarihi
    days_before: Maçtan kaç gün öncesine bakılacak
    
Returns:
    Hype analizi sonuçları

- **get_match_hype_cached**(self, match_id, league_name, home_team, away_team, match_date)
  - Cache'li hype analizi (veritabanından önce kontrol eder)
Twitter çalışmazsa alternatif yöntemleri kullanır

