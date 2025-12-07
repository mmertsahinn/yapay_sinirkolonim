# Module: src\ingestion\twitter_scraper_alt.py

Alternatif Twitter Scraper - Gerçek tweet çekmek için
snscrape Python 3.13 ile uyumlu olmadığı için alternatif yöntemler

## Classes

### TwitterScraperAlternative
Alternatif Twitter scraper - Gerçek tweet çekmek için

#### Methods
- **__init__**(self)

- **scrape_hashtag_via_nitter**(self, hashtag, date_from, date_to, max_results)
  - Nitter instance kullanarak tweet çek (snscrape alternatifi)
Nitter: Twitter'ın açık kaynak alternatifi, API key gerektirmez

- **scrape_hashtag_via_twitter_api_v2**(self, hashtag, date_from, date_to, max_results, bearer_token)
  - Twitter API v2 kullanarak tweet çek (API key gerektirir)

## Functions

### get_twitter_scraper()
Twitter scraper instance döndür

