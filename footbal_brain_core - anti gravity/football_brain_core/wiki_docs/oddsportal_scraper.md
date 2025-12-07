# Module: oddsportal_scraper.py

OddsPortal'dan odds verilerini çeker
https://www.oddsportal.com/football/italy/serie-a-2021-2022/results/

## Functions

### parse_odds_value(odds_str)
Odds string'ini float'a çevirir (örn: "2.50" -> 2.5)

### parse_date_from_url(url)
URL'den tarih bilgisini çıkarır

### get_league_mapping_from_url(url)
URL'den lig bilgisini çıkarır

### scrape_oddsportal_page(url, session)
OddsPortal sayfasını çeker ve parse eder

### extract_match_data_from_row(row, league_name)
HTML satırından maç verilerini çıkarır

### scrape_oddsportal_season(url, db_session)
Bir sezonun tüm odds verilerini çeker

### scrape_oddsportal_url(url)
OddsPortal URL'sinden odds verilerini çeker

