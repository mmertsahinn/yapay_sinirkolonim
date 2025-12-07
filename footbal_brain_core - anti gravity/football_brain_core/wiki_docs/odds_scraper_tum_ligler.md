# Module: odds_scraper_tum_ligler.py

ODDS SCRAPER - VERİTABANINDAKİ TÜM LİGLER İÇİN OTOMATİK
OddsPortal'dan tüm ligler için odds verilerini çeker

## Functions

### parse_odds_value(odds_str)
Odds string'ini float'a çevirir

### scrape_page(url, session, retries)
Sayfayı çeker - retry mekanizması ile

### extract_matches_from_page(soup)
Sayfadan maç verilerini çıkarır

### parse_date(date_str, season)
Tarih string'ini parse eder

### find_match_in_db(session, league_id, home_team_name, away_team_name, match_date, home_score, away_score)
DB'de maçı bulur - SKOR ve TARİH ile

### get_league_url_path(league_name)
Lig ismini OddsPortal URL path'ine çevirir

### scrape_league_season(league, season, db_session)
Bir lig sezonu için odds çeker

### scrape_all_leagues_from_db()
Veritabanındaki TÜM ligler için odds çeker

