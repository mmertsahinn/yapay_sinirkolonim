# Module: oddsportal_full_scraper.py

OddsPortal'dan TÜM LİGLER için odds verilerini çeker

## Functions

### parse_odds_value(odds_str)
Odds string'ini float'a çevirir

### scrape_page(url, session)
Sayfayı çeker

### extract_matches_from_page(soup)
Sayfadan maç verilerini çıkarır - SKOR ve TARİH ile

### parse_date(date_str, season)
Tarih string'ini parse eder

### find_match_in_db(session, league_id, home_team_name, away_team_name, match_date, home_score, away_score)
DB'de maçı bulur - SKOR ve TARİH ile doğru eşleştirme

### scrape_league_season(league_name, season, db_session)
Bir lig sezonu için odds çeker

### scrape_all_leagues()
Tüm ligler için odds çeker

