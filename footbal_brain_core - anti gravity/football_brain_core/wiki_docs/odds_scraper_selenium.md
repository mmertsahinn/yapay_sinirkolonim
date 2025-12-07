# Module: odds_scraper_selenium.py

ODDS SCRAPER - SELENIUM İLE (JavaScript destekli)
OddsPortal modern React uygulaması olduğu için Selenium kullanıyoruz

## Functions

### parse_odds_value(odds_str)
Odds string'ini float'a çevirir

### get_driver()
Chrome WebDriver oluşturur

### extract_matches_from_page(driver, url)
Selenium ile sayfadan maç verilerini çıkarır

### parse_date(date_str, season)
Tarih string'ini parse eder

### find_match_in_db(session, league_id, home_team_name, away_team_name, match_date, home_score, away_score)
DB'de maçı bulur

### get_league_url_path(league_name)
Lig ismini OddsPortal URL path'ine çevirir

### scrape_league_season_selenium(league, season, db_session, driver)
Bir lig sezonu için odds çeker - Selenium ile

### scrape_all_leagues_selenium()
Veritabanındaki TÜM ligler için odds çeker - Selenium ile

