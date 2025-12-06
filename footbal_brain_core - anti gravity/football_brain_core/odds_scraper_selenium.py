"""
ODDS SCRAPER - SELENIUM İLE (JavaScript destekli)
OddsPortal modern React uygulaması olduğu için Selenium kullanıyoruz
"""
import sys
import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("Gerekli paketler yuklu degil!")
    print("   pip install selenium webdriver-manager")
    sys.exit(1)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract
from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('odds_scraper_selenium.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_odds_value(odds_str: str) -> Optional[float]:
    """Odds string'ini float'a çevirir"""
    if not odds_str or odds_str.strip() == '':
        return None
    try:
        odds_str = re.sub(r'[^\d.]', '', odds_str)
        if odds_str:
            return float(odds_str)
    except:
        pass
    return None


def get_driver():
    """Chrome WebDriver oluşturur"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Arka planda çalış
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def extract_matches_from_page(driver, url: str) -> List[Dict]:
    """Selenium ile sayfadan maç verilerini çıkarır"""
    matches = []
    
    try:
        logger.info(f"Sayfa yukleniyor: {url}")
        driver.get(url)
        
        # Sayfanın yüklenmesini bekle (JavaScript render için)
        time.sleep(5)
        
        # Sayfanın tamamen yüklenmesini bekle
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # JavaScript'in çalışması için ek bekleme
        time.sleep(8)
        
        # Sayfa kaynağını kontrol et
        page_source = driver.page_source
        logger.debug(f"Sayfa kaynağı uzunluğu: {len(page_source)}")
        
        # Farklı selector'ları dene - OddsPortal'ın modern yapısı için
        selectors = [
            "table.table-main tbody tr",
            "table[class*='table'] tbody tr",
            "div[class*='eventRow']",
            "div[class*='event-row']",
            "div[data-testid*='match']",
            "div[data-testid*='event']",
            "tr[class*='event']",
            "tr[class*='match']",
            "div[class*='match']",
            "div[class*='result']",
            "a[href*='/match/']",
        ]
        
        rows = []
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and len(elements) > 2:  # En az 3 element olmalı (header + 2 maç)
                    rows = elements
                    logger.info(f"✅ {len(rows)} element bulundu (selector: {selector})")
                    break
            except Exception as e:
                logger.debug(f"Selector hatasi {selector}: {e}")
                continue
        
        # Eğer hala bulunamadıysa, tüm link'leri kontrol et
        if not rows:
            logger.info("Standart selector'lar calismadi, alternatif yontem deneniyor...")
            try:
                # Tüm link'leri bul (maç linkleri genelde /match/ içerir)
                links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/match/']")
                if links:
                    logger.info(f"✅ {len(links)} maç linki bulundu")
                    # Link'lerin parent elementlerini al
                    rows = [link.find_element(By.XPATH, "./..") for link in links[:50]]  # İlk 50
            except Exception as e:
                logger.debug(f"Link yontemi hatasi: {e}")
        
        if not rows:
            logger.warning("Maç satırları bulunamadi - sayfa yapısı değişmiş olabilir")
            # Debug için sayfa kaynağının bir kısmını logla
            logger.debug(f"Sayfa title: {driver.title}")
            logger.debug(f"Sayfa URL: {driver.current_url}")
            return matches
        
        for row in rows:
            try:
                # Tarih
                date_elem = None
                date_selectors = [
                    ".table-time",
                    "[class*='time']",
                    "[class*='date']",
                ]
                for sel in date_selectors:
                    try:
                        date_elem = row.find_element(By.CSS_SELECTOR, sel)
                        break
                    except:
                        continue
                
                if not date_elem:
                    continue
                
                date_str = date_elem.text.strip()
                if not date_str:
                    continue
                
                # Takımlar
                teams_elem = None
                team_selectors = [
                    ".table-participant",
                    "[class*='participant']",
                    "[class*='team']",
                ]
                for sel in team_selectors:
                    try:
                        teams_elem = row.find_element(By.CSS_SELECTOR, sel)
                        break
                    except:
                        continue
                
                if not teams_elem:
                    continue
                
                teams_text = teams_elem.text.strip()
                if ' - ' not in teams_text and ' vs ' not in teams_text:
                    continue
                
                if ' - ' in teams_text:
                    home_team, away_team = teams_text.split(' - ', 1)
                elif ' vs ' in teams_text:
                    home_team, away_team = teams_text.split(' vs ', 1)
                else:
                    continue
                
                # Skor
                score_elem = None
                score_selectors = [
                    ".table-score",
                    "[class*='score']",
                    "[class*='result']",
                ]
                for sel in score_selectors:
                    try:
                        score_elem = row.find_element(By.CSS_SELECTOR, sel)
                        break
                    except:
                        continue
                
                home_score = None
                away_score = None
                
                if score_elem:
                    score_text = score_elem.text.strip()
                    if score_text:
                        score_parts = re.split(r'[:-\s]+', score_text)
                        if len(score_parts) >= 2:
                            try:
                                home_score = int(score_parts[0])
                                away_score = int(score_parts[1])
                            except:
                                pass
                
                # Odds'lar
                odds_elems = row.find_elements(By.CSS_SELECTOR, "[class*='odds']")
                odds_data = {}
                if len(odds_elems) >= 3:
                    odds_data['home'] = parse_odds_value(odds_elems[0].text)
                    odds_data['draw'] = parse_odds_value(odds_elems[1].text)
                    odds_data['away'] = parse_odds_value(odds_elems[2].text)
                
                matches.append({
                    'date': date_str,
                    'home_team': home_team.strip(),
                    'away_team': away_team.strip(),
                    'home_score': home_score,
                    'away_score': away_score,
                    'odds': odds_data
                })
            except Exception as e:
                logger.debug(f"Satir parse hatasi: {e}")
                continue
        
        logger.info(f"{len(matches)} mac verisi cikarildi")
        return matches
        
    except Exception as e:
        logger.error(f"Sayfa isleme hatasi: {e}")
        return matches


def parse_date(date_str: str, season: str) -> Optional[datetime]:
    """Tarih string'ini parse eder"""
    formats = [
        '%d %b %Y',
        '%d/%m/%Y',
        '%d.%m.%Y',
        '%Y-%m-%d',
        '%d %B %Y',
        '%b %d, %Y',
    ]
    
    season_year = int(season.split('-')[0])
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.year < 2000:
                dt = dt.replace(year=season_year)
            return dt
        except:
            continue
    
    return None


def find_match_in_db(session, league_id: int, home_team_name: str, away_team_name: str,
                     match_date: datetime, home_score: Optional[int] = None,
                     away_score: Optional[int] = None) -> Optional[Match]:
    """DB'de maçı bulur"""
    date_start = match_date - timedelta(days=3)
    date_end = match_date + timedelta(days=3)
    
    query = session.query(Match).filter(
        and_(
            Match.league_id == league_id,
            Match.match_date >= date_start,
            Match.match_date <= date_end,
            Match.home_score.isnot(None),
            Match.away_score.isnot(None)
        )
    )
    
    if home_score is not None and away_score is not None:
        query = query.filter(
            and_(
                Match.home_score == home_score,
                Match.away_score == away_score
            )
        )
    
    potential_matches = query.all()
    
    if not potential_matches:
        return None
    
    home_normalized = home_team_name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
    away_normalized = away_team_name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
    
    best_match = None
    best_score = 0
    
    for match in potential_matches:
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        
        if not home_team or not away_team:
            continue
        
        score = 0
        
        # SKOR EŞLEŞMESİ - Öncelikli ama zorunlu değil
        if home_score is not None and away_score is not None:
            if match.home_score == home_score and match.away_score == away_score:
                score += 200  # Tam skor eşleşmesi
            # Skor eşleşmese bile devam et (daha esnek eşleştirme)
        else:
            # Skor yoksa, sadece takım isimleri ve tarih ile eşleştir
            score += 50  # Skor olmadan eşleştirme bonusu
        
        date_diff = abs((match.match_date.date() - match_date.date()).days)
        if date_diff == 0:
            score += 50
        elif date_diff == 1:
            score += 30
        elif date_diff == 2:
            score += 20
        elif date_diff == 3:
            score += 10
        
        home_db = home_team.name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
        if home_normalized == home_db:
            score += 100
        elif home_normalized in home_db or home_db in home_normalized:
            score += 50
        
        away_db = away_team.name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
        if away_normalized == away_db:
            score += 100
        elif away_normalized in away_db or away_db in away_normalized:
            score += 50
        else:
            # Kelime bazında eşleşme
            away_words = set(away_normalized.split())
            db_words = set(away_db.split())
            common = away_words & db_words
            if common:
                score += len(common) * 15  # Her ortak kelime için 15 puan
        
        if score > best_score:
            best_score = score
            best_match = match
    
    # Eşik değerini düşür - daha esnek eşleştirme
    threshold = 250 if (home_score is not None and away_score is not None) else 100
    
    if best_match and best_score >= threshold:
        return best_match
    
    return None


def get_league_url_path(league_name: str) -> Optional[str]:
    """Lig ismini OddsPortal URL path'ine çevirir"""
    LEAGUE_NAME_TO_URL = {
        'Serie A': 'italy/serie-a',
        'Serie B': 'italy/serie-b',
        'Premier League': 'england/premier-league',
        'Championship': 'england/championship',
        'League One': 'england/league-one',
        'League Two': 'england/league-two',
        'La Liga': 'spain/laliga',
        'Segunda División': 'spain/segunda-division',
        'Bundesliga': 'germany/bundesliga',
        '2. Bundesliga': 'germany/2-bundesliga',
        'Ligue 1': 'france/ligue-1',
        'Ligue 2': 'france/ligue-2',
        'Liga Portugal': 'portugal/primeira-liga',
        'Süper Lig': 'turkey/super-lig',
        'Super Lig': 'turkey/super-lig',
    }
    
    if league_name in LEAGUE_NAME_TO_URL:
        return LEAGUE_NAME_TO_URL[league_name]
    
    league_lower = league_name.lower()
    for key, value in LEAGUE_NAME_TO_URL.items():
        if key.lower() in league_lower or league_lower in key.lower():
            return value
    
    return None


def scrape_league_season_selenium(league: League, season: str, db_session, driver) -> Dict[str, int]:
    """Bir lig sezonu için odds çeker - Selenium ile"""
    stats = {
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0
    }
    
    url_path = get_league_url_path(league.name)
    if not url_path:
        logger.warning(f"Lig URL path'i bulunamadi: {league.name}")
        return stats
    
    url = f"https://www.oddsportal.com/football/{url_path}-{season}/results/"
    
    # Maçları çıkar
    matches = extract_matches_from_page(driver, url)
    logger.info(f"{league.name} {season}: {len(matches)} mac bulundu")
    
    for match_data in matches:
        try:
            match_date = parse_date(match_data['date'], season)
            if not match_date:
                continue
            
            match = find_match_in_db(
                db_session,
                league.id,
                match_data['home_team'],
                match_data['away_team'],
                match_date,
                match_data.get('home_score'),
                match_data.get('away_score')
            )
            
            if not match:
                # Daha detaylı log
                logger.warning(f"❌ Mac bulunamadi: {match_data['home_team']} vs {match_data['away_team']} "
                             f"({match_date.date()}, Skor: {match_data.get('home_score')}-{match_data.get('away_score')})")
                
                # DB'de bu tarih aralığında kaç maç var kontrol et
                date_start = match_date - timedelta(days=7)
                date_end = match_date + timedelta(days=7)
                similar_matches = db_session.query(Match).filter(
                    and_(
                        Match.league_id == league.id,
                        Match.match_date >= date_start,
                        Match.match_date <= date_end
                    )
                ).count()
                logger.debug(f"   Bu tarih aralığında {similar_matches} maç var")
                continue
            
            if match_data.get('home_score') is not None and match_data.get('away_score') is not None:
                if match.home_score != match_data['home_score'] or match.away_score != match_data['away_score']:
                    logger.info(f"Skor guncelleniyor: {match_data['home_team']} vs {match_data['away_team']}")
                    match.home_score = match_data['home_score']
                    match.away_score = match_data['away_score']
                    match.updated_at = datetime.utcnow()
            
            stats['matches_found'] += 1
            
            odds_data = match_data.get('odds', {})
            if not odds_data or not any(odds_data.values()):
                continue
            
            existing_odds = db_session.query(MatchOdds).filter(
                MatchOdds.match_id == match.id
            ).first()
            
            if existing_odds:
                if odds_data.get('home'):
                    existing_odds.b365_h = odds_data['home']
                if odds_data.get('draw'):
                    existing_odds.b365_d = odds_data['draw']
                if odds_data.get('away'):
                    existing_odds.b365_a = odds_data['away']
                existing_odds.updated_at = datetime.utcnow()
                stats['odds_updated'] += 1
            else:
                new_odds = MatchOdds(
                    match_id=match.id,
                    b365_h=odds_data.get('home'),
                    b365_d=odds_data.get('draw'),
                    b365_a=odds_data.get('away'),
                )
                db_session.add(new_odds)
                stats['odds_added'] += 1
            
            if (stats['odds_added'] + stats['odds_updated']) % 10 == 0:
                db_session.commit()
        
        except Exception as e:
            stats['errors'] += 1
            logger.error(f"Mac isleme hatasi: {e}")
            continue
    
    db_session.commit()
    return stats


def scrape_all_leagues_selenium():
    """Veritabanındaki TÜM ligler için odds çeker - Selenium ile"""
    print("=" * 80)
    print("ODDS SCRAPER - SELENIUM (JavaScript destekli)")
    print("=" * 80)
    print()
    
    session = get_session()
    driver = None
    
    try:
        driver = get_driver()
        
        all_leagues = LeagueRepository.get_all(session)
        
        if not all_leagues:
            print("⚠️ Veritabanında lig bulunamadi!")
            return
        
        print(f"📊 {len(all_leagues)} lig bulundu")
        print()
        
        seasons = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
        
        total_stats = {
            'matches_found': 0,
            'odds_added': 0,
            'odds_updated': 0,
            'errors': 0
        }
        
        for league in all_leagues:
            print(f"\n{'=' * 80}")
            print(f"LIG: {league.name} (ID: {league.id})")
            print(f"{'=' * 80}")
            
            url_path = get_league_url_path(league.name)
            if not url_path:
                print(f"⚠️ Bu lig için OddsPortal URL'i bulunamadi, atlaniyor...")
                continue
            
            print(f"✅ URL Path: {url_path}")
            
            for season in seasons:
                print(f"\n  📅 Sezon: {season}")
                stats = scrape_league_season_selenium(league, season, session, driver)
                
                for key in total_stats:
                    total_stats[key] += stats[key]
                
                print(f"     ✅ Mac bulundu: {stats['matches_found']}")
                print(f"     ✅ Odds eklendi: {stats['odds_added']}")
                print(f"     ✅ Odds guncellendi: {stats['odds_updated']}")
                if stats['errors'] > 0:
                    print(f"     ⚠️ Hatalar: {stats['errors']}")
                
                time.sleep(3)
        
        print(f"\n{'=' * 80}")
        print("📊 GENEL ÖZET")
        print(f"{'=' * 80}")
        print(f"  Toplam mac bulundu: {total_stats['matches_found']:,}")
        print(f"  Toplam odds eklendi: {total_stats['odds_added']:,}")
        print(f"  Toplam odds guncellendi: {total_stats['odds_updated']:,}")
        print(f"  Toplam hatalar: {total_stats['errors']:,}")
        print(f"{'=' * 80}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Kullanıcı tarafından durduruldu!")
        session.rollback()
    except Exception as e:
        logger.error(f"Genel hata: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
    finally:
        if driver:
            driver.quit()
        session.close()


if __name__ == "__main__":
    scrape_all_leagues_selenium()

