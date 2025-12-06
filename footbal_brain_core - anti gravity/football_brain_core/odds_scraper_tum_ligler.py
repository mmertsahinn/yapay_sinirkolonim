"""
ODDS SCRAPER - VERİTABANINDAKİ TÜM LİGLER İÇİN OTOMATİK
OddsPortal'dan tüm ligler için odds verilerini çeker
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
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Gerekli paketler yuklu degil!")
    print("   pip install beautifulsoup4 lxml requests")
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
        logging.FileHandler('odds_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# User-Agent header
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# Lig isimlerini OddsPortal URL formatına çeviren mapping
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
    'Super Lig': 'turkey/super-lig',  # Alternatif yazım
}


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


def scrape_page(url: str, session: requests.Session, retries: int = 3) -> Optional[BeautifulSoup]:
    """Sayfayı çeker - retry mekanizması ile"""
    for attempt in range(retries):
        try:
            logger.info(f"Sayfa cekiliyor (deneme {attempt+1}/{retries}): {url}")
            response = session.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            time.sleep(2)  # Rate limiting
            soup = BeautifulSoup(response.content, 'lxml')
            return soup
        except Exception as e:
            logger.warning(f"Sayfa cekme hatasi (deneme {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Sayfa cekilemedi: {url}")
    return None


def extract_matches_from_page(soup: BeautifulSoup) -> List[Dict]:
    """Sayfadan maç verilerini çıkarır"""
    matches = []
    
    # OddsPortal'ın farklı HTML yapılarını dene
    table = soup.find('table', class_='table-main')
    if not table:
        tables = soup.find_all('table')
        for t in tables:
            if 'result' in str(t.get('class', [])).lower() or 'match' in str(t.get('class', [])).lower():
                table = t
                break
    
    if not table:
        logger.warning("Tablo bulunamadi")
        return matches
    
    rows = table.find_all('tr')
    for row in rows:
        try:
            # Tarih
            date_cell = row.find('td', class_='table-time')
            if not date_cell:
                date_cell = row.find('td', class_=re.compile(r'time|date'))
            if not date_cell:
                continue
            
            date_str = date_cell.get_text(strip=True)
            if not date_str:
                continue
            
            # Takımlar
            teams_cell = row.find('td', class_='table-participant')
            if not teams_cell:
                teams_cell = row.find('td', class_=re.compile(r'participant|team'))
            if not teams_cell:
                continue
            
            teams_text = teams_cell.get_text(strip=True)
            if ' - ' not in teams_text and ' vs ' not in teams_text:
                continue
            
            # Farklı ayırıcıları dene
            if ' - ' in teams_text:
                home_team, away_team = teams_text.split(' - ', 1)
            elif ' vs ' in teams_text:
                home_team, away_team = teams_text.split(' vs ', 1)
            else:
                continue
            
            # Skor
            score_cell = row.find('td', class_='table-score')
            if not score_cell:
                score_cell = row.find('td', class_=re.compile(r'score|result'))
            
            home_score = None
            away_score = None
            
            if score_cell:
                score = score_cell.get_text(strip=True)
                if score:
                    score_parts = re.split(r'[:-\s]+', score)
                    if len(score_parts) >= 2:
                        try:
                            home_score = int(score_parts[0])
                            away_score = int(score_parts[1])
                        except:
                            pass
            
            # Odds'lar
            odds_cells = row.find_all('td', class_=re.compile(r'odds-nowrp|table-odds'))
            
            odds_data = {}
            if len(odds_cells) >= 3:
                odds_data['home'] = parse_odds_value(odds_cells[0].get_text(strip=True))
                odds_data['draw'] = parse_odds_value(odds_cells[1].get_text(strip=True))
                odds_data['away'] = parse_odds_value(odds_cells[2].get_text(strip=True))
            
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
    
    return matches


def parse_date(date_str: str, season: str) -> Optional[datetime]:
    """Tarih string'ini parse eder"""
    formats = [
        '%d %b %Y',      # "20 Aug 2021"
        '%d/%m/%Y',      # "20/08/2021"
        '%d.%m.%Y',      # "20.08.2021"
        '%Y-%m-%d',      # "2021-08-20"
        '%d %B %Y',      # "20 August 2021"
        '%b %d, %Y',     # "Aug 20, 2021"
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
    """DB'de maçı bulur - SKOR ve TARİH ile"""
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
        
        # SKOR EŞLEŞMESİ - EN ÖNEMLİ
        if home_score is not None and away_score is not None:
            if match.home_score == home_score and match.away_score == away_score:
                score += 200
            else:
                continue  # Skor eşleşmezse atla
        
        # TARİH
        date_diff = abs((match.match_date.date() - match_date.date()).days)
        if date_diff == 0:
            score += 50
        elif date_diff == 1:
            score += 30
        elif date_diff == 2:
            score += 20
        elif date_diff == 3:
            score += 10
        
        # TAKIM İSİMLERİ
        home_db = home_team.name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
        if home_normalized == home_db:
            score += 100
        elif home_normalized in home_db or home_db in home_normalized:
            score += 50
        else:
            home_words = set(home_normalized.split())
            db_words = set(home_db.split())
            common = home_words & db_words
            if common:
                score += len(common) * 10
        
        away_db = away_team.name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
        if away_normalized == away_db:
            score += 100
        elif away_normalized in away_db or away_db in away_normalized:
            score += 50
        else:
            away_words = set(away_normalized.split())
            db_words = set(away_db.split())
            common = away_words & db_words
            if common:
                score += len(common) * 10
        
        if score > best_score:
            best_score = score
            best_match = match
    
    threshold = 300 if (home_score is not None and away_score is not None) else 150
    
    if best_match and best_score >= threshold:
        return best_match
    
    return None


def get_league_url_path(league_name: str) -> Optional[str]:
    """Lig ismini OddsPortal URL path'ine çevirir"""
    # Önce direkt mapping'den bak
    if league_name in LEAGUE_NAME_TO_URL:
        return LEAGUE_NAME_TO_URL[league_name]
    
    # Kısmi eşleşme dene
    league_lower = league_name.lower()
    for key, value in LEAGUE_NAME_TO_URL.items():
        if key.lower() in league_lower or league_lower in key.lower():
            return value
    
    # Genel pattern'ler
    if 'serie a' in league_lower:
        return 'italy/serie-a'
    elif 'serie b' in league_lower:
        return 'italy/serie-b'
    elif 'premier' in league_lower:
        return 'england/premier-league'
    elif 'championship' in league_lower:
        return 'england/championship'
    elif 'la liga' in league_lower:
        return 'spain/laliga'
    elif 'bundesliga' in league_lower and '2' not in league_lower:
        return 'germany/bundesliga'
    elif 'ligue 1' in league_lower or 'ligue1' in league_lower:
        return 'france/ligue-1'
    elif 'super lig' in league_lower or 'süper lig' in league_lower:
        return 'turkey/super-lig'
    
    return None


def scrape_league_season(league: League, season: str, db_session) -> Dict[str, int]:
    """Bir lig sezonu için odds çeker"""
    stats = {
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0
    }
    
    # URL path'i al
    url_path = get_league_url_path(league.name)
    if not url_path:
        logger.warning(f"Lig URL path'i bulunamadi: {league.name}")
        return stats
    
    # URL oluştur
    url = f"https://www.oddsportal.com/football/{url_path}-{season}/results/"
    
    # HTTP session
    http_session = requests.Session()
    
    # Sayfayı çek
    soup = scrape_page(url, http_session)
    if not soup:
        logger.warning(f"Sayfa cekilemedi: {url}")
        return stats
    
    # Maçları çıkar
    matches = extract_matches_from_page(soup)
    logger.info(f"{league.name} {season}: {len(matches)} mac bulundu")
    
    for match_data in matches:
        try:
            # Tarihi parse et
            match_date = parse_date(match_data['date'], season)
            if not match_date:
                continue
            
            # DB'de maçı bul
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
                logger.debug(f"Mac bulunamadi: {match_data['home_team']} vs {match_data['away_team']} "
                           f"({match_date.date()})")
                continue
            
            # Skor kontrolü ve güncelleme
            if match_data.get('home_score') is not None and match_data.get('away_score') is not None:
                if match.home_score != match_data['home_score'] or match.away_score != match_data['away_score']:
                    logger.info(f"Skor guncelleniyor: {match_data['home_team']} vs {match_data['away_team']} "
                               f"OddsPortal: {match_data['home_score']}-{match_data['away_score']}, "
                               f"DB: {match.home_score}-{match.away_score}")
                    match.home_score = match_data['home_score']
                    match.away_score = match_data['away_score']
                    match.updated_at = datetime.utcnow()
            
            stats['matches_found'] += 1
            
            # Odds'ları kaydet
            odds_data = match_data.get('odds', {})
            if not odds_data or not any(odds_data.values()):
                continue
            
            existing_odds = db_session.query(MatchOdds).filter(
                MatchOdds.match_id == match.id
            ).first()
            
            if existing_odds:
                # Güncelle
                if odds_data.get('home'):
                    existing_odds.b365_h = odds_data['home']
                if odds_data.get('draw'):
                    existing_odds.b365_d = odds_data['draw']
                if odds_data.get('away'):
                    existing_odds.b365_a = odds_data['away']
                existing_odds.updated_at = datetime.utcnow()
                stats['odds_updated'] += 1
            else:
                # Yeni ekle
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
    http_session.close()
    return stats


def scrape_all_leagues_from_db():
    """Veritabanındaki TÜM ligler için odds çeker"""
    print("=" * 80)
    print("ODDS SCRAPER - VERİTABANINDAKİ TÜM LİGLER")
    print("=" * 80)
    print()
    
    session = get_session()
    
    # Veritabanındaki tüm ligleri al
    all_leagues = LeagueRepository.get_all(session)
    
    if not all_leagues:
        print("⚠️ Veritabanında lig bulunamadi!")
        session.close()
        return
    
    print(f"📊 {len(all_leagues)} lig bulundu")
    print()
    
    # Sezonlar (2020-2025)
    seasons = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    
    total_stats = {
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0
    }
    
    try:
        for league in all_leagues:
            print(f"\n{'=' * 80}")
            print(f"LIG: {league.name} (ID: {league.id})")
            print(f"{'=' * 80}")
            
            # Bu lig için URL path var mı?
            url_path = get_league_url_path(league.name)
            if not url_path:
                print(f"⚠️ Bu lig için OddsPortal URL'i bulunamadi, atlaniyor...")
                continue
            
            print(f"✅ URL Path: {url_path}")
            
            for season in seasons:
                print(f"\n  📅 Sezon: {season}")
                stats = scrape_league_season(league, season, session)
                
                for key in total_stats:
                    total_stats[key] += stats[key]
                
                print(f"     ✅ Mac bulundu: {stats['matches_found']}")
                print(f"     ✅ Odds eklendi: {stats['odds_added']}")
                print(f"     ✅ Odds guncellendi: {stats['odds_updated']}")
                if stats['errors'] > 0:
                    print(f"     ⚠️ Hatalar: {stats['errors']}")
                
                # Rate limiting
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
        session.close()


if __name__ == "__main__":
    scrape_all_leagues_from_db()





