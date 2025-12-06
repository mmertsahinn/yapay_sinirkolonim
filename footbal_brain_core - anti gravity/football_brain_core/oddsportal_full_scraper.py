"""
OddsPortal'dan TÜM LİGLER için odds verilerini çeker
"""
import sys
import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# User-Agent header
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}


# Tüm ligler için OddsPortal URL mapping
LEAGUE_URLS = {
    # Serie A
    'Serie A': {
        'base_url': 'https://www.oddsportal.com/football/italy/serie-a-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Serie B
    'Serie B': {
        'base_url': 'https://www.oddsportal.com/football/italy/serie-b-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Premier League
    'Premier League': {
        'base_url': 'https://www.oddsportal.com/football/england/premier-league-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Championship
    'Championship': {
        'base_url': 'https://www.oddsportal.com/football/england/championship-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # League One
    'League One': {
        'base_url': 'https://www.oddsportal.com/football/england/league-one-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # League Two
    'League Two': {
        'base_url': 'https://www.oddsportal.com/football/england/league-two-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # La Liga
    'La Liga': {
        'base_url': 'https://www.oddsportal.com/football/spain/laliga-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Segunda División
    'Segunda División': {
        'base_url': 'https://www.oddsportal.com/football/spain/segunda-division-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Bundesliga
    'Bundesliga': {
        'base_url': 'https://www.oddsportal.com/football/germany/bundesliga-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # 2. Bundesliga
    '2. Bundesliga': {
        'base_url': 'https://www.oddsportal.com/football/germany/2-bundesliga-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Ligue 1
    'Ligue 1': {
        'base_url': 'https://www.oddsportal.com/football/france/ligue-1-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Ligue 2
    'Ligue 2': {
        'base_url': 'https://www.oddsportal.com/football/france/ligue-2-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Liga Portugal
    'Liga Portugal': {
        'base_url': 'https://www.oddsportal.com/football/portugal/primeira-liga-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
    # Süper Lig
    'Süper Lig': {
        'base_url': 'https://www.oddsportal.com/football/turkey/super-lig-{season}/results/',
        'seasons': ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    },
}


def parse_odds_value(odds_str: str) -> Optional[float]:
    """Odds string'ini float'a çevirir"""
    if not odds_str or odds_str.strip() == '':
        return None
    try:
        # Sadece sayıları al
        odds_str = re.sub(r'[^\d.]', '', odds_str)
        if odds_str:
            return float(odds_str)
    except:
        pass
    return None


def scrape_page(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    """Sayfayı çeker"""
    try:
        logger.info(f"Sayfa cekiliyor: {url}")
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        time.sleep(3)  # Rate limiting
        soup = BeautifulSoup(response.content, 'lxml')
        return soup
    except Exception as e:
        logger.error(f"Sayfa cekme hatasi: {e}")
        return None


def extract_matches_from_page(soup: BeautifulSoup) -> List[Dict]:
    """Sayfadan maç verilerini çıkarır - SKOR ve TARİH ile"""
    matches = []
    
    # OddsPortal'ın HTML yapısına göre maç satırlarını bul
    # Farklı yapıları dene
    
    # Yöntem 1: table-main class'ı
    table = soup.find('table', class_='table-main')
    if not table:
        # Yöntem 2: Başka tablo yapıları
        tables = soup.find_all('table')
        for t in tables:
            if 'result' in t.get('class', []) or 'match' in str(t.get('class', [])).lower():
                table = t
                break
    
    if table:
        rows = table.find_all('tr')
        for row in rows:
            try:
                # Tarih - farklı yapıları dene
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
                if ' - ' not in teams_text:
                    continue
                
                home_team, away_team = teams_text.split(' - ', 1)
                
                # Skor - ÖNEMLİ: Skor bilgisi ile eşleştirme yapılacak
                score_cell = row.find('td', class_='table-score')
                if not score_cell:
                    score_cell = row.find('td', class_=re.compile(r'score|result'))
                
                score = None
                home_score = None
                away_score = None
                
                if score_cell:
                    score = score_cell.get_text(strip=True)
                    # Skor formatı: "2:1" veya "2-1" veya "2 - 1"
                    if score:
                        # Skor parse et
                        score_parts = re.split(r'[:-\s]+', score)
                        if len(score_parts) >= 2:
                            try:
                                home_score = int(score_parts[0])
                                away_score = int(score_parts[1])
                            except:
                                pass
                
                # Odds'lar - farklı bookmaker'lar için
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
                    'score': score,
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
    # Farklı formatları dene
    formats = [
        '%d %b %Y',      # "20 Aug 2021"
        '%d/%m/%Y',      # "20/08/2021"
        '%d.%m.%Y',      # "20.08.2021"
        '%Y-%m-%d',      # "2021-08-20"
        '%d %B %Y',      # "20 August 2021"
    ]
    
    # Sezon yılından yıl bilgisini çıkar
    season_year = int(season.split('-')[0])
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Eğer yıl yoksa sezon yılını kullan
            if dt.year < 2000:
                dt = dt.replace(year=season_year)
            return dt
        except:
            continue
    
    return None


def find_match_in_db(session, league_id: int, home_team_name: str, away_team_name: str, 
                     match_date: datetime, home_score: Optional[int] = None, 
                     away_score: Optional[int] = None) -> Optional[Match]:
    """DB'de maçı bulur - SKOR ve TARİH ile doğru eşleştirme"""
    # ±3 gün tolerans
    date_start = match_date - timedelta(days=3)
    date_end = match_date + timedelta(days=3)
    
    # Tarih aralığındaki maçları getir
    query = session.query(Match).filter(
        and_(
            Match.league_id == league_id,
            Match.match_date >= date_start,
            Match.match_date <= date_end,
            Match.home_score.isnot(None),
            Match.away_score.isnot(None)
        )
    )
    
    # Eğer skor bilgisi varsa, skora göre filtrele (EN ÖNEMLİ)
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
    
    # Takım isimlerini normalize et
    home_normalized = home_team_name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
    away_normalized = away_team_name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
    
    # En iyi eşleşmeyi bul
    best_match = None
    best_score = 0
    
    for match in potential_matches:
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        
        if not home_team or not away_team:
            continue
        
        score = 0
        
        # SKOR EŞLEŞMESİ - EN YÜKSEK ÖNCELİK (200 puan)
        if home_score is not None and away_score is not None:
            if match.home_score == home_score and match.away_score == away_score:
                score += 200
            else:
                # Skor eşleşmezse bu maçı atla (çok önemli!)
                continue
        
        # TARİH EŞLEŞMESİ - YÜKSEK ÖNCELİK
        date_diff = abs((match.match_date.date() - match_date.date()).days)
        if date_diff == 0:
            score += 50  # Aynı gün
        elif date_diff == 1:
            score += 30  # ±1 gün
        elif date_diff == 2:
            score += 20  # ±2 gün
        elif date_diff == 3:
            score += 10  # ±3 gün
        
        # Ev sahibi takım eşleşmesi
        home_db = home_team.name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
        if home_normalized == home_db:
            score += 100  # Tam eşleşme
        elif home_normalized in home_db or home_db in home_normalized:
            score += 50   # Kısmi eşleşme
        else:
            # Kelime bazında eşleşme
            home_words = set(home_normalized.split())
            db_words = set(home_db.split())
            common = home_words & db_words
            if common:
                score += len(common) * 10
        
        # Deplasman takım eşleşmesi
        away_db = away_team.name.lower().replace('fc ', '').replace(' ac', '').replace('cf ', '').strip()
        if away_normalized == away_db:
            score += 100  # Tam eşleşme
        elif away_normalized in away_db or away_db in away_normalized:
            score += 50   # Kısmi eşleşme
        else:
            # Kelime bazında eşleşme
            away_words = set(away_normalized.split())
            db_words = set(away_db.split())
            common = away_words & db_words
            if common:
                score += len(common) * 10
        
        if score > best_score:
            best_score = score
            best_match = match
    
    # Eşik: Skor varsa çok yüksek, yoksa daha düşük
    threshold = 300 if (home_score is not None and away_score is not None) else 150
    
    if best_match and best_score >= threshold:
        return best_match
    
    return None


def scrape_league_season(league_name: str, season: str, db_session) -> Dict[str, int]:
    """Bir lig sezonu için odds çeker"""
    stats = {
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0
    }
    
    if league_name not in LEAGUE_URLS:
        logger.warning(f"Lig URL mapping'i yok: {league_name}")
        return stats
    
    # URL oluştur
    base_url = LEAGUE_URLS[league_name]['base_url']
    url = base_url.format(season=season)
    
    # Lig'i DB'de bul
    league = LeagueRepository.get_by_name(db_session, league_name)
    if not league:
        logger.warning(f"Lig bulunamadi: {league_name}")
        return stats
    
    # HTTP session
    http_session = requests.Session()
    
    # Sayfayı çek
    soup = scrape_page(url, http_session)
    if not soup:
        return stats
    
    # Maçları çıkar
    matches = extract_matches_from_page(soup)
    logger.info(f"{league_name} {season}: {len(matches)} mac bulundu")
    
    for match_data in matches:
        try:
            # Tarihi parse et
            match_date = parse_date(match_data['date'], season)
            if not match_date:
                continue
            
            # DB'de maçı bul - SKOR ve TARİH ile
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
                           f"({match_date.date()}, Skor: {match_data.get('home_score')}-{match_data.get('away_score')})")
                continue
            
            # Skor kontrolü - DB'deki skor ile eşleşiyor mu?
            if match_data.get('home_score') is not None and match_data.get('away_score') is not None:
                if match.home_score != match_data['home_score'] or match.away_score != match_data['away_score']:
                    logger.warning(f"SKOR UYUSMAZLIĞI: {match_data['home_team']} vs {match_data['away_team']} "
                                 f"OddsPortal: {match_data['home_score']}-{match_data['away_score']}, "
                                 f"DB: {match.home_score}-{match.away_score}")
                    # DB'deki skoru güncelle (OddsPortal daha güvenilir olabilir)
                    match.home_score = match_data['home_score']
                    match.away_score = match_data['away_score']
                    match.updated_at = datetime.utcnow()
            
            stats['matches_found'] += 1
            
            # Odds'ları kaydet
            odds_data = match_data.get('odds', {})
            if not odds_data or not any(odds_data.values()):
                continue
            
            # MatchOdds kaydını kontrol et
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
            
            if stats['odds_added'] % 10 == 0:
                db_session.commit()
        
        except Exception as e:
            stats['errors'] += 1
            logger.debug(f"Mac isleme hatasi: {e}")
            continue
    
    db_session.commit()
    return stats


def scrape_all_leagues():
    """Tüm ligler için odds çeker"""
    print("=" * 80)
    print("ODDS PORTAL - TUM LIGLER SCRAPER")
    print("=" * 80)
    print()
    
    session = get_session()
    http_session = requests.Session()
    
    total_stats = {
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0
    }
    
    try:
        for league_name, config in LEAGUE_URLS.items():
            print(f"\n{'=' * 80}")
            print(f"LIG: {league_name}")
            print(f"{'=' * 80}")
            
            for season in config['seasons']:
                print(f"\nSezon: {season}")
                stats = scrape_league_season(league_name, season, session)
                
                # Toplam istatistikleri güncelle
                for key in total_stats:
                    total_stats[key] += stats[key]
                
                print(f"  Mac bulundu: {stats['matches_found']}")
                print(f"  Odds eklendi: {stats['odds_added']}")
                print(f"  Odds guncellendi: {stats['odds_updated']}")
                print(f"  Hatalar: {stats['errors']}")
                
                # Rate limiting
                time.sleep(5)
        
        print(f"\n{'=' * 80}")
        print("OZET")
        print(f"{'=' * 80}")
        print(f"  Toplam mac bulundu: {total_stats['matches_found']}")
        print(f"  Toplam odds eklendi: {total_stats['odds_added']}")
        print(f"  Toplam odds guncellendi: {total_stats['odds_updated']}")
        print(f"  Toplam hatalar: {total_stats['errors']}")
        print(f"{'=' * 80}")
        
    except Exception as e:
        logger.error(f"Genel hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()
        http_session.close()


if __name__ == "__main__":
    scrape_all_leagues()

