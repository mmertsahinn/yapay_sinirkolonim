"""
OddsPortal'dan odds verilerini çeker
https://www.oddsportal.com/football/italy/serie-a-2021-2022/results/
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
    print("❌ Gerekli paketler yüklü değil!")
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


# User-Agent header (bot olarak algılanmamak için)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


def parse_odds_value(odds_str: str) -> Optional[float]:
    """Odds string'ini float'a çevirir (örn: "2.50" -> 2.5)"""
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


def parse_date_from_url(url: str) -> Optional[datetime]:
    """URL'den tarih bilgisini çıkarır"""
    # Örnek: /football/italy/serie-a-2021-2022/results/
    match = re.search(r'(\d{4})-(\d{4})', url)
    if match:
        start_year = int(match.group(1))
        return datetime(start_year, 8, 1)  # Sezon başlangıcı
    return None


def get_league_mapping_from_url(url: str) -> Optional[Tuple[str, str]]:
    """URL'den lig bilgisini çıkarır"""
    # Örnek: /football/italy/serie-a-2021-2022/results/
    league_mapping = {
        'serie-a': ('Serie A', 'italy'),
        'serie-b': ('Serie B', 'italy'),
        'premier-league': ('Premier League', 'england'),
        'championship': ('Championship', 'england'),
        'league-one': ('League One', 'england'),
        'league-two': ('League Two', 'england'),
        'la-liga': ('La Liga', 'espana'),
        'segunda-division': ('Segunda División', 'espana'),
        'bundesliga': ('Bundesliga', 'bundesliga'),
        '2-bundesliga': ('2. Bundesliga', 'bundesliga'),
        'ligue-1': ('Ligue 1', 'france'),
        'ligue-2': ('Ligue 2', 'france'),
        'liga-portugal': ('Liga Portugal', 'portugal'),
        'super-lig': ('Süper Lig', 'turkey'),
    }
    
    url_lower = url.lower()
    for key, (league_name, folder) in league_mapping.items():
        if key in url_lower:
            return (league_name, folder)
    return None


def scrape_oddsportal_page(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    """OddsPortal sayfasını çeker ve parse eder"""
    try:
        logger.info(f"📥 Sayfa çekiliyor: {url}")
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        # Rate limiting - saygılı olalım
        time.sleep(2)
        
        soup = BeautifulSoup(response.content, 'lxml')
        return soup
    except Exception as e:
        logger.error(f"❌ Sayfa çekme hatası: {e}")
        return None


def extract_match_data_from_row(row, league_name: str) -> Optional[Dict]:
    """HTML satırından maç verilerini çıkarır"""
    try:
        # OddsPortal'ın HTML yapısına göre parse et
        # Bu yapı site güncellemelerine göre değişebilir
        
        # Tarih
        date_elem = row.find('td', class_='table-time')
        if not date_elem:
            return None
        
        date_str = date_elem.get_text(strip=True)
        
        # Takım isimleri
        teams_elem = row.find('td', class_='table-participant')
        if not teams_elem:
            return None
        
        teams_text = teams_elem.get_text(strip=True)
        # "Team1 - Team2" formatı
        if ' - ' in teams_text:
            home_team, away_team = teams_text.split(' - ', 1)
        else:
            return None
        
        # Skor
        score_elem = row.find('td', class_='table-score')
        score = score_elem.get_text(strip=True) if score_elem else None
        
        # Odds'ları bul
        odds_cells = row.find_all('td', class_=re.compile(r'odds-nowrp|table-odds'))
        
        odds_data = {}
        
        # İlk 3 odds genelde 1X2 (Home, Draw, Away)
        if len(odds_cells) >= 3:
            odds_data['home_odds'] = parse_odds_value(odds_cells[0].get_text(strip=True))
            odds_data['draw_odds'] = parse_odds_value(odds_cells[1].get_text(strip=True))
            odds_data['away_odds'] = parse_odds_value(odds_cells[2].get_text(strip=True))
        
        return {
            'date': date_str,
            'home_team': home_team.strip(),
            'away_team': away_team.strip(),
            'score': score,
            'odds': odds_data
        }
    except Exception as e:
        logger.debug(f"Satır parse hatası: {e}")
        return None


def scrape_oddsportal_season(url: str, db_session) -> Dict[str, int]:
    """Bir sezonun tüm odds verilerini çeker"""
    stats = {
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0
    }
    
    # Lig bilgisini URL'den çıkar
    league_info = get_league_mapping_from_url(url)
    if not league_info:
        logger.error(f"❌ Lig bilgisi çıkarılamadı: {url}")
        return stats
    
    league_name, _ = league_info
    
    # Lig'i DB'de bul
    league = LeagueRepository.get_by_name(db_session, league_name)
    if not league:
        logger.error(f"❌ Lig bulunamadı: {league_name}")
        return stats
    
    # HTTP session
    http_session = requests.Session()
    
    # Sayfayı çek
    soup = scrape_oddsportal_page(url, http_session)
    if not soup:
        return stats
    
    # Maç satırlarını bul
    # OddsPortal'ın HTML yapısına göre güncellenmeli
    table = soup.find('table', class_='table-main')
    if not table:
        logger.warning(f"⚠️ Tablo bulunamadı: {url}")
        return stats
    
    rows = table.find_all('tr', class_=re.compile(r'deactivate|odd|even'))
    
    logger.info(f"📊 {len(rows)} maç satırı bulundu")
    
    for row in rows:
        try:
            match_data = extract_match_data_from_row(row, league_name)
            if not match_data:
                continue
            
            # Tarihi parse et
            # OddsPortal formatı: "20 Aug 2021" veya "20/08/2021"
            match_date = None
            date_str = match_data['date']
            
            # Farklı tarih formatlarını dene
            for fmt in ['%d %b %Y', '%d/%m/%Y', '%d.%m.%Y', '%Y-%m-%d']:
                try:
                    match_date = datetime.strptime(date_str, fmt)
                    break
                except:
                    continue
            
            if not match_date:
                logger.debug(f"Tarih parse edilemedi: {date_str}")
                continue
            
            # DB'de maçı bul
            home_team_name = match_data['home_team']
            away_team_name = match_data['away_team']
            
            # ±3 gün toleransla maç ara
            date_start = match_date - timedelta(days=3)
            date_end = match_date + timedelta(days=3)
            
            match = db_session.query(Match).filter(
                and_(
                    Match.league_id == league.id,
                    Match.match_date >= date_start,
                    Match.match_date <= date_end,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None)
                )
            ).join(Team, Match.home_team_id == Team.id).filter(
                Team.name.ilike(f"%{home_team_name}%")
            ).first()
            
            if not match:
                logger.debug(f"Maç bulunamadı: {home_team_name} vs {away_team_name} ({match_date.date()})")
                continue
            
            stats['matches_found'] += 1
            
            # Odds'ları kaydet
            odds_data = match_data.get('odds', {})
            if not odds_data:
                continue
            
            # MatchOdds kaydını kontrol et
            existing_odds = db_session.query(MatchOdds).filter(
                MatchOdds.match_id == match.id
            ).first()
            
            if existing_odds:
                # Güncelle
                if odds_data.get('home_odds'):
                    existing_odds.b365_h = odds_data['home_odds']
                if odds_data.get('draw_odds'):
                    existing_odds.b365_d = odds_data['draw_odds']
                if odds_data.get('away_odds'):
                    existing_odds.b365_a = odds_data['away_odds']
                existing_odds.updated_at = datetime.utcnow()
                stats['odds_updated'] += 1
            else:
                # Yeni ekle
                new_odds = MatchOdds(
                    match_id=match.id,
                    b365_h=odds_data.get('home_odds'),
                    b365_d=odds_data.get('draw_odds'),
                    b365_a=odds_data.get('away_odds'),
                )
                db_session.add(new_odds)
                stats['odds_added'] += 1
            
            if stats['odds_added'] % 10 == 0:
                db_session.commit()
                logger.info(f"   ✅ {stats['odds_added']} odds eklendi...")
        
        except Exception as e:
            stats['errors'] += 1
            logger.debug(f"Satır işleme hatası: {e}")
            continue
    
    db_session.commit()
    return stats


def scrape_oddsportal_url(url: str):
    """OddsPortal URL'sinden odds verilerini çeker"""
    print("=" * 80)
    print("🎲 ODDS PORTAL SCRAPER")
    print("=" * 80)
    print(f"URL: {url}")
    print()
    
    session = get_session()
    
    try:
        stats = scrape_oddsportal_season(url, session)
        
        print()
        print("=" * 80)
        print("📊 ÖZET")
        print("=" * 80)
        print(f"   Maç bulundu: {stats['matches_found']}")
        print(f"   Odds eklendi: {stats['odds_added']}")
        print(f"   Odds güncellendi: {stats['odds_updated']}")
        print(f"   Hatalar: {stats['errors']}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Genel hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    # Örnek kullanım
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Varsayılan: Serie A 2021-2022
        url = "https://www.oddsportal.com/football/italy/serie-a-2021-2022/results/"
    
    scrape_oddsportal_url(url)





