"""
Odds CSV dosyalarını okuyup database'e yükler.
Her lig için odds klasöründeki CSV dosyalarını okur ve MatchOdds tablosuna ekler.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import csv
import logging
from typing import Optional, Dict, Any

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Lig kodları ve klasör eşleştirmesi
LEAGUE_MAPPING = {
    "E0": "england",  # Premier League
    "E1": "england",  # Championship
    "E2": "england",  # League 1
    "E3": "england",  # League 2
    "EC": "england",  # Conference
    "I1": "italy",    # Serie A
    "I2": "italy",    # Serie B
    "D1": "bundesliga",  # Bundesliga
    "D2": "bundesliga",  # 2. Bundesliga
    "F1": "france",   # Ligue 1
    "F2": "france",   # Ligue 2
    "P1": "portugal", # Primeira Liga
    "P2": "portugal", # Liga Portugal 2
    "T1": "turkey",   # Süper Lig
    "SP1": "espana",  # La Liga
    "SP2": "espana",  # Segunda División
}


def parse_date(date_str: str, time_str: Optional[str] = None) -> Optional[datetime]:
    """DD/MM/YYYY formatındaki tarihi parse eder"""
    try:
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                day, month, year = map(int, parts)
                # 2 haneli yıl için 2000 ekle
                if year < 100:
                    year += 2000
                
                hour = 19
                minute = 0
                if time_str:
                    try:
                        time_parts = time_str.split(":")
                        if len(time_parts) >= 2:
                            hour = int(time_parts[0])
                            minute = int(time_parts[1])
                    except:
                        pass
                
                return datetime(year, month, day, hour, minute)
    except Exception as e:
        logger.debug(f"Date parse hatası: {date_str}, {e}")
    return None


def safe_float(value: Any) -> Optional[float]:
    """String değeri float'a çevirir, hata durumunda None döner"""
    if value is None or value == "" or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_odds_row(row: Dict[str, str], match_date: datetime) -> Optional[Dict[str, Any]]:
    """CSV satırından odds bilgilerini çıkarır"""
    odds_data = {}
    
    # Bet365
    odds_data['b365_h'] = safe_float(row.get('B365H'))
    odds_data['b365_d'] = safe_float(row.get('B365D'))
    odds_data['b365_a'] = safe_float(row.get('B365A'))
    
    # Betfair
    odds_data['bf_h'] = safe_float(row.get('BFH'))
    odds_data['bf_d'] = safe_float(row.get('BFD'))
    odds_data['bf_a'] = safe_float(row.get('BFA'))
    
    # Betfred
    odds_data['bfd_h'] = safe_float(row.get('BFDH'))
    odds_data['bfd_d'] = safe_float(row.get('BFDD'))
    odds_data['bfd_a'] = safe_float(row.get('BFDA'))
    
    # BetMGM
    odds_data['bmgm_h'] = safe_float(row.get('BMGMH'))
    odds_data['bmgm_d'] = safe_float(row.get('BMGMD'))
    odds_data['bmgm_a'] = safe_float(row.get('BMGMA'))
    
    # Betvictor
    odds_data['bv_h'] = safe_float(row.get('BVH'))
    odds_data['bv_d'] = safe_float(row.get('BVD'))
    odds_data['bv_a'] = safe_float(row.get('BVA'))
    
    # Coral
    odds_data['cl_h'] = safe_float(row.get('CLH'))
    odds_data['cl_d'] = safe_float(row.get('CLD'))
    odds_data['cl_a'] = safe_float(row.get('CLA'))
    
    # Ladbrokes
    odds_data['lb_h'] = safe_float(row.get('LBH'))
    odds_data['lb_d'] = safe_float(row.get('LBD'))
    odds_data['lb_a'] = safe_float(row.get('LBA'))
    
    # Pinnacle
    odds_data['p_h'] = safe_float(row.get('PSH')) or safe_float(row.get('PH'))
    odds_data['p_d'] = safe_float(row.get('PSD')) or safe_float(row.get('PD'))
    odds_data['p_a'] = safe_float(row.get('PSA')) or safe_float(row.get('PA'))
    
    # William Hill
    odds_data['wh_h'] = safe_float(row.get('WHH'))
    odds_data['wh_d'] = safe_float(row.get('WHD'))
    odds_data['wh_a'] = safe_float(row.get('WHA'))
    
    # Market averages and maximums
    odds_data['max_h'] = safe_float(row.get('MaxH'))
    odds_data['max_d'] = safe_float(row.get('MaxD'))
    odds_data['max_a'] = safe_float(row.get('MaxA'))
    odds_data['avg_h'] = safe_float(row.get('AvgH'))
    odds_data['avg_d'] = safe_float(row.get('AvgD'))
    odds_data['avg_a'] = safe_float(row.get('AvgA'))
    
    # Over/Under 2.5
    odds_data['b365_over_25'] = safe_float(row.get('B365>2.5'))
    odds_data['b365_under_25'] = safe_float(row.get('B365<2.5'))
    odds_data['p_over_25'] = safe_float(row.get('P>2.5'))
    odds_data['p_under_25'] = safe_float(row.get('P<2.5'))
    odds_data['max_over_25'] = safe_float(row.get('Max>2.5'))
    odds_data['max_under_25'] = safe_float(row.get('Max<2.5'))
    odds_data['avg_over_25'] = safe_float(row.get('Avg>2.5'))
    odds_data['avg_under_25'] = safe_float(row.get('Avg<2.5'))
    
    # Asian Handicap
    odds_data['ah_h'] = safe_float(row.get('AHh'))
    odds_data['b365_ah_h'] = safe_float(row.get('B365AHH'))
    odds_data['b365_ah_a'] = safe_float(row.get('B365AHA'))
    odds_data['p_ah_h'] = safe_float(row.get('PAHH'))
    odds_data['p_ah_a'] = safe_float(row.get('PAHA'))
    
    # Closing odds
    odds_data['b365_ch'] = safe_float(row.get('B365CH'))
    odds_data['b365_cd'] = safe_float(row.get('B365CD'))
    odds_data['b365_ca'] = safe_float(row.get('B365CA'))
    
    # Tüm odds'ları JSON olarak sakla
    all_odds_dict = {}
    for key, value in row.items():
        if key not in ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                       'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
                       'HC', 'AC', 'HY', 'AY', 'HR', 'AR']:
            all_odds_dict[key] = safe_float(value)
    odds_data['all_odds'] = all_odds_dict if all_odds_dict else None
    
    return odds_data


def clean_team_name(name: str) -> str:
    """Takım ismindeki skor pattern'ini ve gereksiz karakterleri temizler"""
    import re
    # Skor pattern'ini kaldır: (0-1), (1-0), vb.
    name = re.sub(r'\(\d+-\d+\)\s*', '', name).strip()
    # Başındaki/sonundaki boşlukları temizle
    name = name.strip()
    return name


def normalize_team_name_for_matching(name: str) -> str:
    """Takım ismini eşleştirme için normalize eder"""
    name = clean_team_name(name)
    # Küçük harfe çevir
    name = name.lower()
    
    # Yaygın kelimeleri kaldır (FC, AC, etc.)
    name = name.replace('fc ', '').replace(' ac', '').replace('ac ', '')
    name = name.replace('cf ', '').replace('cfc', '').replace(' calcio', '')
    name = name.replace(' ss', '').replace('ss ', '').replace(' us', '').replace('us ', '')
    name = name.replace(' ssc', '').replace('ssc ', '').replace(' bc', '').replace('bc ', '')
    name = name.replace(' 1909', '').replace(' 1913', '').replace(' 1919', '').replace(' 1907', '')
    name = name.replace(' 1929', '').replace(' 1908', '').replace(' 1906', '')
    
    # Yaygın kısaltmaları genişlet (Serie A)
    aliases = {
        'inter': 'internazionale',
        'internazionale milano': 'internazionale',
        'milan': 'milan',
        'ac milan': 'milan',
        'roma': 'roma',
        'as roma': 'roma',
        'lazio': 'lazio',
        'juventus': 'juventus',
        'napoli': 'napoli',
        'ssc napoli': 'napoli',
        'atalanta': 'atalanta',
        'atalanta bc': 'atalanta',
        'fiorentina': 'fiorentina',
        'acf fiorentina': 'fiorentina',
        'genoa': 'genoa',
        'genoa cfc': 'genoa',
        'torino': 'torino',
        'torino fc': 'torino',
        'udinese': 'udinese',
        'udinese calcio': 'udinese',
        'sampdoria': 'sampdoria',
        'uc sampdoria': 'sampdoria',
        'sassuolo': 'sassuolo',
        'us sassuolo calcio': 'sassuolo',
        'verona': 'verona',
        'hellas verona': 'verona',
        'hellas verona fc': 'verona',
        'empoli': 'empoli',
        'empoli fc': 'empoli',
        'spezia': 'spezia',
        'spezia calcio': 'spezia',
        'venezia': 'venezia',
        'venezia fc': 'venezia',
        'salernitana': 'salernitana',
        'us salernitana': 'salernitana',
        'monza': 'monza',
        'ac monza': 'monza',
        'lecce': 'lecce',
        'us lecce': 'lecce',
        'bologna': 'bologna',
        'bologna fc': 'bologna',
        'cagliari': 'cagliari',
        'cagliari calcio': 'cagliari',
        'frosinone': 'frosinone',
        'frosinone calcio': 'frosinone',
        'como': 'como',
        'como 1907': 'como',
        'cremonese': 'cremonese',
        'us cremonese': 'cremonese',
        'crotone': 'crotone',
        'fc crotone': 'crotone',
        'parma': 'parma',
        'parma calcio': 'parma',
        'benevento': 'benevento',
        'benevento calcio': 'benevento',
        'pisa': 'pisa',
        'ac pisa': 'pisa',
    }
    
    # Alias kontrolü
    for alias, normalized in aliases.items():
        if alias in name:
            return normalized
    
    return name


def find_team_with_flexible_matching(session: Session, team_name: str, league_id: int) -> Optional[Team]:
    """Esnek takım eşleştirmesi yapar - GELİŞTİRİLMİŞ VERSİYON"""
    # 1. Önce temizlenmiş isimle tam eşleşme
    clean_name = clean_team_name(team_name)
    team = session.query(Team).filter(
        and_(Team.name == clean_name, Team.league_id == league_id)
    ).first()
    if team:
        return team
    
    # 2. DB'deki takım isimlerini temizleyip karşılaştır
    all_teams = session.query(Team).filter(Team.league_id == league_id).all()
    
    # Normalize edilmiş isimlerle eşleştir
    normalized_csv = normalize_team_name_for_matching(team_name)
    
    best_match = None
    best_score = 0
    
    for db_team in all_teams:
        clean_db_name = clean_team_name(db_team.name)
        normalized_db = normalize_team_name_for_matching(clean_db_name)
        
        # Tam eşleşme - en yüksek öncelik
        if normalized_csv == normalized_db:
            return db_team
        
        # Skor hesapla (eşleşme kalitesi)
        score = 0
        
        # Kelime bazında eşleşme
        csv_words = set(normalized_csv.split())
        db_words = set(normalized_db.split())
        common_words = csv_words & db_words
        
        if common_words:
            # Ortak kelime sayısına göre skor
            score = len(common_words) * 10
            
            # En önemli kelimeler (ilk kelime genelde takım adı)
            if normalized_csv.split()[0] == normalized_db.split()[0]:
                score += 50
            
            # Kısmi eşleşme (bir isim diğerini içeriyorsa)
            if normalized_csv in normalized_db or normalized_db in normalized_csv:
                score += 30
            
            # Uzunluk farkına göre ceza
            length_diff = abs(len(normalized_csv) - len(normalized_db))
            score -= length_diff
        
        if score > best_score:
            best_score = score
            best_match = db_team
    
    # Eğer iyi bir eşleşme bulunduysa döndür
    if best_match and best_score >= 20:
        return best_match
    
    # 3. Son çare: kısmi eşleşme (case-insensitive) - sadece temiz isimle
    clean_words = clean_name.lower().split()
    if clean_words:
        # İlk kelimeyle eşleşme dene
        first_word = clean_words[0]
        team = session.query(Team).filter(
            and_(
                Team.name.ilike(f"%{first_word}%"),
                Team.league_id == league_id
            )
        ).first()
        if team:
            return team
    
    # 4. En son çare: herhangi bir kelimeyle eşleşme
    for word in clean_words:
        if len(word) > 3:  # Çok kısa kelimeleri atla
            team = session.query(Team).filter(
                and_(
                    Team.name.ilike(f"%{word}%"),
                    Team.league_id == league_id
                )
            ).first()
            if team:
                return team
    
    return None


def find_match_in_db(
    session: Session,
    home_team_name: str,
    away_team_name: str,
    match_date: datetime,
    league_code: str
) -> Optional[Match]:
    """Database'de maçı bulur - ÖNCE TARİH, SONRA TAKIM İSİMLERİ"""
    # Lig adını bul
    league_name_map = {
        "E0": "Premier League",
        "E1": "Championship",
        "E2": "League One",
        "E3": "League Two",
        "I1": "Serie A",
        "I2": "Serie B",
        "D1": "Bundesliga",
        "D2": "2. Bundesliga",
        "F1": "Ligue 1",
        "F2": "Ligue 2",
        "P1": "Liga Portugal",  # Fixed: was "Primeira Liga"
        "P2": "Liga Portugal 2",  # Added for Portuguese second division
        "T1": "Süper Lig",
        "SP1": "La Liga",
        "SP2": "Segunda División",
    }
    
    league_name = league_name_map.get(league_code)
    if not league_name:
        return None
    
    # Lig'i bul
    league = LeagueRepository.get_by_name(session, league_name)
    if not league:
        logger.debug(f"Lig bulunamadı: {league_name}")
        return None
    
    # ÖNCE TARİH İLE MAÇLARI BUL (±3 gün tolerans - artırıldı)
    from datetime import timedelta
    date_start = match_date - timedelta(days=3)
    date_end = match_date + timedelta(days=3)
    
    # Tarih aralığındaki tüm maçları getir
    potential_matches = session.query(Match).filter(
        and_(
            Match.league_id == league.id,
            Match.match_date >= date_start,
            Match.match_date <= date_end,
            Match.home_score.isnot(None),  # Sonuç bilgisi olan maçlar
            Match.away_score.isnot(None)
        )
    ).all()
    
    if not potential_matches:
        logger.debug(f"Tarih aralığında maç bulunamadı: {match_date} ({league_name})")
        return None
    
    # Takım isimlerini normalize et
    normalized_home_csv = normalize_team_name_for_matching(home_team_name)
    normalized_away_csv = normalize_team_name_for_matching(away_team_name)
    
    best_match = None
    best_score = 0
    
    # Her potansiyel maç için takım isimlerini karşılaştır
    for match in potential_matches:
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        
        if not home_team or not away_team:
            continue
        
        # DB'deki takım isimlerini normalize et
        normalized_home_db = normalize_team_name_for_matching(home_team.name)
        normalized_away_db = normalize_team_name_for_matching(away_team.name)
        
        # Eşleşme skoru hesapla
        score = 0
        
        # Ev sahibi takım eşleşmesi
        if normalized_home_csv == normalized_home_db:
            score += 100  # Tam eşleşme
        elif normalized_home_csv in normalized_home_db or normalized_home_db in normalized_home_csv:
            score += 50  # Kısmi eşleşme
        else:
            # Kelime bazında eşleşme
            csv_words = set(normalized_home_csv.split())
            db_words = set(normalized_home_db.split())
            common_words = csv_words & db_words
            if common_words:
                score += len(common_words) * 10
        
        # Deplasman takım eşleşmesi
        if normalized_away_csv == normalized_away_db:
            score += 100  # Tam eşleşme
        elif normalized_away_csv in normalized_away_db or normalized_away_db in normalized_away_csv:
            score += 50  # Kısmi eşleşme
        else:
            # Kelime bazında eşleşme
            csv_words = set(normalized_away_csv.split())
            db_words = set(normalized_away_db.split())
            common_words = csv_words & db_words
            if common_words:
                score += len(common_words) * 10
        
        # Tarih yakınlığı bonusu (daha yakın tarih = daha yüksek skor)
        date_diff = abs((match.match_date.date() - match_date.date()).days)
        if date_diff == 0:
            score += 30  # Aynı gün (artırıldı)
        elif date_diff == 1:
            score += 20  # ±1 gün (artırıldı)
        elif date_diff == 2:
            score += 10  # ±2 gün (yeni)
        elif date_diff == 3:
            score += 5   # ±3 gün (yeni)
        
        if score > best_score:
            best_score = score
            best_match = match
    
    # Eğer yeterince yüksek skor varsa maçı döndür (eşik düşürüldü)
    if best_match and best_score >= 30:  # Daha esnek eşleştirme (50'den 30'a düşürüldü)
        return best_match
    
    # Eğer hiç eşleşme yoksa None döndür
    logger.debug(f"Takım eşleşmesi yetersiz: {home_team_name} vs {away_team_name} (skor: {best_score})")
    return None


def load_odds_from_csv(csv_path: Path, session: Session) -> Dict[str, int]:
    """CSV dosyasından odds'ları yükler"""
    stats = {
        "total_rows": 0,
        "matches_found": 0,
        "odds_added": 0,
        "odds_updated": 0,
        "errors": 0
    }
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                stats["total_rows"] += 1
                
                try:
                    # Tarih ve saat parse et
                    date_str = row.get('Date', '')
                    time_str = row.get('Time', '')
                    match_date = parse_date(date_str, time_str)
                    
                    if not match_date:
                        continue
                    
                    # Lig kodu
                    div = row.get('Div', '')
                    if not div:
                        continue
                    
                    # Takım isimleri
                    home_team = row.get('HomeTeam', '').strip()
                    away_team = row.get('AwayTeam', '').strip()
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Database'de maçı bul
                    match = find_match_in_db(session, home_team, away_team, match_date, div)
                    
                    if not match:
                        continue
                    
                    stats["matches_found"] += 1
                    
                    # Odds verilerini parse et
                    odds_data = parse_odds_row(row, match_date)
                    
                    if not odds_data:
                        continue
                    
                    # MatchOdds kaydını kontrol et
                    existing_odds = session.query(MatchOdds).filter(
                        MatchOdds.match_id == match.id
                    ).first()
                    
                    if existing_odds:
                        # Güncelle
                        for key, value in odds_data.items():
                            setattr(existing_odds, key, value)
                        existing_odds.updated_at = datetime.utcnow()
                        stats["odds_updated"] += 1
                    else:
                        # Yeni ekle
                        new_odds = MatchOdds(
                            match_id=match.id,
                            **odds_data
                        )
                        session.add(new_odds)
                        stats["odds_added"] += 1
                    
                    if stats["odds_added"] % 100 == 0:
                        session.commit()
                        logger.info(f"   ✅ {stats['odds_added']} odds eklendi...")
                
                except Exception as e:
                    stats["errors"] += 1
                    logger.debug(f"Satır işleme hatası: {e}")
                    continue
        
        session.commit()
        logger.info(f"✅ {csv_path.name}: {stats['odds_added']} eklendi, {stats['odds_updated']} güncellendi")
        
    except Exception as e:
        logger.error(f"CSV okuma hatası ({csv_path}): {e}")
        session.rollback()
        stats["errors"] += 1
    
    return stats


def load_all_odds():
    """Tüm odds klasörlerindeki CSV dosyalarını yükler"""
    odds_dir = project_root / "odds"
    
    if not odds_dir.exists():
        logger.error(f"Odds klasörü bulunamadı: {odds_dir}")
        return
    
    session = get_session()
    total_stats = {
        "total_rows": 0,
        "matches_found": 0,
        "odds_added": 0,
        "odds_updated": 0,
        "errors": 0
    }
    
    try:
        # Her lig klasörü için
        for league_folder in odds_dir.iterdir():
            if not league_folder.is_dir():
                continue
            
            logger.info(f"📂 {league_folder.name} klasörü işleniyor...")
            
            # CSV dosyalarını bul
            csv_files = list(league_folder.glob("*.csv"))
            
            for csv_file in csv_files:
                logger.info(f"   📄 {csv_file.name} işleniyor...")
                stats = load_odds_from_csv(csv_file, session)
                
                # Toplam istatistikleri güncelle
                for key in total_stats:
                    total_stats[key] += stats[key]
        
        logger.info("=" * 60)
        logger.info("📊 ÖZET:")
        logger.info(f"   Toplam satır: {total_stats['total_rows']}")
        logger.info(f"   Maç bulundu: {total_stats['matches_found']}")
        logger.info(f"   Odds eklendi: {total_stats['odds_added']}")
        logger.info(f"   Odds güncellendi: {total_stats['odds_updated']}")
        logger.info(f"   Hatalar: {total_stats['errors']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Genel hata: {e}")
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    logger.info("🎲 Odds verileri yükleniyor...")
    load_all_odds()
    logger.info("✅ Odds yükleme tamamlandı!")


