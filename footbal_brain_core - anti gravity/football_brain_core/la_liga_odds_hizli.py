"""
LA LIGA HIZLI ODDS YÜKLEME - Sadece yükleme, analiz yok
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import csv
import logging
from typing import Optional, Dict, Any, List, Tuple
import re
from difflib import SequenceMatcher

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def normalize_team_name(name: str) -> str:
    """Takım ismini normalize eder"""
    if not name:
        return ""
    name = name.strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def team_name_similarity(name1: str, name2: str) -> float:
    """İki takım ismi arasındaki benzerlik skoru"""
    norm1 = normalize_team_name(name1).lower()
    norm2 = normalize_team_name(name2).lower()
    if norm1 == norm2:
        return 1.0
    return SequenceMatcher(None, norm1, norm2).ratio()


def parse_date_safe(date_str: str, time_str: Optional[str] = None) -> Optional[datetime]:
    """Tarih parse eder"""
    try:
        if not date_str or not date_str.strip():
            return None
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                day, month, year = map(int, parts)
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
    except:
        pass
    return None


def safe_float(value: Any) -> Optional[float]:
    """String değeri float'a çevirir"""
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return float(value)
    except:
        return None


def safe_int(value: Any) -> Optional[int]:
    """String değeri int'e çevirir"""
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return int(float(value))
    except:
        return None


def find_match_fast(session: Session, league_id: int, home_team_name: str, away_team_name: str, match_date: datetime, home_score: Optional[int] = None, away_score: Optional[int] = None) -> Optional[Match]:
    """Hızlı maç bulma"""
    try:
        date_start = match_date - timedelta(days=30)
        date_end = match_date + timedelta(days=30)
        
        query = session.query(Match).filter(
            and_(
                Match.league_id == league_id,
                Match.match_date >= date_start,
                Match.match_date <= date_end
            )
        )
        
        potential_matches = query.all()
        
        if not potential_matches:
            return None
        
        home_norm = normalize_team_name(home_team_name).lower()
        away_norm = normalize_team_name(away_team_name).lower()
        
        best_match = None
        best_score = 0.0
        
        for match in potential_matches:
            try:
                home_team = TeamRepository.get_by_id(session, match.home_team_id)
                away_team = TeamRepository.get_by_id(session, match.away_team_id)
                
                if not home_team or not away_team:
                    continue
                
                score = 0.0
                
                # Skor eşleşmesi
                if home_score is not None and away_score is not None:
                    if match.home_score == home_score and match.away_score == away_score:
                        score += 300.0
                
                # Tarih
                date_diff = abs((match.match_date.date() - match_date.date()).days)
                if date_diff == 0:
                    score += 200.0
                elif date_diff <= 7:
                    score += 150.0 - date_diff * 10
                
                # Takım ismi
                home_sim = team_name_similarity(home_team_name, home_team.name)
                away_sim = team_name_similarity(away_team_name, away_team.name)
                
                if home_sim > 0.6 and away_sim > 0.6:
                    score += home_sim * 300.0 + away_sim * 300.0
                
                if score > best_score:
                    best_score = score
                    best_match = match
            except:
                continue
        
        if best_match and best_score >= 50.0:
            return best_match
        return None
    except:
        return None


def load_la_liga_odds_fast():
    """La Liga için hızlı odds yükleme"""
    print("=" * 80)
    print("LA LIGA HIZLI ODDS YÜKLEME")
    print("=" * 80)
    print()
    
    session = get_session()
    odds_dir = Path(__file__).parent / "odds" / "espana"
    
    if not odds_dir.exists():
        print(f"❌ Odds klasörü bulunamadı: {odds_dir}")
        return
    
    la_liga = LeagueRepository.get_by_name(session, "La Liga")
    if not la_liga:
        print("❌ La Liga bulunamadı!")
        return
    
    print(f"✅ La Liga bulundu (ID: {la_liga.id})")
    print()
    
    sp1_files = list(odds_dir.glob("SP1*.csv"))
    print(f"📄 {len(sp1_files)} SP1 (La Liga) CSV dosyası bulundu")
    print()
    
    stats = {
        'total_rows': 0,
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0,
        'unmatched': 0
    }
    
    for file_idx, csv_file in enumerate(sp1_files, 1):
        print(f"📄 [{file_idx}/{len(sp1_files)}] {csv_file.name}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader, 1):
                    stats['total_rows'] += 1
                    
                    if stats['total_rows'] % 500 == 0:
                        print(f"   ⏳ {stats['total_rows']:,} satır | {stats['matches_found']:,} maç | {stats['odds_added']:,} eklendi | {stats['odds_updated']:,} güncellendi | {stats['unmatched']:,} eşleşmedi")
                    
                    try:
                        date_str = row.get('Date', '').strip()
                        time_str = row.get('Time', '').strip()
                        
                        if not date_str:
                            continue
                        
                        match_date = parse_date_safe(date_str, time_str)
                        if not match_date:
                            continue
                        
                        home_team = row.get('HomeTeam', '').strip()
                        away_team = row.get('AwayTeam', '').strip()
                        
                        if not home_team or not away_team:
                            continue
                        
                        home_score = safe_int(row.get('FTHG'))
                        away_score = safe_int(row.get('FTAG'))
                        
                        match = find_match_fast(session, la_liga.id, home_team, away_team, match_date, home_score, away_score)
                        
                        if not match:
                            stats['unmatched'] += 1
                            continue
                        
                        stats['matches_found'] += 1
                        
                        odds_data = {
                            'b365_h': safe_float(row.get('B365H')),
                            'b365_d': safe_float(row.get('B365D')),
                            'b365_a': safe_float(row.get('B365A')),
                        }
                        
                        if not any(odds_data.values()):
                            continue
                        
                        try:
                            # Önce DB'de var mı kontrol et (session'dan bağımsız)
                            existing_odds = session.query(MatchOdds).filter(
                                MatchOdds.match_id == match.id
                            ).first()
                            
                            if existing_odds:
                                # Var olan kaydı güncelle
                                updated = False
                                for key, value in odds_data.items():
                                    if value is not None and getattr(existing_odds, key, None) is None:
                                        setattr(existing_odds, key, value)
                                        updated = True
                                
                                if updated:
                                    existing_odds.updated_at = datetime.utcnow()
                                    stats['odds_updated'] += 1
                            else:
                                # Yeni kayıt ekle - ama önce tekrar kontrol et
                                # (aynı transaction içinde başka bir yerde eklenmiş olabilir)
                                session.flush()  # Pending değişiklikleri DB'ye gönder
                                
                                # Tekrar kontrol et
                                existing_check = session.query(MatchOdds).filter(
                                    MatchOdds.match_id == match.id
                                ).first()
                                
                                if not existing_check:
                                    new_odds = MatchOdds(match_id=match.id, **odds_data)
                                    session.add(new_odds)
                                    stats['odds_added'] += 1
                                else:
                                    # Başka bir yerde eklenmiş, güncelle
                                    updated = False
                                    for key, value in odds_data.items():
                                        if value is not None and getattr(existing_check, key, None) is None:
                                            setattr(existing_check, key, value)
                                            updated = True
                                    if updated:
                                        existing_check.updated_at = datetime.utcnow()
                                        stats['odds_updated'] += 1
                        except Exception as e:
                            # UNIQUE constraint hatası veya başka bir hata
                            try:
                                session.rollback()
                            except:
                                pass
                            # Hata durumunda bu satırı atla
                            continue
                        
                        if stats['total_rows'] % 200 == 0:
                            try:
                                session.commit()
                                session.expunge_all()  # Session'ı temizle
                            except Exception as e:
                                session.rollback()
                                session.expunge_all()
                                logger.warning(f"Commit hatası: {e}")
                    
                    except Exception as e:
                        stats['errors'] += 1
                        session.rollback()
                        session.expunge_all()
                        continue
            
            try:
                session.commit()
                session.expunge_all()
            except Exception as e:
                session.rollback()
                session.expunge_all()
                logger.error(f"Final commit hatası {csv_file}: {e}")
        
        except Exception as e:
            logger.error(f"CSV dosya okuma hatası {csv_file}: {e}")
            stats['errors'] += 1
            try:
                session.rollback()
                session.expunge_all()
            except:
                pass
    
    print()
    print("=" * 80)
    print("📊 ÖZET")
    print("=" * 80)
    print(f"   📄 Toplam satır: {stats['total_rows']:,}")
    print(f"   ✅ Maç bulundu: {stats['matches_found']:,}")
    print(f"   ➕ Odds eklendi: {stats['odds_added']:,}")
    print(f"   🔄 Odds güncellendi: {stats['odds_updated']:,}")
    print(f"   ❌ Eşleşmeyen: {stats['unmatched']:,}")
    print(f"   ⚠️ Hatalar: {stats['errors']:,}")
    print("=" * 80)
    
    session.close()


if __name__ == "__main__":
    load_la_liga_odds_fast()

