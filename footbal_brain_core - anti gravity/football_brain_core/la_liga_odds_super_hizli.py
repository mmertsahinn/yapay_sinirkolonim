"""
LA LIGA SÜPER HIZLI ODDS YÜKLEME - Cache kullanarak
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import csv
from typing import Optional, Set
import re
from difflib import SequenceMatcher

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import LeagueRepository, TeamRepository

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

def safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return float(value)
    except:
        return None

def safe_int(value) -> Optional[int]:
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return int(float(value))
    except:
        return None

def load_la_liga_odds_super_fast():
    """La Liga için süper hızlı odds yükleme - Cache kullanarak"""
    print("=" * 80)
    print("LA LIGA SÜPER HIZLI ODDS YÜKLEME")
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
    
    # ÖNCE: Tüm mevcut MatchOdds'ları cache'le (çok hızlı)
    print("📦 Mevcut odds'lar cache'leniyor...")
    existing_odds_match_ids = set(
        session.query(MatchOdds.match_id).all()
    )
    existing_odds_match_ids = {row[0] for row in existing_odds_match_ids}
    print(f"   ✅ {len(existing_odds_match_ids):,} mevcut odds cache'lendi")
    print()
    
    # La Liga maçlarını cache'le (tarih aralığına göre)
    print("📦 La Liga maçları cache'leniyor...")
    la_liga_matches = session.query(Match).filter(
        Match.league_id == la_liga.id
    ).all()
    
    # Match cache: (date, home_team_name, away_team_name) -> Match
    match_cache = {}
    for match in la_liga_matches:
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        if home_team and away_team:
            key = (
                match.match_date.date(),
                normalize_team_name(home_team.name).lower(),
                normalize_team_name(away_team.name).lower()
            )
            match_cache[key] = match
    print(f"   ✅ {len(match_cache):,} maç cache'lendi")
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
        'unmatched': 0,
        'already_exists': 0
    }
    
    # Toplanacak odds'lar (batch insert için)
    odds_to_add = []
    odds_to_update = []
    
    for file_idx, csv_file in enumerate(sp1_files, 1):
        print(f"📄 [{file_idx}/{len(sp1_files)}] {csv_file.name}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader, 1):
                    stats['total_rows'] += 1
                    
                    if stats['total_rows'] % 1000 == 0:
                        print(f"   ⏳ {stats['total_rows']:,} satır | {stats['matches_found']:,} maç | {stats['odds_added']:,} eklenecek | {stats['odds_updated']:,} güncellenecek | {stats['unmatched']:,} eşleşmedi")
                    
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
                        
                        # Cache'den maçı bul
                        home_norm = normalize_team_name(home_team).lower()
                        away_norm = normalize_team_name(away_team).lower()
                        match_date_key = match_date.date()
                        
                        match = None
                        
                        # Önce tam eşleşme dene
                        key = (match_date_key, home_norm, away_norm)
                        if key in match_cache:
                            match = match_cache[key]
                        else:
                            # Benzerlik ile ara (sadece aynı gün)
                            best_match = None
                            best_score = 0.0
                            
                            for (cached_date, cached_home, cached_away), cached_match in match_cache.items():
                                if cached_date == match_date_key:
                                    home_sim = team_name_similarity(home_team, cached_home)
                                    away_sim = team_name_similarity(away_team, cached_away)
                                    
                                    if home_sim > 0.6 and away_sim > 0.6:
                                        score = home_sim + away_sim
                                        if score > best_score:
                                            best_score = score
                                            best_match = cached_match
                            
                            if best_match and best_score >= 1.2:
                                match = best_match
                        
                        if not match:
                            stats['unmatched'] += 1
                            continue
                        
                        stats['matches_found'] += 1
                        
                        # Odds'ları hazırla
                        odds_data = {
                            'b365_h': safe_float(row.get('B365H')),
                            'b365_d': safe_float(row.get('B365D')),
                            'b365_a': safe_float(row.get('B365A')),
                        }
                        
                        if not any(odds_data.values()):
                            continue
                        
                        # Cache'den kontrol et
                        if match.id in existing_odds_match_ids:
                            stats['already_exists'] += 1
                            # Güncelleme için kaydet
                            odds_to_update.append((match.id, odds_data))
                        else:
                            # Yeni ekleme için kaydet
                            odds_to_add.append((match.id, odds_data))
                            existing_odds_match_ids.add(match.id)  # Cache'e ekle
                    
                    except Exception as e:
                        stats['errors'] += 1
                        continue
            
        except Exception as e:
            print(f"   ❌ CSV dosya okuma hatası: {e}")
            stats['errors'] += 1
    
    print()
    print("=" * 80)
    print("💾 VERİTABANINA YAZILIYOR...")
    print("=" * 80)
    print(f"   ➕ {len(odds_to_add):,} yeni odds eklenecek")
    print(f"   🔄 {len(odds_to_update):,} odds güncellenecek")
    print()
    
    # Batch insert/update
    batch_size = 500
    
    # Yeni eklemeler
    for i in range(0, len(odds_to_add), batch_size):
        batch = odds_to_add[i:i+batch_size]
        try:
            for match_id, odds_data in batch:
                new_odds = MatchOdds(match_id=match_id, **odds_data)
                session.add(new_odds)
            session.commit()
            stats['odds_added'] += len(batch)
            print(f"   ✅ {stats['odds_added']:,}/{len(odds_to_add)} odds eklendi")
        except Exception as e:
            session.rollback()
            # Tek tek dene
            for match_id, odds_data in batch:
                try:
                    existing = session.query(MatchOdds).filter(MatchOdds.match_id == match_id).first()
                    if not existing:
                        new_odds = MatchOdds(match_id=match_id, **odds_data)
                        session.add(new_odds)
                        session.commit()
                        stats['odds_added'] += 1
                    else:
                        stats['already_exists'] += 1
                except:
                    session.rollback()
                    continue
    
    # Güncellemeler
    for i in range(0, len(odds_to_update), batch_size):
        batch = odds_to_update[i:i+batch_size]
        try:
            for match_id, odds_data in batch:
                existing = session.query(MatchOdds).filter(MatchOdds.match_id == match_id).first()
                if existing:
                    updated = False
                    for key, value in odds_data.items():
                        if value is not None and getattr(existing, key, None) is None:
                            setattr(existing, key, value)
                            updated = True
                    if updated:
                        existing.updated_at = datetime.utcnow()
                        stats['odds_updated'] += 1
            session.commit()
            if (i + batch_size) % 1000 == 0:
                print(f"   ✅ {min(i + batch_size, len(odds_to_update)):,}/{len(odds_to_update)} odds güncellendi")
        except Exception as e:
            session.rollback()
            continue
    
    print()
    print("=" * 80)
    print("📊 ÖZET")
    print("=" * 80)
    print(f"   📄 Toplam satır: {stats['total_rows']:,}")
    print(f"   ✅ Maç bulundu: {stats['matches_found']:,}")
    print(f"   ➕ Odds eklendi: {stats['odds_added']:,}")
    print(f"   🔄 Odds güncellendi: {stats['odds_updated']:,}")
    print(f"   ⏭️ Zaten var: {stats['already_exists']:,}")
    print(f"   ❌ Eşleşmeyen: {stats['unmatched']:,}")
    print(f"   ⚠️ Hatalar: {stats['errors']:,}")
    print("=" * 80)
    
    session.close()

if __name__ == "__main__":
    load_la_liga_odds_super_fast()





