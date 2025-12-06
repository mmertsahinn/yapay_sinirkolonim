"""
LA LIGA ODDS YÜKLEME - SKOR VE TAKIM EŞLEŞMESİ ÖNCELİKLİ
Tarih toleransı çok geniş, skor ve takım eşleşmesi öncelikli
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import csv
from typing import Optional, Set, Dict, Tuple
import re
from difflib import SequenceMatcher
from collections import defaultdict

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
    # Özel karakterleri temizle ama boşlukları koru
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def team_name_similarity(name1: str, name2: str) -> float:
    """İki takım ismi arasındaki benzerlik skoru"""
    norm1 = normalize_team_name(name1).lower()
    norm2 = normalize_team_name(name2).lower()
    if norm1 == norm2:
        return 1.0
    if norm1 in norm2 or norm2 in norm1:
        return 0.8
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

def find_match_by_score_and_teams(
    session: Session,
    league_id: int,
    home_team_name: str,
    away_team_name: str,
    home_score: Optional[int],
    away_score: Optional[int],
    match_date: datetime
) -> Optional[Match]:
    """
    SKOR VE TAKIM EŞLEŞMESİ ÖNCELİKLİ
    - Önce skor + takım eşleşmesi ara (tarih çok geniş)
    - Sonra sadece takım eşleşmesi (skor yoksa)
    """
    
    # Çok geniş tarih aralığı: ±365 gün (1 yıl)
    date_start = match_date - timedelta(days=365)
    date_end = match_date + timedelta(days=365)
    
    # La Liga maçlarını al
    potential_matches = session.query(Match).filter(
        and_(
            Match.league_id == league_id,
            Match.match_date >= date_start,
            Match.match_date <= date_end,
            Match.home_score.isnot(None),
            Match.away_score.isnot(None)
        )
    ).all()
    
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
            
            # 1. ÖNCELİK: SKOR EŞLEŞMESİ (en yüksek puan)
            if home_score is not None and away_score is not None:
                if match.home_score == home_score and match.away_score == away_score:
                    score += 1000.0  # Tam skor eşleşmesi - ÇOK YÜKSEK PUAN
                else:
                    # Skor uyuşmuyorsa bu maçı atla (skor varsa kesin olmalı)
                    continue
            
            # 2. TAKIM EŞLEŞMESİ (normal sıra)
            home_sim = team_name_similarity(home_team_name, home_team.name)
            away_sim = team_name_similarity(away_team_name, away_team.name)
            
            if home_sim > 0.5 and away_sim > 0.5:
                score += home_sim * 500.0 + away_sim * 500.0
            else:
                # Ters eşleşme kontrolü (home-away ters olabilir)
                reverse_home_sim = team_name_similarity(home_team_name, away_team.name)
                reverse_away_sim = team_name_similarity(away_team_name, home_team.name)
                
                if reverse_home_sim > 0.5 and reverse_away_sim > 0.5:
                    # Ters eşleşme - skorları da ters kontrol et
                    if home_score is not None and away_score is not None:
                        if match.home_score == away_score and match.away_score == home_score:
                            score += 1000.0  # Ters skor eşleşmesi
                    score += reverse_home_sim * 400.0 + reverse_away_sim * 400.0
                else:
                    # Yeterli benzerlik yok
                    continue
            
            # 3. TARİH YAKINLIĞI (bonus)
            date_diff = abs((match.match_date.date() - match_date.date()).days)
            if date_diff == 0:
                score += 100.0
            elif date_diff <= 7:
                score += 50.0
            elif date_diff <= 30:
                score += 20.0
            elif date_diff <= 90:
                score += 10.0
            
            if score > best_score:
                best_score = score
                best_match = match
        
        except Exception as e:
            continue
    
    # Eşik: Skor eşleşmesi varsa çok yüksek puan alır, yoksa takım eşleşmesi yeterli
    if best_match and best_score >= 500.0:
        return best_match
    
    return None

def load_la_liga_odds_by_score():
    """La Liga için skor öncelikli odds yükleme"""
    print("=" * 80)
    print("LA LIGA ODDS YÜKLEME - SKOR VE TAKIM EŞLEŞMESİ ÖNCELİKLİ")
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
    
    # Mevcut odds'ları cache'le
    print("📦 Mevcut odds'lar cache'leniyor...")
    existing_odds_match_ids = set(
        row[0] for row in session.query(MatchOdds.match_id).all()
    )
    print(f"   ✅ {len(existing_odds_match_ids):,} mevcut odds cache'lendi")
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
        'no_score': 0,
        'score_mismatch': 0
    }
    
    # Toplanacak odds'lar
    odds_to_add = []
    odds_to_update = []
    
    for file_idx, csv_file in enumerate(sp1_files, 1):
        print(f"📄 [{file_idx}/{len(sp1_files)}] {csv_file.name}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader, 1):
                    stats['total_rows'] += 1
                    
                    if stats['total_rows'] % 500 == 0:
                        print(f"   ⏳ {stats['total_rows']:,} satır | {stats['matches_found']:,} maç | {stats['odds_added']:,} eklenecek | {stats['unmatched']:,} eşleşmedi")
                    
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
                        
                        # SKOR YOKSA ATLA (skor öncelikli eşleştirme için gerekli)
                        if home_score is None or away_score is None:
                            stats['no_score'] += 1
                            continue
                        
                        # Skor ve takım eşleşmesi ile maçı bul
                        match = find_match_by_score_and_teams(
                            session, la_liga.id, home_team, away_team, 
                            home_score, away_score, match_date
                        )
                        
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
                            odds_to_update.append((match.id, odds_data))
                        else:
                            odds_to_add.append((match.id, odds_data))
                            existing_odds_match_ids.add(match.id)
                    
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
    
    # Batch insert
    batch_size = 500
    for i in range(0, len(odds_to_add), batch_size):
        batch = odds_to_add[i:i+batch_size]
        try:
            for match_id, odds_data in batch:
                existing = session.query(MatchOdds).filter(MatchOdds.match_id == match_id).first()
                if not existing:
                    new_odds = MatchOdds(match_id=match_id, **odds_data)
                    session.add(new_odds)
            session.commit()
            stats['odds_added'] += len(batch)
            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(odds_to_add):
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
                except:
                    session.rollback()
                    continue
    
    # Batch update
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
            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(odds_to_update):
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
    print(f"   ❌ Eşleşmeyen: {stats['unmatched']:,}")
    print(f"   ⚠️ Skor yok: {stats['no_score']:,}")
    print(f"   ⚠️ Hatalar: {stats['errors']:,}")
    print("=" * 80)
    
    session.close()

if __name__ == "__main__":
    load_la_liga_odds_by_score()





