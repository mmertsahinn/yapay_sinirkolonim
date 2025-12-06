"""
LA LIGA ÖZEL ODDS YÜKLEME
- Takım isimlerini karşılaştırır
- Eşleşmeyen takımları tespit eder
- Neden yüklenemediğini analiz eder
- Özel eşleştirme ile yükler
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import csv
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple, Set
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

from sqlalchemy import and_, extract, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('la_liga_odds_ozel.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def normalize_team_name(name: str) -> str:
    """Takım ismini normalize eder - ÇOK KAPSAMLI"""
    if not name:
        return ""
    
    name = name.strip()
    name_lower = name.lower()
    
    # Yaygın ekleri kaldır
    removals = [
        'fc ', ' ac', ' cf', ' cf ', ' sc', ' sc ', ' united', ' city',
        ' town', ' rovers', ' wanderers', ' athletic', ' albion',
        ' football club', ' soccer club', ' club', ' cf.', ' fc.',
        ' ac.', ' sc.', ' a.c.', ' f.c.', ' s.c.', ' c.f.', ' c.f.',
        ' real ', ' atletico ', ' atlético ', ' deportivo ', ' c.d. ',
        ' c.d.', ' cd ', ' ud ', ' sd ', ' rc ', ' cf ', ' gd ', ' sc ',
        ' sporting ', ' club ', ' de ', ' la ', ' el ', ' los ', ' las ',
    ]
    
    for removal in removals:
        if name_lower.startswith(removal):
            name = name[len(removal):].strip()
            name_lower = name.lower()
        if name_lower.endswith(removal):
            name = name[:-len(removal)].strip()
            name_lower = name.lower()
    
    # Özel karakterleri temizle
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()


def team_name_similarity(name1: str, name2: str) -> float:
    """İki takım ismi arasındaki benzerlik skoru"""
    norm1 = normalize_team_name(name1).lower()
    norm2 = normalize_team_name(name2).lower()
    
    if norm1 == norm2:
        return 1.0
    
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if words1 and words2:
        common_words = words1 & words2
        word_similarity = len(common_words) / max(len(words1), len(words2))
        similarity = max(similarity, word_similarity * 0.8)
    
    if norm1 in norm2 or norm2 in norm1:
        similarity = max(similarity, 0.7)
    
    return similarity


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
    except Exception as e:
        logger.debug(f"Date parse hatası: {date_str}, {e}")
    return None


def safe_float(value: Any) -> Optional[float]:
    """String değeri float'a çevirir"""
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """String değeri int'e çevirir"""
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return int(float(value))
    except (ValueError, TypeError):
        return None


def get_db_teams_for_la_liga(session: Session) -> Dict[str, Team]:
    """La Liga'daki tüm takımları DB'den al"""
    la_liga = LeagueRepository.get_by_name(session, "La Liga")
    if not la_liga:
        return {}
    
    teams = TeamRepository.get_by_league(session, la_liga.id)
    team_dict = {}
    
    for team in teams:
        # Normalize edilmiş isimle kaydet
        normalized = normalize_team_name(team.name)
        team_dict[normalized.lower()] = team
        # Orijinal isimle de kaydet
        team_dict[team.name.lower()] = team
    
    return team_dict


def get_csv_teams_from_la_liga_files() -> Set[str]:
    """La Liga CSV dosyalarındaki tüm takım isimlerini topla"""
    csv_teams = set()
    odds_dir = Path(__file__).parent / "odds" / "espana"
    
    if not odds_dir.exists():
        return csv_teams
    
    # SP1 dosyalarını bul (La Liga)
    sp1_files = list(odds_dir.glob("SP1*.csv"))
    
    for csv_file in sp1_files:
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    home_team = row.get('HomeTeam', '').strip()
                    away_team = row.get('AwayTeam', '').strip()
                    if home_team:
                        csv_teams.add(home_team)
                    if away_team:
                        csv_teams.add(away_team)
        except Exception as e:
            logger.error(f"CSV okuma hatası {csv_file}: {e}")
    
    return csv_teams


def analyze_team_mismatches():
    """Takım isimlerini karşılaştır ve eşleşmeyenleri bul"""
    session = get_session()
    
    try:
        print("=" * 80)
        print("LA LIGA TAKIM İSİM KARŞILAŞTIRMASI")
        print("=" * 80)
        print()
        
        # DB'deki takımlar
        db_teams = get_db_teams_for_la_liga(session)
        db_team_names = set(db_teams.keys())
        
        print(f"📊 DB'deki takım sayısı: {len(db_teams)}")
        print("DB Takımları:")
        la_liga = LeagueRepository.get_by_name(session, "La Liga")
        if la_liga:
            all_db_teams = TeamRepository.get_by_league(session, la_liga.id)
            for team in sorted(all_db_teams, key=lambda x: x.name):
                print(f"   - {team.name}")
        print()
        
        # CSV'deki takımlar
        csv_teams = get_csv_teams_from_la_liga_files()
        print(f"📊 CSV'deki takım sayısı: {len(csv_teams)}")
        print("CSV Takımları:")
        for team in sorted(csv_teams):
            print(f"   - {team}")
        print()
        
        # Eşleşmeyen takımları bul
        print("=" * 80)
        print("EŞLEŞMEYEN TAKIMLAR")
        print("=" * 80)
        
        unmatched_csv = []
        unmatched_db = []
        
        for csv_team in csv_teams:
            csv_normalized = normalize_team_name(csv_team).lower()
            found = False
            
            # Tam eşleşme
            if csv_normalized in db_team_names:
                found = True
            else:
                # Benzerlik kontrolü
                best_match = None
                best_sim = 0.0
                
                for db_normalized, db_team in db_teams.items():
                    sim = team_name_similarity(csv_team, db_team.name)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = db_team
                
                if best_sim >= 0.6:  # %60 benzerlik
                    print(f"✅ {csv_team:<30} -> {best_match.name:<30} (Benzerlik: {best_sim:.1%})")
                    found = True
                else:
                    unmatched_csv.append((csv_team, best_match.name if best_match else "BULUNAMADI", best_sim))
        
        print()
        print("❌ EŞLEŞMEYEN CSV TAKIMLARI:")
        for csv_team, closest_db, sim in unmatched_csv:
            print(f"   {csv_team:<30} -> En yakın: {closest_db:<30} (Benzerlik: {sim:.1%})")
        
        print()
        print("=" * 80)
        
    finally:
        session.close()


def find_match_ultra_flexible(
    session: Session,
    league_id: int,
    home_team_name: str,
    away_team_name: str,
    match_date: datetime,
    home_score: Optional[int] = None,
    away_score: Optional[int] = None
) -> Tuple[Optional[Match], float, str]:
    """
    DB'de maçı bulur - ULTRA ESNEK (La Liga için özel)
    Returns: (Match, confidence, reason)
    """
    try:
        # Çok geniş tarih toleransı: ±60 gün
        date_start = match_date - timedelta(days=60)
        date_end = match_date + timedelta(days=60)
        
        # Önce lig ID ile dene
        query = session.query(Match).filter(
            and_(
                Match.league_id == league_id,
                Match.match_date >= date_start,
                Match.match_date <= date_end
            )
        )
        
        potential_matches = query.all()
        
        # Bulunamazsa tüm liglerde ara
        if not potential_matches:
            query_all = session.query(Match).filter(
                and_(
                    Match.match_date >= date_start,
                    Match.match_date <= date_end
                )
            )
            potential_matches = query_all.all()
        
        if not potential_matches:
            return None, 0.0, "Tarih aralığında maç bulunamadı"
        
        home_normalized = normalize_team_name(home_team_name)
        away_normalized = normalize_team_name(away_team_name)
        
        best_match = None
        best_score = 0.0
        best_reason = ""
        
        for match in potential_matches:
            try:
                home_team = TeamRepository.get_by_id(session, match.home_team_id)
                away_team = TeamRepository.get_by_id(session, match.away_team_id)
                
                if not home_team or not away_team:
                    continue
                
                score = 0.0
                reasons = []
                
                # Skor eşleşmesi (opsiyonel ama bonus)
                if home_score is not None and away_score is not None:
                    if match.home_score is not None and match.away_score is not None:
                        if match.home_score == home_score and match.away_score == away_score:
                            score += 400.0
                            reasons.append("Tam skor eşleşmesi")
                        else:
                            score_diff = abs(match.home_score - home_score) + abs(match.away_score - away_score)
                            score -= score_diff * 30
                
                # Tarih eşleşmesi
                date_diff = abs((match.match_date.date() - match_date.date()).days)
                if date_diff == 0:
                    score += 200.0
                    reasons.append("Aynı gün")
                elif date_diff <= 7:
                    score += 150.0 - date_diff * 10
                    reasons.append(f"{date_diff} gün fark")
                elif date_diff <= 30:
                    score += 80.0 - (date_diff - 7) * 2
                    reasons.append(f"{date_diff} gün fark")
                elif date_diff <= 60:
                    score += 20.0 - (date_diff - 30)
                    reasons.append(f"{date_diff} gün fark")
                
                # Takım ismi eşleşmesi - ÇOK ESNEK
                home_sim = team_name_similarity(home_team_name, home_team.name)
                away_sim = team_name_similarity(away_team_name, away_team.name)
                
                # Normal eşleşme
                if home_sim > 0.6 and away_sim > 0.6:
                    score += home_sim * 300.0 + away_sim * 300.0
                    reasons.append(f"Takım eşleşmesi ({home_sim:.1%}, {away_sim:.1%})")
                elif home_sim > 0.4 and away_sim > 0.4:
                    score += home_sim * 200.0 + away_sim * 200.0
                    reasons.append(f"Kısmi takım eşleşmesi ({home_sim:.1%}, {away_sim:.1%})")
                elif home_sim > 0.2 or away_sim > 0.2:
                    score += (home_sim + away_sim) * 100.0
                    reasons.append(f"Düşük takım eşleşmesi ({home_sim:.1%}, {away_sim:.1%})")
                
                # Ters eşleşme kontrolü
                home_sim_reverse = team_name_similarity(home_team_name, away_team.name)
                away_sim_reverse = team_name_similarity(away_team_name, home_team.name)
                
                if home_sim_reverse > 0.5 and away_sim_reverse > 0.5:
                    reverse_score = home_sim_reverse * 250.0 + away_sim_reverse * 250.0
                    if reverse_score > score * 0.7:
                        score = reverse_score * 0.85
                        reasons.append("Ters takım eşleşmesi")
                
                # Lig eşleşmesi bonusu
                if match.league_id == league_id:
                    score += 100.0
                    reasons.append("Lig eşleşmesi")
                
                if score > best_score:
                    best_score = score
                    best_match = match
                    best_reason = " + ".join(reasons)
            
            except Exception as e:
                logger.debug(f"Match scoring hatası: {e}")
                continue
        
        # Çok düşük eşik - daha fazla maç bulsun
        threshold = 20.0
        
        if best_match and best_score >= threshold:
            confidence = min(1.0, best_score / 600.0)
            return best_match, confidence, best_reason
        
        return None, 0.0, f"Eşik altı (en iyi skor: {best_score:.1f})"
    
    except Exception as e:
        logger.error(f"find_match_ultra_flexible hatası: {e}")
        return None, 0.0, f"Hata: {str(e)}"


def load_la_liga_odds_special():
    """La Liga için özel odds yükleme"""
    print("=" * 80)
    print("LA LIGA ÖZEL ODDS YÜKLEME")
    print("=" * 80)
    print()
    
    # Takım karşılaştırmasını atla - direkt yükleme
    # analyze_team_mismatches()  # Çok yavaş, atlanıyor
    
    # Odds yükleme
    print("ODDS YÜKLEME BAŞLIYOR...")
    print("=" * 80)
    print()
    
    session = get_session()
    odds_dir = Path(__file__).parent / "odds" / "espana"
    
    if not odds_dir.exists():
        logger.error(f"❌ Odds klasörü bulunamadi: {odds_dir}")
        return
    
    la_liga = LeagueRepository.get_by_name(session, "La Liga")
    if not la_liga:
        logger.error("❌ La Liga bulunamadi!")
        return
    
    print(f"✅ La Liga bulundu (ID: {la_liga.id})")
    print()
    
    # SP1 dosyalarını bul
    sp1_files = list(odds_dir.glob("SP1*.csv"))
    print(f"📄 {len(sp1_files)} SP1 (La Liga) CSV dosyası bulundu")
    print()
    
    stats = {
        'total_rows': 0,
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0,
        'unmatched': [],
        'team_mismatches': defaultdict(int)
    }
    
    for file_idx, csv_file in enumerate(sp1_files, 1):
        print(f"📄 [{file_idx}/{len(sp1_files)}] İşleniyor: {csv_file.name}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                rows = list(reader)  # Tüm satırları önce oku
                total_file_rows = len(rows)
                
                for row_num, row in enumerate(rows, 1):
                    stats['total_rows'] += 1
                    
                    # Progress göster (her 100 satırda bir)
                    if row_num % 100 == 0 or row_num == total_file_rows:
                        print(f"   ⏳ {row_num}/{total_file_rows} satır işlendi... (Toplam: {stats['matches_found']} maç bulundu, {stats['odds_added']} eklendi, {stats['odds_updated']} güncellendi)")
                    
                    try:
                        # Tarih
                        date_str = row.get('Date', '').strip()
                        time_str = row.get('Time', '').strip()
                        
                        if not date_str:
                            continue
                        
                        match_date = parse_date_safe(date_str, time_str)
                        if not match_date:
                            continue
                        
                        # Takımlar
                        home_team = row.get('HomeTeam', '').strip()
                        away_team = row.get('AwayTeam', '').strip()
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Skorlar
                        home_score = safe_int(row.get('FTHG'))
                        away_score = safe_int(row.get('FTAG'))
                        
                        # Maçı bul - ULTRA ESNEK
                        match, confidence, reason = find_match_ultra_flexible(
                            session, la_liga.id, home_team, away_team, match_date, home_score, away_score
                        )
                        
                        if not match:
                            stats['unmatched'].append({
                                'file': csv_file.name,
                                'row': row_num,
                                'home_team': home_team,
                                'away_team': away_team,
                                'date': match_date.strftime('%Y-%m-%d'),
                                'score': f"{home_score}-{away_score}" if home_score else "N/A",
                                'reason': reason
                            })
                            stats['team_mismatches'][f"{home_team} vs {away_team}"] += 1
                            continue
                        
                        stats['matches_found'] += 1
                        
                        # Düşük güven skorlu eşleşmeleri logla
                        if confidence < 0.3:
                            logger.warning(f"⚠️ Düşük güven ({confidence:.1%}): {home_team} vs {away_team} -> Match ID {match.id} - {reason}")
                        
                        # Odds'ları hazırla
                        odds_data = {
                            'b365_h': safe_float(row.get('B365H')),
                            'b365_d': safe_float(row.get('B365D')),
                            'b365_a': safe_float(row.get('B365A')),
                        }
                        
                        if not any(odds_data.values()):
                            continue
                        
                        # MatchOdds kaydını kontrol et - Daha güvenli
                        try:
                            existing_odds = session.query(MatchOdds).filter(
                                MatchOdds.match_id == match.id
                            ).first()
                            
                            if existing_odds:
                                updated = False
                                for key, value in odds_data.items():
                                    if value is not None and getattr(existing_odds, key, None) is None:
                                        setattr(existing_odds, key, value)
                                        updated = True
                                
                                if updated:
                                    existing_odds.updated_at = datetime.utcnow()
                                    stats['odds_updated'] += 1
                            else:
                                new_odds = MatchOdds(match_id=match.id, **odds_data)
                                session.add(new_odds)
                                stats['odds_added'] += 1
                        except Exception as e:
                            # UNIQUE constraint hatası durumunda rollback ve devam et
                            session.rollback()
                            logger.warning(f"MatchOdds ekleme/güncelleme hatası (Match ID {match.id}): {e}")
                            continue
                        
                        # Her 100 satırda bir commit (daha hızlı)
                        if stats['total_rows'] % 100 == 0:
                            try:
                                session.commit()
                            except Exception as e:
                                session.rollback()
                                logger.warning(f"Commit hatası: {e}")
                    
                    except Exception as e:
                        stats['errors'] += 1
                        logger.error(f"Satır işleme hatası {csv_file.name}:{row_num}: {e}")
                        session.rollback()  # Hata durumunda rollback
                        continue
            
            session.commit()
        
        except Exception as e:
            logger.error(f"CSV dosya okuma hatası {csv_file}: {e}")
            stats['errors'] += 1
    
    # Özet
    print()
    print("=" * 80)
    print("📊 ÖZET")
    print("=" * 80)
    print(f"   📄 Toplam satır: {stats['total_rows']:,}")
    print(f"   ✅ Maç bulundu: {stats['matches_found']:,}")
    print(f"   ➕ Odds eklendi: {stats['odds_added']:,}")
    print(f"   🔄 Odds güncellendi: {stats['odds_updated']:,}")
    print(f"   ❌ Eşleşmeyen: {len(stats['unmatched']):,}")
    print(f"   ⚠️ Hatalar: {stats['errors']:,}")
    print()
    
    # Eşleşmeyen takımları göster
    if stats['team_mismatches']:
        print("=" * 80)
        print("❌ EN ÇOK EŞLEŞMEYEN TAKIM ÇİFTLERİ (İlk 10)")
        print("=" * 80)
        sorted_mismatches = sorted(stats['team_mismatches'].items(), key=lambda x: x[1], reverse=True)
        for team_pair, count in sorted_mismatches[:10]:
            print(f"   {team_pair}: {count} kez eşleşmedi")
        print()
    
    # Eşleşmeyen maçları göster (ilk 10)
    if stats['unmatched']:
        print("=" * 80)
        print("❌ EŞLEŞMEYEN MAÇLAR (İlk 10)")
        print("=" * 80)
        for unmatched in stats['unmatched'][:10]:
            print(f"   {unmatched['home_team']} vs {unmatched['away_team']} "
                 f"({unmatched['date']}, Skor: {unmatched['score']})")
            print(f"      Sebep: {unmatched['reason']}")
        if len(stats['unmatched']) > 10:
            print(f"   ... ve {len(stats['unmatched']) - 10} maç daha")
        print()
    
    session.close()
    
    print("=" * 80)
    print("✅ LA LIGA ÖZEL ODDS YÜKLEME TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    load_la_liga_odds_special()

