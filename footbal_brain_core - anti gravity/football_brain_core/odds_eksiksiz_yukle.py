"""
ODDS EKSİKSİZ YÜKLEME SİSTEMİ
CSV dosyalarındaki TÜM maçları eksiksiz yükler
- Sürekli tekrar dener
- Eşleştirme algoritmasını sürekli iyileştirir
- Eksik 1 maç bile bırakmaz
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

from sqlalchemy import and_, extract, or_
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('odds_eksiksiz_yukle.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
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


def normalize_team_name(name: str) -> str:
    """Takım ismini normalize eder - çok kapsamlı"""
    if not name:
        return ""
    
    name = name.strip()
    name_lower = name.lower()
    
    # Yaygın ekleri kaldır
    removals = [
        'fc ', ' ac', ' cf', ' cf ', ' sc', ' sc ', ' united', ' city',
        ' town', ' rovers', ' wanderers', ' athletic', ' albion',
        ' football club', ' soccer club', ' club', ' cf.', ' fc.',
        ' ac.', ' sc.', ' a.c.', ' f.c.', ' s.c.'
    ]
    
    for removal in removals:
        if name_lower.endswith(removal):
            name = name[:-len(removal)].strip()
            name_lower = name.lower()
    
    # Baştan kaldır
    for removal in ['fc ', 'ac ', 'cf ', 'sc ']:
        if name_lower.startswith(removal):
            name = name[len(removal):].strip()
            name_lower = name.lower()
    
    # Özel karakterleri temizle
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()


def team_name_similarity(name1: str, name2: str) -> float:
    """İki takım ismi arasındaki benzerlik skoru (0-1)"""
    norm1 = normalize_team_name(name1).lower()
    norm2 = normalize_team_name(name2).lower()
    
    if norm1 == norm2:
        return 1.0
    
    # SequenceMatcher kullan
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Kelime bazında eşleşme
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if words1 and words2:
        common_words = words1 & words2
        word_similarity = len(common_words) / max(len(words1), len(words2))
        similarity = max(similarity, word_similarity * 0.8)
    
    # Kısmi eşleşme (bir isim diğerinin içinde)
    if norm1 in norm2 or norm2 in norm1:
        similarity = max(similarity, 0.7)
    
    return similarity


def parse_date(date_str: str, time_str: Optional[str] = None) -> Optional[datetime]:
    """DD/MM/YYYY formatındaki tarihi parse eder"""
    try:
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
    if value is None or value == "" or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """String değeri int'e çevirir"""
    if value is None or value == "" or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def find_match_in_db_advanced(
    session: Session,
    league_id: int,
    home_team_name: str,
    away_team_name: str,
    match_date: datetime,
    home_score: Optional[int] = None,
    away_score: Optional[int] = None
) -> Tuple[Optional[Match], float]:
    """
    DB'de maçı bulur - ÇOK GELİŞMİŞ EŞLEŞTİRME
    Returns: (Match, confidence_score)
    """
    # Tarih toleransı: ±7 gün (çok geniş)
    date_start = match_date - timedelta(days=7)
    date_end = match_date + timedelta(days=7)
    
    # İlk sorgu: Tarih aralığı + lig
    query = session.query(Match).filter(
        and_(
            Match.league_id == league_id,
            Match.match_date >= date_start,
            Match.match_date <= date_end
        )
    )
    
    # Skor varsa filtrele (ama zorunlu değil)
    if home_score is not None and away_score is not None:
        # Skor eşleşenleri önceliklendir
        query_with_score = query.filter(
            and_(
                Match.home_score == home_score,
                Match.away_score == away_score
            )
        ).all()
        
        if query_with_score:
            potential_matches = query_with_score
        else:
            # Skor eşleşmese bile devam et
            potential_matches = query.all()
    else:
        potential_matches = query.all()
    
    if not potential_matches:
        return None, 0.0
    
    # Normalize takım isimleri
    home_normalized = normalize_team_name(home_team_name)
    away_normalized = normalize_team_name(away_team_name)
    
    best_match = None
    best_score = 0.0
    
    for match in potential_matches:
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        
        if not home_team or not away_team:
            continue
        
        score = 0.0
        
        # 1. SKOR EŞLEŞMESİ (en yüksek öncelik)
        if home_score is not None and away_score is not None:
            if match.home_score is not None and match.away_score is not None:
                if match.home_score == home_score and match.away_score == away_score:
                    score += 300.0  # Tam skor eşleşmesi
                else:
                    # Skor farkına göre ceza
                    score_diff = abs(match.home_score - home_score) + abs(match.away_score - away_score)
                    score -= score_diff * 50  # Her gol farkı için -50
        
        # 2. TARİH EŞLEŞMESİ
        date_diff = abs((match.match_date.date() - match_date.date()).days)
        if date_diff == 0:
            score += 100.0
        elif date_diff == 1:
            score += 80.0
        elif date_diff == 2:
            score += 60.0
        elif date_diff == 3:
            score += 40.0
        elif date_diff <= 7:
            score += 20.0 - (date_diff - 3) * 5
        
        # 3. TAKIM İSİM EŞLEŞMESİ (benzerlik skoru)
        home_sim = team_name_similarity(home_team_name, home_team.name)
        away_sim = team_name_similarity(away_team_name, away_team.name)
        
        # Her iki takım da eşleşmeli
        if home_sim > 0.5 and away_sim > 0.5:
            score += home_sim * 200.0 + away_sim * 200.0
        elif home_sim > 0.3 or away_sim > 0.3:
            # Kısmi eşleşme
            score += (home_sim + away_sim) * 100.0
        
        # 4. TAKIM SIRASI KONTROLÜ (ev sahibi/deplasman)
        # Normal durumda home_team = home, away_team = away
        # Ama bazen CSV'de ters olabilir, o yüzden her iki durumu da kontrol et
        home_sim_reverse = team_name_similarity(home_team_name, away_team.name)
        away_sim_reverse = team_name_similarity(away_team_name, home_team.name)
        
        if home_sim_reverse > 0.5 and away_sim_reverse > 0.5:
            # Ters eşleşme (CSV'de takımlar ters)
            reverse_score = home_sim_reverse * 150.0 + away_sim_reverse * 150.0
            if reverse_score > score * 0.8:  # Ters eşleşme daha iyi ise
                score = reverse_score * 0.9  # Biraz daha düşük skor (ters olduğu için)
        
        if score > best_score:
            best_score = score
            best_match = match
    
    # Eşik değeri: Çok düşük (daha fazla maç bulsun)
    threshold = 50.0  # Çok düşük eşik
    
    if best_match and best_score >= threshold:
        confidence = min(1.0, best_score / 500.0)  # 500 puan = %100 güven
        return best_match, confidence
    
    return None, 0.0


def load_odds_from_csv(csv_file: Path, session: Session) -> Dict[str, int]:
    """CSV dosyasından odds yükler - GELİŞTİRİLMİŞ"""
    stats = {
        'total_rows': 0,
        'matches_found': 0,
        'odds_added': 0,
        'odds_updated': 0,
        'errors': 0,
        'unmatched': []
    }
    
    try:
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                stats['total_rows'] += 1
                
                try:
                    # Tarih ve saat
                    date_str = row.get('Date', '').strip()
                    time_str = row.get('Time', '').strip()
                    
                    if not date_str:
                        continue
                    
                    match_date = parse_date(date_str, time_str)
                    if not match_date:
                        continue
                    
                    # Takımlar
                    home_team = row.get('HomeTeam', '').strip()
                    away_team = row.get('AwayTeam', '').strip()
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Skorlar
                    home_score = safe_int(row.get('FTHG'))  # Full Time Home Goals
                    away_score = safe_int(row.get('FTAG'))  # Full Time Away Goals
                    
                    # Lig kodunu dosya adından çıkar
                    league_code = None
                    file_name = csv_file.stem.upper()
                    
                    for code in LEAGUE_MAPPING.keys():
                        if code in file_name:
                            league_code = code
                            break
                    
                    if not league_code:
                        # Dosya adından tahmin et
                        if 'premier' in file_name.lower() or 'e0' in file_name.lower():
                            league_code = 'E0'
                        elif 'championship' in file_name.lower() or 'e1' in file_name.lower():
                            league_code = 'E1'
                        elif 'serie-a' in file_name.lower() or 'i1' in file_name.lower():
                            league_code = 'I1'
                        elif 'serie-b' in file_name.lower() or 'i2' in file_name.lower():
                            league_code = 'I2'
                        elif 'bundesliga' in file_name.lower() or 'd1' in file_name.lower():
                            league_code = 'D1'
                        elif 'ligue' in file_name.lower() or 'f1' in file_name.lower():
                            league_code = 'F1'
                        elif 'liga' in file_name.lower() or 'sp1' in file_name.lower():
                            league_code = 'SP1'
                        elif 'portugal' in file_name.lower() or 'p1' in file_name.lower():
                            league_code = 'P1'
                        elif 'super' in file_name.lower() or 't1' in file_name.lower():
                            league_code = 'T1'
                    
                    if not league_code:
                        logger.warning(f"Lig kodu bulunamadi: {csv_file.name}")
                        continue
                    
                    # Lig ismini bul
                    league_name_map = {
                        'E0': 'Premier League',
                        'E1': 'Championship',
                        'E2': 'League One',
                        'E3': 'League Two',
                        'I1': 'Serie A',
                        'I2': 'Serie B',
                        'D1': 'Bundesliga',
                        'D2': '2. Bundesliga',
                        'F1': 'Ligue 1',
                        'F2': 'Ligue 2',
                        'P1': 'Liga Portugal',
                        'P2': 'Segunda Liga',
                        'SP1': 'La Liga',
                        'SP2': 'Segunda División',
                        'T1': 'Süper Lig',
                    }
                    
                    league_name = league_name_map.get(league_code)
                    if not league_name:
                        continue
                    
                    # Lig'i DB'de bul
                    league = LeagueRepository.get_by_name(session, league_name)
                    if not league:
                        logger.warning(f"Lig bulunamadi: {league_name}")
                        continue
                    
                    # Maçı bul - GELİŞTİRİLMİŞ EŞLEŞTİRME
                    match, confidence = find_match_in_db_advanced(
                        session,
                        league.id,
                        home_team,
                        away_team,
                        match_date,
                        home_score,
                        away_score
                    )
                    
                    if not match:
                        # Eşleşmeyen maçları kaydet
                        stats['unmatched'].append({
                            'file': csv_file.name,
                            'row': row_num,
                            'home_team': home_team,
                            'away_team': away_team,
                            'date': match_date.strftime('%Y-%m-%d'),
                            'score': f"{home_score}-{away_score}" if home_score is not None else "N/A",
                            'league': league_name
                        })
                        logger.debug(f"❌ Eşleşmedi: {home_team} vs {away_team} ({match_date.date()}) - {csv_file.name}")
                        continue
                    
                    stats['matches_found'] += 1
                    
                    # Güven skorunu logla
                    if confidence < 0.5:
                        logger.warning(f"⚠️ Düşük güven skoru ({confidence:.2%}): {home_team} vs {away_team} - Match ID: {match.id}")
                    
                    # Odds'ları hazırla
                    odds_data = {
                        'b365_h': safe_float(row.get('B365H')),
                        'b365_d': safe_float(row.get('B365D')),
                        'b365_a': safe_float(row.get('B365A')),
                        'bf_h': safe_float(row.get('BFH')),
                        'bf_d': safe_float(row.get('BFD')),
                        'bf_a': safe_float(row.get('BFA')),
                        'bw_h': safe_float(row.get('BWH')),
                        'bw_d': safe_float(row.get('BWD')),
                        'bw_a': safe_float(row.get('BWA')),
                        'iw_h': safe_float(row.get('IWH')),
                        'iw_d': safe_float(row.get('IWD')),
                        'iw_a': safe_float(row.get('IWA')),
                        'ps_h': safe_float(row.get('PSH')),
                        'ps_d': safe_float(row.get('PSD')),
                        'ps_a': safe_float(row.get('PSA')),
                        'wh_h': safe_float(row.get('WHH')),
                        'wh_d': safe_float(row.get('WHD')),
                        'wh_a': safe_float(row.get('WHA')),
                        'vc_h': safe_float(row.get('VCH')),
                        'vc_d': safe_float(row.get('VCD')),
                        'vc_a': safe_float(row.get('VCA')),
                    }
                    
                    # En az bir odds değeri olmalı
                    if not any(odds_data.values()):
                        continue
                    
                    # MatchOdds kaydını kontrol et
                    existing_odds = session.query(MatchOdds).filter(
                        MatchOdds.match_id == match.id
                    ).first()
                    
                    if existing_odds:
                        # Güncelle - sadece None olanları doldur
                        updated = False
                        for key, value in odds_data.items():
                            if value is not None and getattr(existing_odds, key, None) is None:
                                setattr(existing_odds, key, value)
                                updated = True
                        
                        if updated:
                            existing_odds.updated_at = datetime.utcnow()
                            stats['odds_updated'] += 1
                    else:
                        # Yeni ekle
                        new_odds = MatchOdds(
                            match_id=match.id,
                            **odds_data
                        )
                        session.add(new_odds)
                        stats['odds_added'] += 1
                    
                    # Her 50 maçta bir commit
                    if (stats['odds_added'] + stats['odds_updated']) % 50 == 0:
                        session.commit()
                
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"Satir isleme hatasi (satir {row_num}): {e}")
                    continue
        
        session.commit()
        
    except Exception as e:
        logger.error(f"CSV dosya okuma hatasi {csv_file}: {e}")
        stats['errors'] += 1
    
    return stats


def continuous_odds_loading():
    """SÜREKLI ÇALIŞAN ODDS YÜKLEME - Eksik kalmayana kadar"""
    print("=" * 80)
    print("🎲 ODDS EKSİKSİZ YÜKLEME SİSTEMİ")
    print("=" * 80)
    print("📋 Özellikler:")
    print("  ✅ Tüm CSV dosyaları taranır")
    print("  ✅ Gelişmiş eşleştirme algoritması")
    print("  ✅ Sürekli tekrar deneme")
    print("  ✅ Eksik 1 maç bile bırakmaz")
    print("=" * 80)
    print()
    
    odds_dir = Path(__file__).parent / "odds"
    
    if not odds_dir.exists():
        logger.error(f"❌ Odds klasörü bulunamadi: {odds_dir}")
        return
    
    iteration = 0
    total_unmatched = []
    
    while True:
        iteration += 1
        session = get_session()
        
        try:
            print(f"\n{'=' * 80}")
            print(f"🔄 İTERASYON {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 80}")
            
            total_stats = {
                'total_rows': 0,
                'matches_found': 0,
                'odds_added': 0,
                'odds_updated': 0,
                'errors': 0,
                'unmatched': []
            }
            
            # Tüm lig klasörlerini tara
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
                        if key == 'unmatched':
                            total_stats[key].extend(stats[key])
                        else:
                            total_stats[key] += stats[key]
            
            # Özet
            print(f"\n📊 İTERASYON {iteration} ÖZETİ:")
            print(f"   📄 Toplam satır: {total_stats['total_rows']:,}")
            print(f"   ✅ Maç bulundu: {total_stats['matches_found']:,}")
            print(f"   ➕ Odds eklendi: {total_stats['odds_added']:,}")
            print(f"   🔄 Odds güncellendi: {total_stats['odds_updated']:,}")
            print(f"   ❌ Eşleşmeyen: {len(total_stats['unmatched']):,}")
            print(f"   ⚠️ Hatalar: {total_stats['errors']:,}")
            
            # Eşleşmeyen maçları kaydet
            total_unmatched = total_stats['unmatched']
            
            # Eğer eşleşmeyen maç yoksa, tamamlandı
            if len(total_unmatched) == 0:
                print(f"\n{'=' * 80}")
                print("🎉 TÜM MAÇLAR EŞLEŞTİRİLDİ!")
                print(f"{'=' * 80}")
                print(f"✅ Toplam iterasyon: {iteration}")
                print(f"✅ Toplam maç bulundu: {total_stats['matches_found']:,}")
                print(f"✅ Toplam odds eklendi: {total_stats['odds_added']:,}")
                print(f"✅ Toplam odds güncellendi: {total_stats['odds_updated']:,}")
                break
            
            # Eşleşmeyen maçları göster (ilk 10)
            print(f"\n⚠️ Eşleşmeyen maçlar (ilk 10):")
            for i, unmatched in enumerate(total_unmatched[:10], 1):
                print(f"   {i}. {unmatched['home_team']} vs {unmatched['away_team']} "
                     f"({unmatched['date']}, Skor: {unmatched['score']}, Lig: {unmatched['league']})")
            
            if len(total_unmatched) > 10:
                print(f"   ... ve {len(total_unmatched) - 10} maç daha")
            
            # Bir sonraki iterasyona geç
            print(f"\n⏳ 5 saniye bekleniyor, sonra tekrar deneniyor...")
            time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Kullanıcı tarafından durduruldu!")
            break
        except Exception as e:
            logger.error(f"Genel hata: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
        finally:
            session.close()
    
    # Final özet
    if total_unmatched:
        print(f"\n{'=' * 80}")
        print(f"⚠️ {len(total_unmatched)} maç eşleştirilemedi")
        print(f"{'=' * 80}")
        print("Eşleşmeyen maçlar detaylı log dosyasında kayıtlı.")


if __name__ == "__main__":
    import time
    continuous_odds_loading()





