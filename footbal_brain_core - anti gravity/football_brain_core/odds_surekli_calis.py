"""
ODDS SÜREKLI ÇALIŞAN SİSTEM - ASLA DURMAZ
- Her hata verdiğinde bildirim verir
- Satır işleme hatalarını yakalar ve çözer
- Sürekli kontrol eder ve düzeltir
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import csv
import logging
import traceback
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
from sqlalchemy.exc import SQLAlchemyError

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('odds_surekli.log', encoding='utf-8'),
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
    """Takım ismini normalize eder"""
    if not name:
        return ""
    
    name = name.strip()
    name_lower = name.lower()
    
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
    
    for removal in ['fc ', 'ac ', 'cf ', 'sc ']:
        if name_lower.startswith(removal):
            name = name[len(removal):].strip()
            name_lower = name.lower()
    
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
    """Tarih parse eder - GÜVENLİ VERSİYON"""
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
    """String değeri float'a çevirir - GÜVENLİ"""
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return float(value)
    except (ValueError, TypeError) as e:
        logger.debug(f"Float conversion hatası: {value}, {e}")
        return None


def safe_int(value: Any) -> Optional[int]:
    """String değeri int'e çevirir - GÜVENLİ"""
    try:
        if value is None or value == "" or str(value).strip() == "":
            return None
        return int(float(value))
    except (ValueError, TypeError) as e:
        logger.debug(f"Int conversion hatası: {value}, {e}")
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
    """DB'de maçı bulur - ÇOK GELİŞTİRİLMİŞ (30 gün tolerans)"""
    try:
        # Tarih toleransını 30 güne çıkar (çok daha esnek)
        date_start = match_date - timedelta(days=30)
        date_end = match_date + timedelta(days=30)
        
        # Önce lig ID ile dene
        query = session.query(Match).filter(
            and_(
                Match.league_id == league_id,
                Match.match_date >= date_start,
                Match.match_date <= date_end
            )
        )
        
        potential_matches = query.all()
        
        # Eğer bulunamazsa, tüm liglerde ara (lig eşleştirmesi yanlış olabilir)
        if not potential_matches:
            logger.debug(f"Lig ID {league_id} ile bulunamadı, tüm liglerde aranıyor...")
            query_all = session.query(Match).filter(
                and_(
                    Match.match_date >= date_start,
                    Match.match_date <= date_end
                )
            )
            potential_matches = query_all.all()
        
        if not potential_matches:
            return None, 0.0
        
        home_normalized = normalize_team_name(home_team_name)
        away_normalized = normalize_team_name(away_team_name)
        
        best_match = None
        best_score = 0.0
        
        for match in potential_matches:
            try:
                home_team = TeamRepository.get_by_id(session, match.home_team_id)
                away_team = TeamRepository.get_by_id(session, match.away_team_id)
                
                if not home_team or not away_team:
                    continue
                
                score = 0.0
                
                if home_score is not None and away_score is not None:
                    if match.home_score is not None and match.away_score is not None:
                        if match.home_score == home_score and match.away_score == away_score:
                            score += 300.0
                        else:
                            score_diff = abs(match.home_score - home_score) + abs(match.away_score - away_score)
                            score -= score_diff * 50
                
                # Tarih eşleşmesi - 30 gün tolerans
                date_diff = abs((match.match_date.date() - match_date.date()).days)
                if date_diff == 0:
                    score += 150.0  # Artırıldı
                elif date_diff == 1:
                    score += 120.0
                elif date_diff <= 7:
                    score += 100.0 - (date_diff - 1) * 10
                elif date_diff <= 14:
                    score += 50.0 - (date_diff - 7) * 5
                elif date_diff <= 30:
                    score += 20.0 - (date_diff - 14) * 1
                
                home_sim = team_name_similarity(home_team_name, home_team.name)
                away_sim = team_name_similarity(away_team_name, away_team.name)
                
                # Takım ismi eşleşmesi - daha esnek
                if home_sim > 0.7 and away_sim > 0.7:
                    score += home_sim * 250.0 + away_sim * 250.0  # Artırıldı
                elif home_sim > 0.5 and away_sim > 0.5:
                    score += home_sim * 200.0 + away_sim * 200.0
                elif home_sim > 0.3 or away_sim > 0.3:
                    score += (home_sim + away_sim) * 150.0  # Artırıldı
                elif home_sim > 0.2 or away_sim > 0.2:
                    score += (home_sim + away_sim) * 50.0  # Yeni: çok düşük eşik
                
                # Ters eşleşme kontrolü (CSV'de takımlar ters olabilir)
                home_sim_reverse = team_name_similarity(home_team_name, away_team.name)
                away_sim_reverse = team_name_similarity(away_team_name, home_team.name)
                
                if home_sim_reverse > 0.5 and away_sim_reverse > 0.5:
                    reverse_score = home_sim_reverse * 200.0 + away_sim_reverse * 200.0
                    if reverse_score > score * 0.8:
                        score = reverse_score * 0.9  # Ters eşleşme
                
                # Lig eşleşmesi bonusu
                if match.league_id == league_id:
                    score += 50.0
                
                if score > best_score:
                    best_score = score
                    best_match = match
            except Exception as e:
                logger.debug(f"Match scoring hatası: {e}")
                continue
        
        # Eşik değerini düşür - daha fazla maç bulsun
        threshold = 30.0  # 50'den 30'a düşürüldü
        
        if best_match and best_score >= threshold:
            confidence = min(1.0, best_score / 500.0)
            # Düşük güven skorlu eşleşmeleri logla
            if confidence < 0.3:
                logger.warning(f"⚠️ Düşük güven skoru ({confidence:.2%}): "
                             f"{home_team_name} vs {away_team_name} "
                             f"({match_date.date()}) -> Match ID {best_match.id}")
            return best_match, confidence
        
        # Eşleşme bulunamadı - detaylı log
        logger.debug(f"❌ Eşleşme bulunamadı: {home_team_name} vs {away_team_name} "
                     f"({match_date.date()}, Lig ID: {league_id}, "
                     f"Skor: {home_score}-{away_score}, "
                     f"En iyi skor: {best_score:.1f})")
        return None, 0.0
    except Exception as e:
        logger.error(f"find_match_in_db_advanced hatası: {e}")
        return None, 0.0


def process_csv_row_safe(row: Dict[str, str], csv_file: Path, row_num: int, session: Session) -> Dict[str, Any]:
    """
    CSV satırını işler - GÜVENLİ VERSİYON
    Returns: {'success': bool, 'match_id': int, 'odds_added': bool, 'error': str}
    """
    result = {'success': False, 'match_id': None, 'odds_added': False, 'error': None}
    
    try:
        # Tarih ve saat
        date_str = row.get('Date', '').strip()
        time_str = row.get('Time', '').strip()
        
        if not date_str:
            result['error'] = "Tarih bulunamadı"
            return result
        
        match_date = parse_date_safe(date_str, time_str)
        if not match_date:
            result['error'] = f"Tarih parse edilemedi: {date_str}"
            return result
        
        # Takımlar
        home_team = row.get('HomeTeam', '').strip()
        away_team = row.get('AwayTeam', '').strip()
        
        if not home_team or not away_team:
            result['error'] = "Takım isimleri eksik"
            return result
        
        # Skorlar
        home_score = safe_int(row.get('FTHG'))
        away_score = safe_int(row.get('FTAG'))
        
        # Lig kodu
        league_code = None
        file_name = csv_file.stem.upper()
        
        for code in LEAGUE_MAPPING.keys():
            if code in file_name:
                league_code = code
                break
        
        if not league_code:
            # Tahmin et - daha kapsamlı
            file_lower = file_name.lower()
            if 'premier' in file_lower or 'e0' in file_lower:
                league_code = 'E0'
            elif 'championship' in file_lower or 'e1' in file_lower:
                league_code = 'E1'
            elif 'league1' in file_lower or 'league-1' in file_lower or 'e2' in file_lower:
                league_code = 'E2'
            elif 'league2' in file_lower or 'league-2' in file_lower or 'e3' in file_lower:
                league_code = 'E3'
            elif 'serie-a' in file_lower or 'seriea' in file_lower or 'i1' in file_lower:
                league_code = 'I1'
            elif 'serie-b' in file_lower or 'serieb' in file_lower or 'i2' in file_lower:
                league_code = 'I2'
            elif ('bundesliga' in file_lower or 'd1' in file_lower) and '2' not in file_lower:
                league_code = 'D1'
            elif '2-bundesliga' in file_lower or '2bundesliga' in file_lower or 'd2' in file_lower:
                league_code = 'D2'
            elif ('ligue-1' in file_lower or 'ligue1' in file_lower or 'f1' in file_lower) and '2' not in file_lower:
                league_code = 'F1'
            elif 'ligue-2' in file_lower or 'ligue2' in file_lower or 'f2' in file_lower:
                league_code = 'F2'
            elif ('laliga' in file_lower or 'la-liga' in file_lower or 'sp1' in file_lower) and '2' not in file_lower:
                league_code = 'SP1'
            elif 'segunda' in file_lower or 'sp2' in file_lower:
                league_code = 'SP2'
            elif 'portugal' in file_lower or 'primeira' in file_lower or 'p1' in file_lower:
                league_code = 'P1'
            elif 'super-lig' in file_lower or 'superlig' in file_lower or 't1' in file_lower:
                league_code = 'T1'
        
        if not league_code:
            result['error'] = f"Lig kodu bulunamadı: {csv_file.name}"
            return result
        
        # Lig ismi - daha kapsamlı mapping
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
            result['error'] = f"Lig ismi bulunamadı: {league_code}"
            return result
        
        # Lig'i DB'de bul
        league = LeagueRepository.get_by_name(session, league_name)
        if not league:
            result['error'] = f"Lig bulunamadı: {league_name}"
            return result
        
        # Maçı bul
        match, confidence = find_match_in_db_advanced(
            session, league.id, home_team, away_team, match_date, home_score, away_score
        )
        
        if not match:
            result['error'] = f"Maç bulunamadı: {home_team} vs {away_team} ({match_date.date()})"
            return result
        
        result['match_id'] = match.id
        result['success'] = True
        
        # Odds'ları hazırla
        odds_data = {
            'b365_h': safe_float(row.get('B365H')),
            'b365_d': safe_float(row.get('B365D')),
            'b365_a': safe_float(row.get('B365A')),
        }
        
        # En az bir odds değeri olmalı
        if not any(odds_data.values()):
            result['error'] = "Odds verisi yok"
            return result
        
        # MatchOdds kaydını kontrol et
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
                    result['odds_added'] = True
            else:
                new_odds = MatchOdds(match_id=match.id, **odds_data)
                session.add(new_odds)
                result['odds_added'] = True
            
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            result['error'] = f"Veritabanı hatası: {str(e)}"
            return result
        
        return result
        
    except Exception as e:
        result['error'] = f"Beklenmeyen hata: {str(e)}"
        logger.error(f"❌ Satır işleme hatası (satır {row_num}): {e}")
        logger.error(traceback.format_exc())
        return result


def continuous_odds_loading():
    """SÜREKLI ÇALIŞAN ODDS YÜKLEME - ASLA DURMAZ"""
    print("=" * 80)
    print("🎲 ODDS SÜREKLI ÇALIŞAN SİSTEM - ASLA DURMAZ")
    print("=" * 80)
    print("📋 Özellikler:")
    print("  ✅ Sürekli çalışır (asla durmaz)")
    print("  ✅ Her hata bildirilir ve çözülür")
    print("  ✅ Satır işleme hataları yakalanır")
    print("  ✅ Otomatik hata yönetimi")
    print("=" * 80)
    print()
    
    odds_dir = Path(__file__).parent / "odds"
    
    if not odds_dir.exists():
        logger.error(f"❌ Odds klasörü bulunamadi: {odds_dir}")
        return
    
    iteration = 0
    total_unmatched = []
    consecutive_errors = 0
    max_consecutive_errors = 20
    
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
                'unmatched': [],
                'row_errors': []
            }
            
            # Tüm lig klasörlerini tara
            for league_folder in odds_dir.iterdir():
                if not league_folder.is_dir():
                    continue
                
                logger.info(f"📂 {league_folder.name} klasörü işleniyor...")
                
                csv_files = list(league_folder.glob("*.csv"))
                
                for csv_file in csv_files:
                    try:
                        logger.info(f"   📄 {csv_file.name} işleniyor...")
                        
                        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                            reader = csv.DictReader(f)
                            
                            for row_num, row in enumerate(reader, 1):
                                total_stats['total_rows'] += 1
                                
                                # Satırı işle - GÜVENLİ VERSİYON
                                result = process_csv_row_safe(row, csv_file, row_num, session)
                                
                                if result['success']:
                                    total_stats['matches_found'] += 1
                                    if result['odds_added']:
                                        total_stats['odds_added'] += 1
                                else:
                                    total_stats['errors'] += 1
                                    total_stats['row_errors'].append({
                                        'file': csv_file.name,
                                        'row': row_num,
                                        'error': result['error']
                                    })
                                    
                                    # Eşleşmeyen maçları kaydet
                                    if 'bulunamadı' in result['error']:
                                        home_team = row.get('HomeTeam', '').strip()
                                        away_team = row.get('AwayTeam', '').strip()
                                        date_str = row.get('Date', '').strip()
                                        total_stats['unmatched'].append({
                                            'file': csv_file.name,
                                            'row': row_num,
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'date': date_str,
                                            'error': result['error']
                                        })
                                    
                                    # Hata bildirimi
                                    logger.warning(f"⚠️ SATIR İŞLEME HATASI - {csv_file.name}:{row_num}: {result['error']}")
                                
                                # Her 50 satırda bir commit - OPTİMİZE EDİLDİ (daha sık kayıt)
                                if total_stats['total_rows'] % 50 == 0:
                                    try:
                                        session.commit()
                                    except:
                                        session.rollback()
                    
                    except Exception as e:
                        logger.error(f"❌ CSV dosya okuma hatası {csv_file}: {e}")
                        logger.error(traceback.format_exc())
                        total_stats['errors'] += 1
                        consecutive_errors += 1
                        continue
                
                # Her klasör sonunda commit
                try:
                    session.commit()
                except:
                    session.rollback()
            
            # Özet
            print(f"\n📊 İTERASYON {iteration} ÖZETİ:")
            print(f"   📄 Toplam satır: {total_stats['total_rows']:,}")
            print(f"   ✅ Maç bulundu: {total_stats['matches_found']:,}")
            print(f"   ➕ Odds eklendi: {total_stats['odds_added']:,}")
            print(f"   ❌ Eşleşmeyen: {len(total_stats['unmatched']):,}")
            print(f"   ⚠️ Satır hataları: {len(total_stats['row_errors']):,}")
            
            # Hataları göster
            if total_stats['row_errors']:
                print(f"\n⚠️ SATIR HATALARI (ilk 5):")
                for err in total_stats['row_errors'][:5]:
                    print(f"   - {err['file']}:{err['row']}: {err['error']}")
            
            total_unmatched = total_stats['unmatched']
            
            # Eğer eşleşmeyen maç yoksa, tamamlandı
            if len(total_unmatched) == 0 and len(total_stats['row_errors']) == 0:
                print(f"\n{'=' * 80}")
                print("🎉 TÜM MAÇLAR EŞLEŞTİRİLDİ VE HATA YOK!")
                print(f"{'=' * 80}")
                print("⏳ 60 saniye bekleniyor, sonra tekrar kontrol edilecek...")
                time.sleep(60)
                continue
            
            # Çok fazla ardışık hata varsa bekle
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"❌ {max_consecutive_errors} ardışık hata! 30 saniye bekleniyor...")
                print(f"\n⚠️ Çok fazla ardışık hata ({consecutive_errors}), 30 saniye bekleniyor...")
                time.sleep(30)
                consecutive_errors = 0
            
            # Bir sonraki iterasyona geç - OPTİMİZE EDİLDİ
            print(f"\n⏳ 5 saniye bekleniyor, sonra tekrar deneniyor...")
            time.sleep(5)  # 10'dan 5'e düşürüldü
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Kullanıcı tarafından durduruldu!")
            break
        except Exception as e:
            logger.error(f"❌ GENEL HATA: {e}")
            logger.error(traceback.format_exc())
            consecutive_errors += 1
            print(f"\n❌ Genel hata oluştu, 10 saniye bekleniyor...")
            time.sleep(10)
        finally:
            try:
                session.close()
            except:
                pass


if __name__ == "__main__":
    import time
    try:
        continuous_odds_loading()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program sonlandırıldı!")
    except Exception as e:
        logger.error(f"❌ Kritik hata: {e}")
        logger.error(traceback.format_exc())

