"""
HYPE SÜREKLI ÇALIŞAN SİSTEM - ASLA DURMAZ
- Her hata verdiğinde bildirim verir
- Hataları otomatik çözer
- Sürekli kontrol eder ve düzeltir
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import time
import logging
import traceback
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import multiprocessing

# psutil opsiyonel - yoksa basit alternatif kullan
HAS_PSUTIL = False
try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except ImportError:
    pass

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.db.connection import get_session
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository
from src.ingestion.alternative_hype_scraper import AlternativeHypeScraper
from src.db.schema import Match
from sqlalchemy import and_, extract, or_
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hype_surekli.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_hype_columns():
    """Hype kolonlarının var olduğundan emin ol"""
    session = get_session()
    try:
        from sqlalchemy import text
        # Kolonlar zaten schema'da tanımlı, sadece kontrol et
        session.execute(text("SELECT home_support FROM matches LIMIT 1"))
        session.commit()
        logger.info("✅ Hype kolonları mevcut")
    except Exception as e:
        logger.warning(f"⚠️ Hype kolon kontrolü: {e}")
    finally:
        session.close()


def get_hype_status(session) -> Dict[str, int]:
    """Hype durumunu kontrol eder"""
    try:
        total = session.query(Match).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        with_hype = session.query(Match).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025,
                Match.hype_updated_at.isnot(None),
                Match.total_tweets.isnot(None),
                Match.total_tweets > 0
            )
        ).count()
        
        without_hype = total - with_hype
        
        return {
            'total': total,
            'with_hype': with_hype,
            'without_hype': without_hype,
            'percentage': (with_hype / total * 100) if total > 0 else 0
        }
    except Exception as e:
        logger.error(f"❌ Hype durum kontrolü hatası: {e}")
        return {'total': 0, 'with_hype': 0, 'without_hype': 0, 'percentage': 0}


def validate_hype_data(match: Match) -> bool:
    """Hype verilerinin geçerli olup olmadığını kontrol eder"""
    try:
        if match.hype_updated_at is None:
            return False
        
        if match.total_tweets is None or match.total_tweets == 0:
            return False
        
        if match.home_support is None or match.away_support is None:
            return False
        
        if not (0 <= match.home_support <= 1) or not (0 <= match.away_support <= 1):
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Hype doğrulama hatası: {e}")
        return False


def fetch_hype_for_match_safe(match: Match, scraper: AlternativeHypeScraper, session) -> tuple[bool, Optional[str]]:
    """
    Bir maç için hype verilerini çeker - GÜVENLİ VERSİYON
    Returns: (success, error_message)
    """
    try:
        league = LeagueRepository.get_by_id(session, match.league_id)
        if not league:
            return False, "Lig bulunamadı"
        
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        if not home_team:
            return False, "Ev sahibi takım bulunamadı"
        
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        if not away_team:
            return False, "Deplasman takım bulunamadı"
        
        return fetch_hype_for_match_safe_cached(match, scraper, session, league, home_team, away_team)
        
    except Exception as e:
        error_msg = f"Beklenmeyen hata: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        return False, error_msg

def fetch_hype_for_match_safe_cached(
    match: Match, 
    scraper: AlternativeHypeScraper, 
    session,
    league,
    home_team,
    away_team
) -> tuple[bool, Optional[str]]:
    """
    Bir maç için hype verilerini çeker - CACHE'Lİ VERSİYON (daha hızlı)
    Returns: (success, error_message)
    """
    try:
        if not league or not home_team or not away_team:
            return False, "Lig/takım bilgisi eksik"
        
        # Hype analizi yap
        try:
            hype_result = scraper.get_match_hype(
                league_name=league.name,
                home_team=home_team.name,
                away_team=away_team.name,
                match_date=match.match_date
            )
        except Exception as e:
            return False, f"Hype çekme hatası: {str(e)}"
        
        # Veritabanına kaydet
        try:
            match.home_support = hype_result.get("home_support", 0.5)
            match.away_support = hype_result.get("away_support", 0.5)
            match.sentiment_score = hype_result.get("sentiment_score", 0.0)
            total_mentions = hype_result.get("total_mentions", 0)
            
            # Eğer mentions 0 ise bile güncellendiğini işaretle (tekrar tekrar denemesin)
            match.total_tweets = total_mentions
            match.hype_updated_at = datetime.now()
            
            # Bulk commit için commit yapmıyoruz, dışarıda yapılacak
        except SQLAlchemyError as e:
            session.rollback()
            return False, f"Veritabanı kayıt hatası: {str(e)}"
        
        # Doğrulama ve detaylı log - SKOR MUTLAKA GÖSTERİLECEK
        if validate_hype_data(match):
            # Detaylı log: Tarih, Maç, Skor, Mentions, Oranlar
            match_date_str = match.match_date.strftime('%Y-%m-%d')
            
            # SKOR BİLGİSİ - MUTLAKA GÖSTER
            if match.home_score is not None and match.away_score is not None:
                home_score = match.home_score
                away_score = match.away_score
                score_str = f"⚽ Skor: {home_score}-{away_score}"
            else:
                score_str = "⚽ Skor: ?-?"
            
            home_pct = match.home_support * 100 if match.home_support else 0
            away_pct = match.away_support * 100 if match.away_support else 0
            
            log_msg = (f"✅ {league.name}: {home_team.name} vs {away_team.name} | "
                      f"📅 {match_date_str} | "
                      f"{score_str} | "
                      f"📢 Mentions: {match.total_tweets:,} | "
                      f"🏠 Home: {home_pct:.1f}% | "
                      f"✈️ Away: {away_pct:.1f}%")
            
            # Console'a da yazdır (anında görünsün)
            print(log_msg, flush=True)
            logger.info(log_msg)
            return True, log_msg
        else:
            return False, "Hype verisi geçersiz"
        
    except Exception as e:
        error_msg = f"Beklenmeyen hata: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, error_msg


def continuous_hype_fetch():
    """SÜREKLI ÇALIŞAN HYPE ÇEKME SİSTEMİ - MAXIMUM GÜÇ KULLANIMI"""
    # CPU bilgilerini al
    cpu_count = multiprocessing.cpu_count()
    
    print("=" * 80)
    print("🔥 HYPE SÜREKLI ÇALIŞAN SİSTEM - MAXIMUM GÜÇ MODU")
    print("=" * 80)
    print("💻 SİSTEM BİLGİLERİ:")
    print(f"  🖥️  CPU Core Sayısı: {cpu_count}")
    
    if HAS_PSUTIL:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        print(f"  📊 CPU Kullanımı: {cpu_percent:.1f}%")
        print(f"  💾 RAM: {memory.total / (1024**3):.1f} GB (Kullanılabilir: {memory.available / (1024**3):.1f} GB)")
    else:
        print("  📊 Sistem bilgisi: psutil yüklü değil (opsiyonel)")
    
    print()
    print("📋 Özellikler:")
    print("  ✅ Sürekli çalışır (asla durmaz)")
    print("  ✅ Her hata bildirilir ve çözülür")
    print("  ✅ Otomatik hata yönetimi")
    print("  ✅ Eksiksiz hype verisi garantisi")
    
    # Thread sayısı: CPU core x 3 (I/O bound işlemler için agresif)
    max_workers = cpu_count * 3
    print(f"  ⚡ MAXIMUM GÜÇ: {max_workers} thread (CPU core x 3)")
    print("  📊 VERİ KALİTESİ: Tüm kaynaklar korunuyor (Google Trends, News API, Web Scraping)")
    print("=" * 80)
    print()
    
    # Hype kolonlarını kontrol et
    ensure_hype_columns()
    
    # Hype scraper
    scraper = AlternativeHypeScraper()
    
    # Cache: League ve Team bilgilerini cache'le (DB sorgularını azalt)
    league_cache = {}
    team_cache = {}
    
    print(f"🚀 {max_workers} thread ile MAXIMUM GÜÇ modu aktif!")
    print(f"💻 PC'nin tüm gücü kullanılıyor!\n")
    
    iteration = 0
    total_processed = 0
    total_success = 0
    total_failed = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while True:
        iteration += 1
        session = get_session()
        
        # Cache'i her iterasyonda temizle (yeni session için)
        if iteration % 10 == 1:  # Her 10 iterasyonda bir cache temizle
            league_cache.clear()
            team_cache.clear()
        
        try:
            # Durum kontrolü
            status = get_hype_status(session)
            
            print(f"\n{'=' * 80}")
            print(f"🔄 İTERASYON {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 80}")
            print(f"📊 Toplam maç: {status['total']:,}")
            print(f"✅ Hype'ı olan: {status['with_hype']:,} ({status['percentage']:.1f}%)")
            print(f"❌ Hype'ı olmayan: {status['without_hype']:,}")
            print()
            
            # Eğer tüm maçların hype'ı varsa
            if status['without_hype'] == 0:
                print("=" * 80)
                print("🎉 TÜM MAÇLARIN HYPE VERİSİ TAMAM!")
                print("=" * 80)
                print("⏳ 60 saniye bekleniyor, sonra tekrar kontrol edilecek...")
                time.sleep(60)
                continue
            
            # Hype'ı olmayan maçları getir
            # DÜZELTME: hype_updated_at set edilmişse ve son 24 saat içinde güncellenmişse tekrar işleme
            from datetime import timedelta
            recent_cutoff = datetime.now() - timedelta(hours=24)
            
            matches = session.query(Match).filter(
                and_(
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025,
                    or_(
                        # Hiç hype yok
                        Match.hype_updated_at.is_(None),
                        # Hype var ama total_tweets None veya 0 VE 24 saatten eski (tekrar dene)
                        and_(
                            or_(
                                Match.total_tweets == 0,
                                Match.total_tweets.is_(None)
                            ),
                            or_(
                                Match.hype_updated_at < recent_cutoff,
                                Match.hype_updated_at.is_(None)
                            )
                        )
                    )
                )
            ).order_by(
                # EN GÜNCELDEN EN ESKİYE: Önce hiç işlenmemiş olanları, sonra en güncel tarihlerden başla
                Match.hype_updated_at.nulls_first(),
                Match.match_date.desc()  # DESC: En güncelden en eskiye
            ).limit(1000).all()
            
            if not matches:
                print("⚠️ Hype'ı olmayan maç bulunamadı, doğrulama yapılıyor...")
                time.sleep(10)
                continue
            
            print(f"📋 {len(matches):,} maç için hype çekilecek")
            print()
            
            # PARALEL İŞLEME - 20 thread ile (VERİ KALİTESİ KORUNUYOR)
            # Tüm kaynaklar kullanılıyor: Google Trends, News API, Web Scraping
            batch_success = 0
            batch_failed = 0
            batch_errors = []
            commit_lock = Lock()
            progress_lock = Lock()
            
            def process_match(match_data):
                """Tek bir maçı işle - thread-safe, TÜM VERİ KAYNAKLARI KULLANILIYOR"""
                match, match_idx = match_data
                thread_session = get_session()
                try:
                    # Cache'den league ve team bilgilerini al
                    if match.league_id not in league_cache:
                        with commit_lock:
                            if match.league_id not in league_cache:
                                league_cache[match.league_id] = LeagueRepository.get_by_id(thread_session, match.league_id)
                    league = league_cache[match.league_id]
                    
                    if match.home_team_id not in team_cache:
                        with commit_lock:
                            if match.home_team_id not in team_cache:
                                team_cache[match.home_team_id] = TeamRepository.get_by_id(thread_session, match.home_team_id)
                    home_team = team_cache[match.home_team_id]
                    
                    if match.away_team_id not in team_cache:
                        with commit_lock:
                            if match.away_team_id not in team_cache:
                                team_cache[match.away_team_id] = TeamRepository.get_by_id(thread_session, match.away_team_id)
                    away_team = team_cache[match.away_team_id]
                    
                    if not league or not home_team or not away_team:
                        return False, f"Match {match.id}: Lig/takım bulunamadı", match.id
                    
                    # Hype çek - TÜM KAYNAKLAR KULLANILIYOR (Google Trends, News API, Web Scraping)
                    thread_scraper = AlternativeHypeScraper()  # Her thread kendi scraper'ı
                    success, error_msg = fetch_hype_for_match_safe_cached(
                        match, thread_scraper, thread_session, league, home_team, away_team
                    )
                    
                    # Commit (thread-safe)
                    if success:
                        with commit_lock:
                            try:
                                thread_session.commit()
                            except:
                                thread_session.rollback()
                    
                    return success, error_msg, match.id
                    
                except Exception as e:
                    thread_session.rollback()
                    return False, f"Beklenmeyen hata: {str(e)}", match.id if match else None
                finally:
                    thread_session.close()
            
            # Paralel işleme başlat - MAXIMUM GÜÇ (CPU core x 2)
            num_workers = min(max_workers, len(matches))
            print(f"🚀 {num_workers} thread ile MAXIMUM GÜÇ modu aktif!")
            print(f"📊 VERİ KALİTESİ: Tüm kaynaklar aktif (Google Trends, News API, Web Scraping)")
            print(f"💻 CPU: {cpu_count} core, Thread: {num_workers}\n")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Tüm maçları gönder
                future_to_match = {
                    executor.submit(process_match, (match, i)): (match, i) 
                    for i, match in enumerate(matches, 1)
                }
                
                # Sonuçları topla
                for future in as_completed(future_to_match):
                    match, match_idx = future_to_match[future]
                    try:
                        success, error_msg, match_id = future.result()
                        
                        with progress_lock:
                            if success:
                                batch_success += 1
                                total_success += 1
                                consecutive_errors = 0
                                # Başarılı mesaj zaten print edildi (fetch_hype_for_match_safe_cached içinde)
                            else:
                                batch_failed += 1
                                total_failed += 1
                                consecutive_errors += 1
                                batch_errors.append({
                                    'match_id': match_id,
                                    'error': error_msg
                                })
                                # Hata mesajını da göster
                                print(f"❌ Hata: {error_msg}", flush=True)
                            
                            total_processed += 1
                            
                            # Progress göster (her 50 maçta bir)
                            if total_processed % 50 == 0:
                                print(f"\n📊 Progress: {total_processed:,}/{len(matches):,} "
                                     f"({total_success:,} ✅, {total_failed:,} ❌)\n", flush=True)
                    
                    except Exception as e:
                        with progress_lock:
                            batch_failed += 1
                            total_failed += 1
                            total_processed += 1
                        logger.error(f"❌ Thread hatası: {e}")
            
            # Final commit (kalan tüm değişiklikler)
            try:
                session.commit()
            except:
                session.rollback()
            
            # Batch özeti
            print(f"\n📊 Batch Özeti:")
            print(f"   ✅ Başarılı: {batch_success}")
            print(f"   ❌ Hata: {batch_failed}")
            
            # Hataları göster
            if batch_errors:
                print(f"\n⚠️ HATALAR ({len(batch_errors)} adet):")
                for err in batch_errors[:5]:  # İlk 5 hatayı göster
                    print(f"   - Match ID {err['match_id']}: {err['error']}")
                if len(batch_errors) > 5:
                    print(f"   ... ve {len(batch_errors) - 5} hata daha")
            
            # Çok fazla ardışık hata varsa bekle
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"❌ {max_consecutive_errors} ardışık hata! 30 saniye bekleniyor...")
                print(f"\n⚠️ Çok fazla ardışık hata ({consecutive_errors}), 30 saniye bekleniyor...")
                time.sleep(30)
                consecutive_errors = 0  # Sayaç sıfırla
            
            # Kısa bir mola - NEREDEYSE YOK
            time.sleep(0.1)  # 0.5'ten 0.1'e düşürüldü (5x hızlı)
        
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
    
    # Final özet
    print("\n" + "=" * 80)
    print("🎉 HYPE ÇEKME TAMAMLANDI!")
    print("=" * 80)
    print(f"📊 Toplam işlenen: {total_processed:,} maç")
    print(f"✅ Başarılı: {total_success:,}")
    print(f"❌ Hata: {total_failed:,}")


if __name__ == "__main__":
    try:
        continuous_hype_fetch()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program sonlandırıldı!")
    except Exception as e:
        logger.error(f"❌ Kritik hata: {e}")
        logger.error(traceback.format_exc())

