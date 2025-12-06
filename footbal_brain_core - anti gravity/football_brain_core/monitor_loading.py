"""
Sürekli yükleme durumunu kontrol et
Durduğunda uyarı ver ve yeni API key beklesin
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match
from football_brain_core.src.ingestion.api_client import APIFootballClient

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("SUREKLI YUKLEME MONITOR")
    print("=" * 60)
    print("Her 30 saniyede bir kontrol ediliyor...")
    print("Durdugunda uyari verilecek ve yeni API key bekleyecek")
    print("Durdurmak icin: Ctrl+C")
    print("=" * 60)
    print()
    
    last_match_count = 0
    last_check_time = datetime.now()
    check_interval = 30  # 30 saniye
    stuck_threshold = 120  # 2 dakika değişiklik yoksa durmuş sayılır
    
    try:
        while True:
            current_time = datetime.now()
            
            # Veritabanı kontrolü
            session = get_session()
            try:
                current_match_count = session.query(Match).count()
                
                # Son maç bilgisi
                last_match = session.query(Match).order_by(Match.created_at.desc()).first()
                last_match_time = last_match.created_at if last_match else None
                
            finally:
                session.close()
            
            # İlerleme hesapla
            target_matches = 13300  # 5 sezon x 7 lig x ~380 maç
            progress = (current_match_count / target_matches) * 100 if target_matches > 0 else 0
            
            # Durum mesajı
            print(f"[{current_time.strftime('%H:%M:%S')}] Mac sayisi: {current_match_count} ({progress:.1f}%)")
            
            # Değişiklik var mı?
            if current_match_count > last_match_count:
                print(f"  [+] {current_match_count - last_match_count} yeni mac eklendi!")
                last_match_count = current_match_count
                last_check_time = current_time
            else:
                # Ne kadar süredir değişiklik yok?
                time_since_last_update = (current_time - last_check_time).total_seconds()
                
                if time_since_last_update > stuck_threshold:
                    print()
                    print("!" * 60)
                    print("YUKLEME DURDU!")
                    print("!" * 60)
                    print(f"Son guncelleme: {time_since_last_update:.0f} saniye once")
                    print(f"Son mac sayisi: {current_match_count}")
                    print()
                    
                    # API kontrolü
                    try:
                        api_key = os.getenv("API_FOOTBALL_KEY", "")
                        if api_key:
                            client = APIFootballClient(api_key=api_key)
                            remaining = client.daily_limit - client.requests_today
                            
                            if remaining <= 0:
                                print("[UYARI] API limiti asildi!")
                                print(f"Gunluk limit: {client.daily_limit}")
                                print(f"Kullanilan: {client.requests_today}")
                                print()
                                print("YENI API KEY GEREKLI!")
                                print("Yeni key'i ayarlamak icin:")
                                print('  $env:API_FOOTBALL_KEY="YENI_KEY_BURAYA"')
                                print()
                                print("Sonra yuklemeyi yeniden baslat:")
                                print("  python load_all_sequential.py")
                            else:
                                print(f"[INFO] API limiti var: {remaining} kalan")
                                print("Yukleme durmus olabilir, kontrol et!")
                        else:
                            print("[HATA] API_FOOTBALL_KEY ayarlanmamis!")
                    except Exception as e:
                        print(f"[HATA] API kontrol hatasi: {e}")
                    
                    print("!" * 60)
                    print()
            
            # API durumu
            try:
                api_key = os.getenv("API_FOOTBALL_KEY", "")
                if api_key:
                    client = APIFootballClient(api_key=api_key)
                    remaining = client.daily_limit - client.requests_today
                    if remaining < 10:
                        print(f"  [UYARI] API limiti yaklasiyor: {remaining} kalan")
            except:
                pass
            
            # Bekle
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitor durduruldu.")
        print(f"Son mac sayisi: {current_match_count}")
        print(f"Ilerleme: {progress:.1f}%")







