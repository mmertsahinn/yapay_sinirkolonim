"""
Sürekli Döngüsel Kontrol Sistemi
Her 30 saniyede bir kontrol eder, durduğunda uyarı verir
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

def check_status():
    """Durum kontrolü yap"""
    session = get_session()
    try:
        match_count = session.query(Match).count()
        last_match = session.query(Match).order_by(Match.created_at.desc()).first()
        
        return {
            "match_count": match_count,
            "last_match_time": last_match.created_at if last_match else None,
            "last_match_id": last_match.id if last_match else None
        }
    finally:
        session.close()

def check_api_status():
    """API durumunu kontrol et"""
    try:
        api_key = os.getenv("API_FOOTBALL_KEY", "")
        if not api_key:
            return {"status": "no_key", "remaining": 0}
        
        client = APIFootballClient(api_key=api_key)
        remaining = client.daily_limit - client.requests_today
        
        return {
            "status": "ok" if remaining > 0 else "limit_reached",
            "remaining": remaining,
            "daily_limit": client.daily_limit,
            "used": client.requests_today
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """Ana döngü"""
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 70)
    print("SUREKLI DOGUSEL KONTROL SISTEMI BASLATILIYOR")
    print("=" * 70)
    print(f"Baslangic: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Kontrol araligi: 30 saniye")
    print("Durdurmak icin: Ctrl+C")
    print("=" * 70)
    print()
    
    check_interval = 30  # 30 saniye
    stuck_threshold = 120  # 2 dakika değişiklik yoksa durmuş sayılır
    
    last_match_count = 0
    last_match_time = None
    last_check_time = datetime.now()
    check_count = 0
    
    try:
        while True:
            check_count += 1
            current_time = datetime.now()
            
            # Veritabanı kontrolü
            db_status = check_status()
            current_match_count = db_status["match_count"]
            current_match_time = db_status["last_match_time"]
            
            # İlerleme hesapla
            target_matches = 13300
            progress = (current_match_count / target_matches) * 100 if target_matches > 0 else 0
            
            # API kontrolü
            api_status = check_api_status()
            
            # Durum mesajı
            print(f"[{current_time.strftime('%H:%M:%S')}] Kontrol #{check_count}")
            print(f"  Mac sayisi: {current_match_count} ({progress:.1f}%)")
            
            # Değişiklik kontrolü
            if current_match_count > last_match_count:
                new_matches = current_match_count - last_match_count
                print(f"  [+] {new_matches} yeni mac eklendi!")
                last_match_count = current_match_count
                last_match_time = current_match_time
                last_check_time = current_time
            elif current_match_count == last_match_count:
                # Ne kadar süredir değişiklik yok?
                if last_match_time:
                    time_since_update = (current_time - last_match_time).total_seconds()
                else:
                    time_since_update = (current_time - last_check_time).total_seconds()
                
                if time_since_update > stuck_threshold:
                    print()
                    print("!" * 70)
                    print("YUKLEME DURDU!")
                    print("!" * 70)
                    print(f"  Son guncelleme: {time_since_update:.0f} saniye once")
                    print(f"  Son mac sayisi: {current_match_count}")
                    print(f"  Son mac ID: {db_status.get('last_match_id', 'N/A')}")
                    print()
                    
                    # API durumu
                    if api_status["status"] == "limit_reached":
                        print("  [UYARI] API LIMITI ASILDI!")
                        print(f"  Gunluk limit: {api_status.get('daily_limit', 0)}")
                        print(f"  Kullanilan: {api_status.get('used', 0)}")
                        print()
                        print("  YENI API KEY GEREKLI!")
                        print("  Yeni key'i ayarlamak icin:")
                        print('    $env:API_FOOTBALL_KEY="YENI_KEY_BURAYA"')
                        print()
                        print("  Sonra yuklemeyi yeniden baslat:")
                        print("    python load_all_sequential.py")
                    elif api_status["status"] == "ok":
                        remaining = api_status.get("remaining", 0)
                        if remaining > 0:
                            print(f"  [INFO] API limiti var: {remaining} kalan")
                            print("  Yukleme durmus olabilir, kontrol et!")
                        else:
                            print("  [UYARI] API limiti bitti!")
                    elif api_status["status"] == "no_key":
                        print("  [HATA] API_FOOTBALL_KEY ayarlanmamis!")
                    else:
                        print(f"  [HATA] API kontrol hatasi: {api_status.get('error', 'Bilinmeyen')}")
                    
                    print("!" * 70)
                    print()
                else:
                    print(f"  [BEKLEMEDE] {time_since_update:.0f} saniye gecen, henuz normal")
            
            # API durumu göster
            if api_status["status"] == "ok":
                remaining = api_status.get("remaining", 0)
                if remaining < 10:
                    print(f"  [UYARI] API limiti yaklasiyor: {remaining} kalan")
                else:
                    print(f"  [API] {remaining} request kalan")
            
            print("-" * 70)
            print()
            
            # Bekle
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("KONTROL SISTEMI DURDURULDU")
        print("=" * 70)
        print(f"Toplam kontrol: {check_count}")
        print(f"Son mac sayisi: {current_match_count}")
        print(f"Son ilerleme: {progress:.1f}%")
        print(f"Bitis zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

if __name__ == "__main__":
    main()







