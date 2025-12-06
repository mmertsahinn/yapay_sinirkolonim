"""
API durumunu ve yükleme işlemini kontrol et
"""
import sys
import os
from pathlib import Path

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.ingestion.api_client import APIFootballClient

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("API DURUMU KONTROLU")
    print("=" * 60)
    
    try:
        api_key = os.getenv("API_FOOTBALL_KEY", "")
        if not api_key:
            print("[HATA] API_FOOTBALL_KEY ayarlanmamis!")
            exit(1)
        
        client = APIFootballClient(api_key=api_key)
        
        print(f"\nAPI Key: {api_key[:20]}...")
        print(f"Gunluk limit: {client.daily_limit}")
        print(f"Bugun kullanilan: {client.requests_today}")
        print(f"Kalan: {client.daily_limit - client.requests_today}")
        
        # Test request
        print("\nAPI test ediliyor...")
        from datetime import date
        fixtures = client.get_fixtures(date_from=date.today(), date_to=date.today())
        print(f"[OK] API calisiyor! Bugun {len(fixtures)} fikstur bulundu.")
        
        print(f"\nGunluk limit: {client.daily_limit}")
        print(f"Kalan request: {client.daily_limit - client.requests_today}")
        
        if client.requests_today >= client.daily_limit:
            print("\n[UYARI] Gunluk limit asildi! Yarın devam edebilirsin.")
        elif client.requests_today >= client.daily_limit * 0.9:
            print("\n[UYARI] Limit yaklasiyor! Dikkatli kullan.")
        else:
            print("\n[OK] Yükleme devam edebilir!")
        
    except Exception as e:
        print(f"[HATA] API kontrol hatasi: {e}")







