"""Hızlı API testi"""
import os
import sys
from datetime import date

# API key'i ayarla
os.environ["API_FOOTBALL_KEY"] = "647f5de88a29d150a9d4e2c0c7b636fb"

# Path ayarla
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.ingestion.api_client import APIFootballClient
    
    print("=" * 60)
    print("API Veri Cekme Testi")
    print("=" * 60)
    
    client = APIFootballClient()
    print("[OK] API client olusturuldu")
    
    # Bugünün fikstürlerini çek
    today = date.today()
    print(f"\nTarih: {today}")
    print("Fiksturler cekiliyor...")
    
    try:
        fixtures = client.get_fixtures(date_from=today, date_to=today)
    except Exception as e:
        print(f"[HATA] Fikstur cekme hatasi: {e}")
        fixtures = None
    
    if fixtures:
        print(f"\n[OK] API VERI CEKIYOR! {len(fixtures)} fikstur bulundu.")
        print("\nIlk 3 fikstur:")
        for i, fixture in enumerate(fixtures[:3], 1):
            teams = fixture.get("teams", {})
            home = teams.get("home", {}).get("name", "N/A")
            away = teams.get("away", {}).get("name", "N/A")
            print(f"  {i}. {home} vs {away}")
    else:
        print("\n[UYARI] Bugun icin fikstur bulunamadi")
        print("[OK] Ancak API baglantisi calisiyor!")
        
        # Alternatif test: Geçmiş bir tarihten fikstür çek (Free plan için)
        from datetime import timedelta
        yesterday = today - timedelta(days=1)
        print(f"\nDunun fiksturlerini deniyorum ({yesterday})...")
        try:
            fixtures = client.get_fixtures(date_from=yesterday, date_to=yesterday)
            if fixtures:
                print(f"[OK] API VERI CEKIYOR! {len(fixtures)} fikstur bulundu.")
                print("\nIlk 3 fikstur:")
                for i, fixture in enumerate(fixtures[:3], 1):
                    teams = fixture.get("teams", {})
                    home = teams.get("home", {}).get("name", "N/A")
                    away = teams.get("away", {}).get("name", "N/A")
                    print(f"  {i}. {home} vs {away}")
            else:
                print("[UYARI] Dunun fiksturleri de bulunamadi")
        except Exception as e:
            print(f"[UYARI] Fikstur testi hatasi: {e}")
        
        # Başka bir test: Ligleri çek
        print("\nLigleri cekiyorum (England, 2024)...")
        try:
            leagues = client.get_leagues(country="England", season=2024)
            if leagues:
                print(f"[OK] API VERI CEKIYOR! {len(leagues)} lig bulundu.")
                print("\nIlk 3 lig:")
                for i, league in enumerate(leagues[:3], 1):
                    league_name = league.get("league", {}).get("name", "N/A")
                    print(f"  {i}. {league_name}")
            else:
                print("[UYARI] Lig verisi bos dondu")
        except Exception as e:
            print(f"[HATA] Lig testi hatasi: {e}")
            import traceback
            traceback.print_exc()
        
        # API Limit kontrolü
        print("\nAPI Limit Durumu Kontrol Ediliyor...")
        try:
            import requests
            url = "https://v3.football.api-sports.io/fixtures"
            headers = {
                "x-apisports-key": os.environ["API_FOOTBALL_KEY"],
                "Content-Type": "application/json"
            }
            # Free plan için önerilen tarih aralığını kullan (2025-11-28 to 2025-11-30)
            params = {"date": "2025-11-28"}
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            # Rate limit bilgilerini kontrol et
            rate_limit_headers = {
                "x-ratelimit-requests-limit": response.headers.get("x-ratelimit-requests-limit", "N/A"),
                "x-ratelimit-requests-remaining": response.headers.get("x-ratelimit-requests-remaining", "N/A"),
                "x-ratelimit-requests-reset": response.headers.get("x-ratelimit-requests-reset", "N/A")
            }
            
            print("\nAPI Limit Bilgileri:")
            for key, value in rate_limit_headers.items():
                print(f"  {key}: {value}")
            
            data = response.json()
            
            if "errors" in data and data["errors"]:
                print(f"\nAPI Hatalari: {data['errors']}")
            
            if "response" in data:
                print(f"\nResponse Array Length: {len(data['response'])}")
                if data['response']:
                    print(f"[OK] API VERI CEKIYOR! {len(data['response'])} fikstur bulundu.")
                    print("\nIlk 3 fikstur:")
                    for i, fixture in enumerate(data['response'][:3], 1):
                        teams = fixture.get("teams", {})
                        home = teams.get("home", {}).get("name", "N/A")
                        away = teams.get("away", {}).get("name", "N/A")
                        print(f"  {i}. {home} vs {away}")
                else:
                    print("[UYARI] Veri bulunamadi (tarih icin fikstur olmayabilir)")
            
            # Limit durumu özeti
            remaining = rate_limit_headers.get("x-ratelimit-requests-remaining", "N/A")
            limit = rate_limit_headers.get("x-ratelimit-requests-limit", "N/A")
            
            if remaining != "N/A" and limit != "N/A":
                remaining_int = int(remaining)
                limit_int = int(limit)
                if remaining_int == 0:
                    print(f"\n[UYARI] API HAKKI DOLMUS! {limit_int} request kullanilmis.")
                elif remaining_int < 10:
                    print(f"\n[UYARI] API hakki az kaliyor: {remaining_int}/{limit_int} kaldi")
                else:
                    print(f"\n[OK] API hakki yeterli: {remaining_int}/{limit_int} kaldi")
                    
        except Exception as e:
            print(f"[HATA] Detayli kontrol hatasi: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("[OK] API CALISIYOR VE VERI CEKIYOR!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[HATA] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

