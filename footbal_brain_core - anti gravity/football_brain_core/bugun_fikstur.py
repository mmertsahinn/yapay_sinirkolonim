"""Bugünün fikstürlerini kontrol et"""
import os
import sys
from datetime import date, datetime

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# API key'i ayarla
os.environ["API_FOOTBALL_KEY"] = "647f5de88a29d150a9d4e2c0c7b636fb"

# Path ayarla
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("BUGUNUN FIKSTURLERI")
print("=" * 80)

try:
    from src.ingestion.api_client import APIFootballClient
    
    client = APIFootballClient()
    today = date.today()
    
    print(f"\nTarih: {today.strftime('%Y-%m-%d')} ({today.strftime('%A')})")
    print(f"Gun: {today.strftime('%d %B %Y')}")
    print("\nFiksturler cekiliyor...\n")
    
    # Bugünün fikstürlerini çek
    fixtures = client.get_fixtures(date_from=today, date_to=today)
    
    if fixtures and len(fixtures) > 0:
        print(f"[OK] {len(fixtures)} fikstur bulundu!\n")
        print("-" * 80)
        
        # Fikstürleri göster
        for i, fixture in enumerate(fixtures, 1):
            teams = fixture.get("teams", {})
            home = teams.get("home", {}).get("name", "N/A")
            away = teams.get("away", {}).get("name", "N/A")
            
            fixture_data = fixture.get("fixture", {})
            status = fixture_data.get("status", {})
            status_short = status.get("short", "NS")  # NS = Not Started
            status_long = status.get("long", "Not Started")
            
            # Tarih ve saat
            fixture_date = fixture_data.get("date")
            if fixture_date:
                try:
                    dt = datetime.fromisoformat(fixture_date.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M")
                except:
                    time_str = "N/A"
            else:
                time_str = "N/A"
            
            # Skor
            goals = fixture_data.get("goals", {})
            home_score = goals.get("home")
            away_score = goals.get("away")
            
            if home_score is not None and away_score is not None:
                score_str = f"{home_score}-{away_score}"
            else:
                score_str = "-"
            
            # Lig bilgisi
            league = fixture.get("league", {})
            league_name = league.get("name", "N/A")
            country = league.get("country", "N/A")
            
            print(f"{i}. {home} vs {away}")
            print(f"   Lig: {league_name} ({country})")
            print(f"   Saat: {time_str}")
            print(f"   Durum: {status_long} ({status_short})")
            if score_str != "-":
                print(f"   Skor: {score_str}")
            print()
        
        print("-" * 80)
        print(f"\nToplam: {len(fixtures)} mac bugun oynanacak/oynaniyor/oynandi")
        
    else:
        print("[UYARI] Bugun icin fikstur bulunamadi!")
        print("\nBu normal olabilir cunku:")
        print("  - Bugun mac olmayabilir")
        print("  - Free plan'da bugunun tarihi icin erisim kisitli olabilir")
        print("  - API limiti az kaldi (5/100)")
        
        # Alternatif: Geçmiş bir tarihten örnek fikstürler göster
        print("\n" + "-" * 80)
        print("ORNEK FIKSTURLER (Gecmis tarih - Free plan erisilebilir)")
        print("-" * 80)
        from datetime import timedelta
        # Free plan için erişilebilir bir tarih (2025-11-28)
        example_date = date(2025, 11, 28)
        print(f"\nTarih: {example_date.strftime('%Y-%m-%d')} (Ornek)")
        
        try:
            fixtures_example = client.get_fixtures(date_from=example_date, date_to=example_date)
            if fixtures_example and len(fixtures_example) > 0:
                print(f"[OK] {len(fixtures_example)} fikstur bulundu!\n")
                for i, fixture in enumerate(fixtures_example[:10], 1):  # İlk 10
                    teams = fixture.get("teams", {})
                    home = teams.get("home", {}).get("name", "N/A")
                    away = teams.get("away", {}).get("name", "N/A")
                    league = fixture.get("league", {})
                    league_name = league.get("name", "N/A")
                    country = league.get("country", "N/A")
                    
                    # Skor
                    fixture_data = fixture.get("fixture", {})
                    goals = fixture_data.get("goals", {})
                    home_score = goals.get("home")
                    away_score = goals.get("away")
                    
                    if home_score is not None and away_score is not None:
                        score_str = f" | Skor: {home_score}-{away_score}"
                    else:
                        score_str = ""
                    
                    print(f"  {i}. {home} vs {away}{score_str}")
                    print(f"      Lig: {league_name} ({country})")
                if len(fixtures_example) > 10:
                    print(f"\n  ... ve {len(fixtures_example) - 10} mac daha")
                print(f"\n[NOT] Bu ornek fiksturlerdir. Bugun icin fikstur yoksa,")
                print("      yarin veya gelecek hafta icin kontrol edebilirsiniz.")
            else:
                print("[UYARI] Ornek tarih icin de fikstur bulunamadi")
        except Exception as e:
            print(f"[HATA] {e}")
    
    # API Limit durumu
    remaining = client.daily_limit - client.requests_today
    print("\n" + "=" * 80)
    print("API LIMIT DURUMU")
    print("=" * 80)
    print(f"Kullanilan: {client.requests_today}/{client.daily_limit}")
    print(f"Kalan: {remaining}/{client.daily_limit}")
    if remaining <= 10:
        print(f"[UYARI] Limit azaliyor! {remaining} kaldi.")
    
except Exception as e:
    print(f"\n[HATA] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

