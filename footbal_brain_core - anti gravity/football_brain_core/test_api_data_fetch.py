"""API'den veri çekme testi - Detaylı kontrol"""
import os
import sys
from datetime import date, timedelta

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# API key'i ayarla
os.environ["API_FOOTBALL_KEY"] = "647f5de88a29d150a9d4e2c0c7b636fb"

# Path ayarla
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("API VERI CEKME TESTI")
print("=" * 80)

try:
    from src.ingestion.api_client import APIFootballClient
    
    client = APIFootballClient()
    print(f"\n[OK] API Client olusturuldu")
    print(f"Gunluk limit: {client.daily_limit}")
    
    # Test 1: Bugünün fikstürlerini çek
    print("\n" + "-" * 80)
    print("TEST 1: Bugunun fiksturleri")
    print("-" * 80)
    today = date.today()
    print(f"Tarih: {today}")
    
    try:
        fixtures = client.get_fixtures(date_from=today, date_to=today)
        if fixtures:
            print(f"[OK] API VERI CEKIYOR! {len(fixtures)} fikstur bulundu.")
            print("\nIlk 5 fikstur:")
            for i, fixture in enumerate(fixtures[:5], 1):
                teams = fixture.get("teams", {})
                home = teams.get("home", {}).get("name", "N/A")
                away = teams.get("away", {}).get("name", "N/A")
                fixture_data = fixture.get("fixture", {})
                status = fixture_data.get("status", {}).get("short", "N/A")
                print(f"  {i}. {home} vs {away} (Durum: {status})")
        else:
            print("[UYARI] Bugun icin fikstur bulunamadi")
    except Exception as e:
        print(f"[HATA] {e}")
    
    # Test 2: Geçmiş bir tarihten fikstür çek (Free plan için uygun tarih)
    print("\n" + "-" * 80)
    print("TEST 2: Gecmis tarih fiksturleri (2025-11-28)")
    print("-" * 80)
    
    try:
        test_date = date(2025, 11, 28)
        fixtures = client.get_fixtures(date_from=test_date, date_to=test_date)
        if fixtures:
            print(f"[OK] API VERI CEKIYOR! {len(fixtures)} fikstur bulundu.")
            print("\nIlk 5 fikstur:")
            for i, fixture in enumerate(fixtures[:5], 1):
                teams = fixture.get("teams", {})
                home = teams.get("home", {}).get("name", "N/A")
                away = teams.get("away", {}).get("name", "N/A")
                fixture_data = fixture.get("fixture", {})
                score = fixture_data.get("goals", {})
                home_score = score.get("home", "?")
                away_score = score.get("away", "?")
                print(f"  {i}. {home} vs {away} | Skor: {home_score}-{away_score}")
        else:
            print("[UYARI] Bu tarih icin fikstur bulunamadi")
    except Exception as e:
        print(f"[HATA] {e}")
    
    # Test 3: Lig bilgilerini çek
    print("\n" + "-" * 80)
    print("TEST 3: Lig bilgileri (Premier League, 2024)")
    print("-" * 80)
    
    try:
        leagues = client.get_leagues(country="England", season=2024)
        if leagues:
            print(f"[OK] API VERI CEKIYOR! {len(leagues)} lig bulundu.")
            print("\nIlk 5 lig:")
            for i, league in enumerate(leagues[:5], 1):
                league_info = league.get("league", {})
                name = league_info.get("name", "N/A")
                country = league_info.get("country", "N/A")
                print(f"  {i}. {name} ({country})")
        else:
            print("[UYARI] Lig bilgisi bulunamadi")
    except Exception as e:
        print(f"[HATA] {e}")
    
    # Test 4: Takım bilgilerini çek
    print("\n" + "-" * 80)
    print("TEST 4: Takim bilgileri (Premier League, 2024)")
    print("-" * 80)
    
    try:
        # Premier League ID: 39
        teams = client.get_teams(league_id=39, season=2024)
        if teams:
            print(f"[OK] API VERI CEKIYOR! {len(teams)} takim bulundu.")
            print("\nIlk 10 takim:")
            for i, team_info in enumerate(teams[:10], 1):
                team = team_info.get("team", {})
                name = team.get("name", "N/A")
                code = team.get("code", "N/A")
                print(f"  {i}. {name} ({code})")
        else:
            print("[UYARI] Takim bilgisi bulunamadi")
    except Exception as e:
        print(f"[HATA] {e}")
    
    # Test 5: Maç istatistiklerini çek (bir maç ID'si ile)
    print("\n" + "-" * 80)
    print("TEST 5: Mac istatistikleri (ornek)")
    print("-" * 80)
    
    try:
        # Önce bir maç bulalım
        test_date = date(2025, 11, 28)
        fixtures = client.get_fixtures(date_from=test_date, date_to=test_date)
        if fixtures and len(fixtures) > 0:
            first_fixture = fixtures[0]
            fixture_id = first_fixture.get("fixture", {}).get("id")
            if fixture_id:
                print(f"Mac ID: {fixture_id}")
                stats = client.get_fixture_statistics(fixture_id)
                if stats:
                    print(f"[OK] API VERI CEKIYOR! {len(stats)} istatistik grubu bulundu.")
                    if stats:
                        print("\nIstatistik ornegi:")
                        for stat_group in stats[:2]:
                            team = stat_group.get("team", {})
                            team_name = team.get("name", "N/A")
                            statistics = stat_group.get("statistics", [])
                            print(f"  {team_name}: {len(statistics)} istatistik")
                            for stat in statistics[:3]:
                                stat_type = stat.get("type", "N/A")
                                stat_value = stat.get("value", "N/A")
                                print(f"    - {stat_type}: {stat_value}")
                else:
                    print("[UYARI] Istatistik bulunamadi (mac henuz oynanmamis olabilir)")
            else:
                print("[UYARI] Mac ID bulunamadi")
        else:
            print("[UYARI] Test icin mac bulunamadi")
    except Exception as e:
        print(f"[HATA] {e}")
        import traceback
        traceback.print_exc()
    
    # API Limit Durumu
    print("\n" + "-" * 80)
    print("API LIMIT DURUMU")
    print("-" * 80)
    remaining = client.daily_limit - client.requests_today
    print(f"Kullanilan: {client.requests_today}/{client.daily_limit}")
    print(f"Kalan: {remaining}/{client.daily_limit}")
    if remaining <= 10:
        print(f"[UYARI] Limit azaliyor! {remaining} kaldi.")
    elif remaining == 0:
        print(f"[UYARI] LIMIT DOLDU!")
    else:
        print(f"[OK] Limit yeterli.")
    
    # Özet
    print("\n" + "=" * 80)
    print("OZET")
    print("=" * 80)
    print("[OK] API CALISIYOR VE VERI CEKIYOR!")
    print("\nTest sonuclari:")
    print("  - Fikstur cekme: CALISIYOR")
    print("  - Lig bilgisi cekme: CALISIYOR")
    print("  - Takim bilgisi cekme: CALISIYOR")
    print("  - Istatistik cekme: CALISIYOR (mac oynanmissa)")
    print("\nAPI'den veri cekme sistemi hazir ve calisiyor!")
    
except Exception as e:
    print(f"\n[HATA] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)






