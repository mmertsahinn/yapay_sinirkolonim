"""
API-FOOTBALL bağlantısını test etmek için basit script
"""
import os
import sys

# API key'i buraya yapıştır veya ortam değişkeninden al
API_KEY = os.getenv("API_FOOTBALL_KEY", "647f5de88a29d150a9d4e2c0c7b636fb")

if not API_KEY:
    print("❌ API_FOOTBALL_KEY ortam değişkeni ayarlanmamış!")
    sys.exit(1)

print(f"🔑 API Key kullanılıyor: {API_KEY[:10]}...")

try:
    from football_brain_core.src.ingestion.api_client import APIFootballClient
    
    client = APIFootballClient(api_key=API_KEY)
    
    print("\n📡 API-FOOTBALL bağlantısı test ediliyor...")
    
    # Basit bir test: Bugünün fikstürlerini çek
    from datetime import date
    today = date.today()
    
    print(f"📅 Tarih: {today}")
    print("🔍 Fikstürler çekiliyor...\n")
    
    fixtures = client.get_fixtures(date_from=today, date_to=today)
    
    if fixtures:
        print(f"✅ Başarılı! {len(fixtures)} fikstür bulundu.\n")
        print("İlk 3 fikstür örneği:")
        for i, fixture in enumerate(fixtures[:3], 1):
            fixture_data = fixture.get("fixture", {})
            teams = fixture.get("teams", {})
            home = teams.get("home", {}).get("name", "N/A")
            away = teams.get("away", {}).get("name", "N/A")
            print(f"  {i}. {home} vs {away}")
    else:
        print("⚠️  Bugün için fikstür bulunamadı (normal olabilir)")
        print("✅ Ancak API bağlantısı çalışıyor!")
    
    print("\n✅ API testi başarılı! Projeyi kullanmaya hazırsın.")
    
except ValueError as e:
    print(f"❌ Hata: {e}")
    print("\n💡 Çözüm:")
    print("   Ortam değişkenini ayarla:")
    print(f'   $env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"')
    sys.exit(1)
except Exception as e:
    print(f"❌ Beklenmeyen hata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)







