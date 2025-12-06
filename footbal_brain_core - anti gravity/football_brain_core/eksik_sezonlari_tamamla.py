"""
Eksik sezonları tamamla - 2022, 2023, 2024
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime, date

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# YENİ API KEY
os.environ["API_FOOTBALL_KEY"] = "81cf96e9b61dfdcef9ed54dc8c1ad772"

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.ingestion.historical_loader import HistoricalLoader
from src.config import Config
from src.db.connection import get_session
from src.db.schema import Match, League

print("=" * 80)
print("EKSIK SEZONLARI TAMAMLA - 2021, 2022, 2023, 2024")
print("TOPLAM 6,126 EKSIK MAC VAR - HEPSI TAMAMLANACAK!")
print("=" * 80)
print(f"Baslangic zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"API Key: {os.environ['API_FOOTBALL_KEY'][:20]}...")
print()

config = Config()
loader = HistoricalLoader()
session = get_session()

# Eksik sezonlar listesi - TÜM EKSİKLER (2021-2024)
eksik_sezonlar = [
    # 2021 sezonu
    ("Bundesliga", 2021),  # 306 mac eksik
    ("Ligue 1", 2021),  # 380 mac eksik
    ("Liga Portugal", 2021),  # 306 mac eksik
    ("Süper Lig", 2021),  # 380 mac eksik
    
    # 2022 sezonu
    ("Premier League", 2022),  # 380 mac eksik
    ("La Liga", 2022),  # 380 mac eksik
    ("Serie A", 2022),  # 380 mac eksik
    ("Bundesliga", 2022),  # 306 mac eksik
    ("Süper Lig", 2022),  # 38 mac eksik (342 mevcut)
    
    # 2023 sezonu
    ("Ligue 1", 2023),  # 72 mac eksik (308 mevcut)
    ("Liga Portugal", 2023),  # 306 mac eksik
    ("Süper Lig", 2023),  # 380 mac eksik
    
    # 2024 sezonu
    ("Premier League", 2024),  # 380 mac eksik
    ("La Liga", 2024),  # 380 mac eksik
    ("Serie A", 2024),  # 380 mac eksik
    ("Bundesliga", 2024),  # 306 mac eksik
    ("Ligue 1", 2024),  # 380 mac eksik
    ("Liga Portugal", 2024),  # 306 mac eksik
    ("Süper Lig", 2024),  # 380 mac eksik
]

total_operations = len(eksik_sezonlar)
print(f"Toplam eksik sezon/lig kombinasyonu: {total_operations}")
print("=" * 80)
print()

operation_count = 0
total_new_matches = 0

try:
    for league_name, season in eksik_sezonlar:
        operation_count += 1
        
        print(f"\n[{operation_count}/{total_operations}] {league_name} - Sezon {season}")
        print("-" * 80)
        
        # Mevcut maç sayısını kontrol et
        league_db = session.query(League).filter(League.name == league_name).first()
        if league_db:
            season_start = datetime(season, 8, 1)
            season_end = datetime(season + 1, 7, 31)
            before_count = session.query(Match).filter(
                Match.league_id == league_db.id,
                Match.match_date >= season_start,
                Match.match_date <= season_end
            ).count()
            print(f"  Mevcut mac sayisi: {before_count}")
        else:
            before_count = 0
        
        try:
            # Ligleri yükle
            loader.load_leagues(season)
            
            # Takımları yükle
            print(f"  Takimlar yukleniyor...")
            loader.load_teams_for_league(league_name, season)
            print(f"  [OK] Takimlar yuklendi")
            
            # Maçları yükle (tam sezon)
            print(f"  Maclar yukleniyor...")
            loader.load_matches_for_league(league_name, season)
            print(f"  [OK] Maclar yuklendi")
            
            # Yeni maç sayısını kontrol et
            if league_db:
                after_count = session.query(Match).filter(
                    Match.league_id == league_db.id,
                    Match.match_date >= season_start,
                    Match.match_date <= season_end
                ).count()
                new_matches = after_count - before_count
                total_new_matches += new_matches
                print(f"  Yeni eklenen mac: {new_matches}")
                print(f"  Toplam mac: {after_count}")
            else:
                print(f"  [UYARI] Lig veritabaninda bulunamadi, sayim yapilamadi")
            
            # API limit bilgisi
            if hasattr(loader.api_client, 'requests_today'):
                remaining = loader.api_client.daily_limit - loader.api_client.requests_today
                if remaining <= 0:
                    print(f"  [UYARI] API LIMITI DOLDU! {remaining} kalan")
                    print(f"  [UYARI] Yeni API key gerekiyor!")
                    print(f"\n  Durduruldu! {operation_count}/{total_operations} tamamlandi.")
                    break
                elif remaining < 10:
                    print(f"  [UYARI] API limiti yaklasiyor: {remaining} kalan")
                else:
                    print(f"  [INFO] API requests kalan: {remaining}/{loader.api_client.daily_limit}")
            
        except Exception as e:
            print(f"  [HATA] {league_name} {season} yukleme hatasi: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Kısa bir bekleme (rate limit için)
        time.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("EKSIK SEZONLAR TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitis zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Toplam islem: {operation_count}/{total_operations}")
    print(f"Yeni eklenen mac: {total_new_matches}")
    
    # Son durumu göster
    final_total = session.query(Match).count()
    print(f"\nToplam mac sayisi: {final_total}")
    
    # Sezon bazında özet
    print("\nSezon bazinda ozet:")
    for season in [2022, 2023, 2024]:
        season_start = datetime(season, 8, 1)
        season_end = datetime(season + 1, 7, 31)
        season_count = session.query(Match).filter(
            Match.match_date >= season_start,
            Match.match_date <= season_end
        ).count()
        print(f"  {season}: {season_count} mac")
    
except KeyboardInterrupt:
    print("\n\n[UYARI] Kullanici tarafindan durduruldu!")
    print(f"Toplam islem: {operation_count}/{total_operations}")
    print(f"Yeni mac: {total_new_matches}")
except Exception as e:
    print(f"\n\n[HATA] Genel hata: {e}")
    import traceback
    traceback.print_exc()
    print(f"Toplam islem: {operation_count}/{total_operations}")

finally:
    session.close()

print("\n" + "=" * 80)

