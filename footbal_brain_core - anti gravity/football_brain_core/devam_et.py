"""
Kaldığı yerden devam et - İşaretten sonraki maçları çek
"""
import sys
import os
import json
from pathlib import Path
import time
from datetime import datetime, date

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Yeni API key'i ayarla
os.environ["API_FOOTBALL_KEY"] = "81cf96e9b61dfdcef9ed54dc8c1ad772"

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.ingestion.historical_loader import HistoricalLoader
from football_brain_core.src.config import Config
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match

print("=" * 80)
print("KALDIĞI YERDEN DEVAM ET")
print("=" * 80)
print(f"Baslangic zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# İşaret dosyasını oku
marker_file = project_root / "loaded_matches_marker.json"
if not marker_file.exists():
    print("[HATA] Isaret dosyasi bulunamadi!")
    print("Once isaret koymak icin: python isaret_koy.py")
    sys.exit(1)

with open(marker_file, 'r', encoding='utf-8') as f:
    marker = json.load(f)

print("Isaret bilgileri:")
print(f"  Son mac tarihi: {marker['last_match_date']}")
print(f"  Son mac: {marker['last_match_home_team']} vs {marker['last_match_away_team']}")
print(f"  Isaret zamani: {marker['marked_at']}")
print()

# Mevcut durumu kontrol et
session = get_session()
try:
    current_count = session.query(Match).count()
    print(f"Mevcut mac sayisi: {current_count}")
    print(f"Isaretlenen mac sayisi: {marker['total_matches_marked']}")
    if current_count > marker['total_matches_marked']:
        print(f"[BILGI] {current_count - marker['total_matches_marked']} yeni mac eklendi.")
except:
    pass
finally:
    session.close()

print()
print("=" * 80)
print("YENI API KEY ILE CEKMEYE DEVAM EDILIYOR...")
print("=" * 80)
print()

config = Config()
loader = HistoricalLoader()

# Tüm sezonlar
seasons = [2021, 2022, 2023, 2024, 2025]

# Tüm ligler
leagues = config.TARGET_LEAGUES

total_seasons = len(seasons)
total_leagues = len(leagues)
total_operations = total_seasons * total_leagues

print(f"Toplam sezon: {total_seasons}")
print(f"Toplam lig: {total_leagues}")
print(f"Toplam islem: {total_operations}")
print("=" * 80)
print()

operation_count = 0

try:
    for season_idx, season in enumerate(seasons, 1):
        print(f"\n{'='*80}")
        print(f"SEZON {season} ({season_idx}/{total_seasons})")
        print(f"{'='*80}")
        
        # Ligleri yükle
        try:
            loader.load_leagues(season)
            print(f"[OK] Ligler yuklendi (Sezon {season})")
        except Exception as e:
            print(f"[HATA] Lig yukleme hatasi (Sezon {season}): {e}")
            continue
        
        # Her lig için veri yükle
        for league_idx, league_config in enumerate(leagues, 1):
            operation_count += 1
            league_name = league_config.name
            
            print(f"\n[{operation_count}/{total_operations}] {league_name} - Sezon {season}")
            print("-" * 80)
            
            try:
                # Takımları yükle
                loader.load_teams_for_league(league_name, season)
                print(f"  [OK] Takimlar yuklendi")
                
                # Maçları yükle (tarih sırasına göre, işaretten sonraki maçlar da gelecek)
                loader.load_matches_for_league(league_name, season)
                print(f"  [OK] Maclar yuklendi")
                
                # API limit bilgisi
                if hasattr(loader.api_client, 'requests_today'):
                    remaining = loader.api_client.daily_limit - loader.api_client.requests_today
                    if remaining <= 0:
                        print(f"  [UYARI] API LIMITI DOLDU! {remaining} kalan")
                        print(f"  [UYARI] Yeni API key gerekiyor!")
                        break
                    elif remaining < 10:
                        print(f"  [UYARI] API limiti yaklasiyor: {remaining} kalan")
                    else:
                        print(f"  [INFO] API requests kalan: {remaining}/{loader.api_client.daily_limit}")
                
            except Exception as e:
                print(f"  [HATA] {league_name} yukleme hatasi: {e}")
                import traceback
                traceback.print_exc()
                # Hata olsa bile devam et
                continue
            
            # Kısa bir bekleme (rate limit için)
            time.sleep(0.5)
        
        print(f"\n[OK] Sezon {season} tamamlandi!")
        print()
    
    print("=" * 80)
    print("VERI YUKLEME TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitis zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Son durumu göster
    session = get_session()
    try:
        final_count = session.query(Match).count()
        new_matches = final_count - marker['total_matches_marked']
        print(f"\nToplam mac sayisi: {final_count}")
        print(f"Yeni eklenen mac: {new_matches}")
    finally:
        session.close()
    
except KeyboardInterrupt:
    print("\n\n[UYARI] Kullanici tarafindan durduruldu!")
    print(f"Toplam islem: {operation_count}/{total_operations}")
except Exception as e:
    print(f"\n\n[HATA] Genel hata: {e}")
    import traceback
    traceback.print_exc()
    print(f"Toplam islem: {operation_count}/{total_operations}")

print("\n" + "=" * 80)






