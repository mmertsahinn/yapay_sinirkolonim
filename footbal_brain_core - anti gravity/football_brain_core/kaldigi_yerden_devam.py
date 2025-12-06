"""
Kaldığı yerden devam et - 2024-06-02'den sonraki maçları çek
Yeni API key ile
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

# YENİ API KEY
os.environ["API_FOOTBALL_KEY"] = "81cf96e9b61dfdcef9ed54dc8c1ad772"

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.ingestion.historical_loader import HistoricalLoader
from src.config import Config
from src.db.connection import get_session
from src.db.schema import Match

print("=" * 80)
print("KALDIĞI YERDEN DEVAM ET - YENİ API KEY İLE")
print("=" * 80)
print(f"Baslangic zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"API Key: {os.environ['API_FOOTBALL_KEY'][:20]}...")
print()

# İşaret dosyasını oku
marker_file = project_root / "loaded_matches_marker.json"
if not marker_file.exists():
    print("[HATA] Isaret dosyasi bulunamadi!")
    sys.exit(1)

with open(marker_file, 'r', encoding='utf-8') as f:
    marker = json.load(f)

marker_date = datetime.fromisoformat(marker['last_match_date']).date()
print(f"Isaret bilgileri:")
print(f"  Son mac tarihi: {marker_date}")
print(f"  Son mac: {marker['last_match_home_team']} vs {marker['last_match_away_team']}")
print()

# Mevcut durumu kontrol et
session = get_session()
try:
    current_count = session.query(Match).count()
    print(f"Mevcut mac sayisi: {current_count}")
    print(f"Isaretlenen mac sayisi: {marker['total_matches_marked']}")
except:
    pass
finally:
    session.close()

print()
print("=" * 80)
print("CEKMEYE BASLIYOR...")
print("=" * 80)
print()

config = Config()
loader = HistoricalLoader()

# Sezonlar - 2024'ten itibaren devam et
seasons = [2024, 2025]

# Tüm ligler
leagues = config.TARGET_LEAGUES

total_operations = len(seasons) * len(leagues)

print(f"Sezonlar: {seasons}")
print(f"Toplam lig: {len(leagues)}")
print(f"Toplam islem: {total_operations}")
print("=" * 80)
print()

operation_count = 0
total_new_matches = 0

try:
    for season_idx, season in enumerate(seasons, 1):
        print(f"\n{'='*80}")
        print(f"SEZON {season} ({season_idx}/{len(seasons)})")
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
                
                # Maçları yükle - İşaret tarihinden sonraki maçlar için
                # Sezon 2024 ise ve işaret 2024-06-02 ise, o tarihten sonrasını çek
                if season == 2024:
                    # 2024-06-02'den sonraki maçları çek
                    date_from = marker_date
                    date_to = date(season + 1, 7, 31)
                    print(f"  [INFO] Tarih araligi: {date_from} - {date_to}")
                else:
                    # 2025 sezonu için tam sezon
                    date_from = None
                    date_to = None
                
                # Maç sayısını kontrol et (önce)
                session = get_session()
                try:
                    before_count = session.query(Match).count()
                finally:
                    session.close()
                
                # Maçları yükle
                loader.load_matches_for_league(
                    league_name, 
                    season,
                    date_from=date_from if date_from else None,
                    date_to=date_to if date_to else None
                )
                
                # Yeni maç sayısını kontrol et (sonra)
                session = get_session()
                try:
                    after_count = session.query(Match).count()
                    new_matches = after_count - before_count
                    total_new_matches += new_matches
                    print(f"  [OK] Maclar yuklendi ({new_matches} yeni mac)")
                finally:
                    session.close()
                
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
        new_matches_total = final_count - marker['total_matches_marked']
        print(f"\nToplam mac sayisi: {final_count}")
        print(f"Yeni eklenen mac: {new_matches_total}")
        
        # Son maç
        last_match = session.query(Match).order_by(Match.match_date.desc()).first()
        if last_match:
            print(f"Son mac tarihi: {last_match.match_date.strftime('%Y-%m-%d')}")
            print(f"Son mac: {last_match.home_team.name if last_match.home_team else 'N/A'} vs {last_match.away_team.name if last_match.away_team else 'N/A'}")
    finally:
        session.close()
    
except KeyboardInterrupt:
    print("\n\n[UYARI] Kullanici tarafindan durduruldu!")
    print(f"Toplam islem: {operation_count}/{total_operations}")
    print(f"Yeni mac: {total_new_matches}")
except Exception as e:
    print(f"\n\n[HATA] Genel hata: {e}")
    import traceback
    traceback.print_exc()
    print(f"Toplam islem: {operation_count}/{total_operations}")

print("\n" + "=" * 80)

