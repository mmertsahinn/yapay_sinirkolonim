"""
Tüm verileri sırayla yükle - Optimize edilmiş versiyon
Her sezon ve lig için düzenli yükleme
"""
import sys
from pathlib import Path
import time
from datetime import datetime

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.ingestion.historical_loader import HistoricalLoader
from football_brain_core.src.config import Config

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("TUM VERILERI SIRAYLA YUKLEME BASLATILIYOR")
    print("=" * 60)
    print(f"Baslangic zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    print("=" * 60)
    print()
    
    operation_count = 0
    
    try:
        for season_idx, season in enumerate(seasons, 1):
            print(f"\n{'='*60}")
            print(f"SEZON {season} ({season_idx}/{total_seasons})")
            print(f"{'='*60}")
            
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
                print("-" * 60)
                
                try:
                    # Takımları yükle
                    loader.load_teams_for_league(league_name, season)
                    print(f"  [OK] Takimlar yuklendi")
                    
                    # Maçları yükle
                    loader.load_matches_for_league(league_name, season)
                    print(f"  [OK] Maclar yuklendi")
                    
                    # API limit bilgisi
                    if hasattr(loader.api_client, 'requests_today'):
                        remaining = loader.api_client.daily_limit - loader.api_client.requests_today
                        if remaining < 10:
                            print(f"  [UYARI] API limiti yaklasiyor: {remaining} kalan")
                        else:
                            print(f"  [INFO] API requests kalan: {remaining}/{loader.api_client.daily_limit}")
                    
                except Exception as e:
                    print(f"  [HATA] {league_name} yukleme hatasi: {e}")
                    # Hata olsa bile devam et
                    continue
                
                # Kısa bir bekleme (rate limit için)
                time.sleep(0.5)
            
            print(f"\n[OK] Sezon {season} tamamlandi!")
            print()
        
        print("=" * 60)
        print("TUM VERI YUKLEME TAMAMLANDI!")
        print("=" * 60)
        print(f"Bitis zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n[UYARI] Kullanici tarafindan durduruldu!")
        print(f"Toplam islem: {operation_count}/{total_operations}")
    except Exception as e:
        print(f"\n\n[HATA] Genel hata: {e}")
        print(f"Toplam islem: {operation_count}/{total_operations}")







