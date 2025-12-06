"""Eksik maçları bul ve çek"""
import os
import sys
from pathlib import Path
from datetime import date, datetime

os.environ["API_FOOTBALL_KEY"] = "81cf96e9b61dfdcef9ed54dc8c1ad772"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.historical_loader import HistoricalLoader
from src.config import Config
from src.db.connection import get_session
from src.db.schema import Match, League

print("=" * 80)
print("EKSIK MACLARI BUL VE CEK")
print("=" * 80)

config = Config()
loader = HistoricalLoader()
session = get_session()

try:
    # Her lig ve sezon için kontrol et
    seasons = [2021, 2022, 2023, 2024]
    leagues = config.TARGET_LEAGUES
    
    print("\nLig ve sezon bazinda mac sayilari:\n")
    
    for league_config in leagues:
        league_name = league_config.name
        league_db = session.query(League).filter(League.name == league_name).first()
        
        if not league_db:
            print(f"{league_name}: Lig bulunamadi")
            continue
        
        print(f"{league_name}:")
        
        for season in seasons:
            # Veritabanındaki maç sayısı
            season_start = datetime(season, 8, 1)
            season_end = datetime(season + 1, 7, 31)
            
            db_count = session.query(Match).filter(
                Match.league_id == league_db.id,
                Match.match_date >= season_start,
                Match.match_date <= season_end
            ).count()
            
            # API'den çek (test)
            try:
                fixtures = loader.api_client.get_fixtures(
                    league_id=league_config.api_league_id,
                    season=season
                )
                api_count = len(fixtures) if fixtures else 0
            except:
                api_count = 0
            
            diff = api_count - db_count
            status = "OK" if diff <= 5 else "EKSIK"  # 5 fark normal (oynanmamış maçlar)
            
            print(f"  {season}: DB={db_count}, API={api_count}, Fark={diff} [{status}]")
            
            # Eksik varsa çek
            if diff > 5:
                print(f"    -> Eksik maclar cekiliyor...")
                try:
                    loader.load_matches_for_league(league_name, season)
                    new_count = session.query(Match).filter(
                        Match.league_id == league_db.id,
                        Match.match_date >= season_start,
                        Match.match_date <= season_end
                    ).count()
                    print(f"    -> Yeni toplam: {new_count}")
                except Exception as e:
                    print(f"    -> Hata: {e}")
        
        print()
    
    # Toplam
    total = session.query(Match).count()
    print(f"Toplam mac: {total}")
    
finally:
    session.close()






