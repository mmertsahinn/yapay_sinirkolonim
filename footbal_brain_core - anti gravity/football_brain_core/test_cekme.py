"""Test çekme - neden değişiklik yok?"""
import os
import sys
from pathlib import Path
from datetime import date, datetime

os.environ["API_FOOTBALL_KEY"] = "81cf96e9b61dfdcef9ed54dc8c1ad772"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.historical_loader import HistoricalLoader
from src.config import Config

print("Test basladi...")
print(f"API Key: {os.environ['API_FOOTBALL_KEY'][:20]}...")

loader = HistoricalLoader()
config = Config()

# Test: Premier League 2024, 2024-06-02'den sonra
league_name = "Premier League"
season = 2024
marker_date = date(2024, 6, 2)

print(f"\nTest: {league_name} - Sezon {season}")
print(f"Tarih araligi: {marker_date} - 2025-07-31")

try:
    # Takımları yükle
    print("\nTakimlar yukleniyor...")
    loader.load_teams_for_league(league_name, season)
    print("[OK] Takimlar yuklendi")
    
    # Maçları yükle
    print("\nMaclar yukleniyor...")
    loader.load_matches_for_league(
        league_name,
        season,
        date_from=marker_date,
        date_to=date(2025, 7, 31)
    )
    print("[OK] Maclar yuklendi")
    
    # API limit
    remaining = loader.api_client.daily_limit - loader.api_client.requests_today
    print(f"\nAPI kalan: {remaining}/{loader.api_client.daily_limit}")
    
except Exception as e:
    print(f"[HATA] {e}")
    import traceback
    traceback.print_exc()






