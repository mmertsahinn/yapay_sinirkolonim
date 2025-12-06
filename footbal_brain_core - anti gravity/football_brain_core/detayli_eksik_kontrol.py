"""2021-2024 sezonlarında eksik maç sayısını detaylı kontrol et"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.schema import Match, League
from src.config import Config

print("=" * 80)
print("DETAYLI EKSIK MAC KONTROLU - 2021, 2022, 2023, 2024")
print("=" * 80)

config = Config()
session = get_session()

# Beklenen maç sayıları (her lig için)
expected_matches = {
    "Premier League": 380,  # 20 takım x 19 maç x 2 (ev-deplasman) = 380
    "La Liga": 380,
    "Serie A": 380,
    "Bundesliga": 306,  # 18 takım x 17 maç x 2 = 306
    "Ligue 1": 380,
    "Liga Portugal": 306,  # 18 takım
    "Süper Lig": 380,  # 20 takım
}

seasons = [2021, 2022, 2023, 2024]
eksik_listesi = []

print("\nLig ve sezon bazinda detayli kontrol:\n")

for league_config in config.TARGET_LEAGUES:
    league_name = league_config.name
    league_db = session.query(League).filter(League.name == league_name).first()
    
    if not league_db:
        print(f"{league_name}: [HATA] Lig bulunamadi")
        continue
    
    expected = expected_matches.get(league_name, 380)
    print(f"{league_name} (Beklenen: ~{expected} mac/sezon):")
    
    for season in seasons:
        season_start = datetime(season, 8, 1)
        season_end = datetime(season + 1, 7, 31)
        
        db_count = session.query(Match).filter(
            Match.league_id == league_db.id,
            Match.match_date >= season_start,
            Match.match_date <= season_end
        ).count()
        
        eksik = max(0, expected - db_count)
        
        if db_count == 0:
            status = f"[TAMAMEN EKSIK - {expected} mac]"
            eksik_listesi.append((league_name, season, expected, 0))
        elif eksik > 0:
            status = f"[KISMEN EKSIK - {eksik} mac eksik]"
            eksik_listesi.append((league_name, season, eksik, db_count))
        else:
            status = "[TAM]"
        
        print(f"  {season}: {db_count}/{expected} mac {status}")
    
    print()

print("=" * 80)
print("OZET - EKSIK MACLAR")
print("=" * 80)

if eksik_listesi:
    toplam_eksik = sum(eksik for _, _, eksik, _ in eksik_listesi)
    print(f"Toplam eksik mac: {toplam_eksik}")
    print(f"Eksik sezon/lig kombinasyonu: {len(eksik_listesi)}")
    print("\nDetayli liste:")
    for league_name, season, eksik, mevcut in eksik_listesi:
        print(f"  - {league_name} {season}: {eksik} mac eksik (Mevcut: {mevcut})")
else:
    print("[OK] Eksik mac yok! Tum sezonlar tam!")

session.close()

print("\n" + "=" * 80)






