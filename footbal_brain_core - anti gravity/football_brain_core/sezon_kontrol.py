"""2022, 2023, 2024 sezonlarının tam yüklenip yüklenmediğini kontrol et"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.schema import Match, League
from src.config import Config

print("=" * 80)
print("SEZON KONTROLU - 2022, 2023, 2024")
print("=" * 80)

config = Config()
session = get_session()

try:
    seasons = [2022, 2023, 2024]
    leagues = config.TARGET_LEAGUES
    
    print("\nHer lig ve sezon icin mac sayilari:\n")
    
    total_all = 0
    eksik_sezonlar = []
    
    for league_config in leagues:
        league_name = league_config.name
        league_db = session.query(League).filter(League.name == league_name).first()
        
        if not league_db:
            print(f"{league_name}: [HATA] Lig bulunamadi")
            continue
        
        print(f"{league_name}:")
        
        for season in seasons:
            # Sezon tarih aralığı
            season_start = datetime(season, 8, 1)
            season_end = datetime(season + 1, 7, 31)
            
            # Veritabanındaki maç sayısı
            db_count = session.query(Match).filter(
                Match.league_id == league_db.id,
                Match.match_date >= season_start,
                Match.match_date <= season_end
            ).count()
            
            # Beklenen maç sayısı (her lig için ~380 maç normal, bazı liglerde daha az)
            # Premier League, La Liga, Serie A: ~380
            # Bundesliga: ~306 (18 takım)
            # Ligue 1: ~380
            # Liga Portugal: ~306
            # Süper Lig: ~380
            
            expected_ranges = {
                "Premier League": (370, 390),
                "La Liga": (370, 390),
                "Serie A": (370, 390),
                "Bundesliga": (300, 320),
                "Ligue 1": (370, 390),
                "Liga Portugal": (300, 320),
                "Süper Lig": (370, 390)
            }
            
            expected_min, expected_max = expected_ranges.get(league_name, (300, 400))
            
            # Durum kontrolü
            if db_count == 0:
                status = "[EKSIK]"
                eksik_sezonlar.append(f"{league_name} {season}")
            elif db_count < expected_min:
                status = f"[EKSIK - {db_count}/{expected_min}]"
                eksik_sezonlar.append(f"{league_name} {season} ({db_count}/{expected_min})")
            elif db_count >= expected_min:
                status = "[TAM]"
            else:
                status = "[KONTROL GEREKLI]"
            
            print(f"  {season}: {db_count} mac {status}")
            total_all += db_count
        
        print()
    
    print("=" * 80)
    print("OZET")
    print("=" * 80)
    print(f"Toplam mac (2022-2024): {total_all}")
    
    if eksik_sezonlar:
        print(f"\n[UYARI] Eksik sezonlar ({len(eksik_sezonlar)}):")
        for eksik in eksik_sezonlar:
            print(f"  - {eksik}")
    else:
        print("\n[OK] Tum sezonlar tam yuklenmis!")
    
    # Detaylı kontrol: Her sezon için ilk ve son maç tarihleri
    print("\n" + "=" * 80)
    print("DETAYLI TARIH KONTROLU")
    print("=" * 80)
    
    for season in seasons:
        print(f"\n{season} sezonu:")
        season_start = datetime(season, 8, 1)
        season_end = datetime(season + 1, 7, 31)
        
        first_match = session.query(Match).filter(
            Match.match_date >= season_start,
            Match.match_date <= season_end
        ).order_by(Match.match_date.asc()).first()
        
        last_match = session.query(Match).filter(
            Match.match_date >= season_start,
            Match.match_date <= season_end
        ).order_by(Match.match_date.desc()).first()
        
        if first_match and last_match:
            print(f"  Ilk mac: {first_match.match_date.strftime('%Y-%m-%d')}")
            print(f"  Son mac: {last_match.match_date.strftime('%Y-%m-%d')}")
            print(f"  Toplam: {session.query(Match).filter(Match.match_date >= season_start, Match.match_date <= season_end).count()} mac")
        else:
            print(f"  [EKSIK] Bu sezon icin mac bulunamadi!")
    
finally:
    session.close()

print("\n" + "=" * 80)






