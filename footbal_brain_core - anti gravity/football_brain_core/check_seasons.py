"""
Veritabanında hangi sezonların verileri var kontrol et
"""
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match, League

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    session = get_session()
    try:
        # Tüm maçları al
        matches = session.query(Match).all()
        
        print("=" * 60)
        print("VERITABANI DURUMU")
        print("=" * 60)
        
        if not matches:
            print("Henuz mac verisi yok!")
            session.close()
            exit(0)
        
        # Yıllara göre grupla
        years = Counter()
        seasons = Counter()  # Sezon bazlı (2021-2022 gibi)
        leagues = Counter()
        
        for match in matches:
            year = match.match_date.year
            years[year] += 1
            
            # Sezon belirleme (Ağustos'tan Temmuz'a)
            if match.match_date.month >= 8:
                season = f"{year}-{year+1}"
            else:
                season = f"{year-1}-{year}"
            seasons[season] += 1
            
            # Lig bilgisi
            league = session.query(League).filter(League.id == match.league_id).first()
            if league:
                leagues[league.name] += 1
        
        print(f"\nTOPLAM MAC: {len(matches)}")
        
        print("\n" + "-" * 60)
        print("YILLARA GORE DAGILIM:")
        print("-" * 60)
        for year in sorted(years.keys()):
            print(f"  {year}: {years[year]} mac")
        
        print("\n" + "-" * 60)
        print("SEZONLARA GORE DAGILIM:")
        print("-" * 60)
        for season in sorted(seasons.keys()):
            print(f"  {season}: {seasons[season]} mac")
        
        print("\n" + "-" * 60)
        print("LIGLERE GORE DAGILIM:")
        print("-" * 60)
        for league_name in sorted(leagues.keys()):
            print(f"  {league_name}: {leagues[league_name]} mac")
        
        # En son yüklenen maçlar
        print("\n" + "-" * 60)
        print("EN SON YUKLENEN 10 MAC:")
        print("-" * 60)
        recent = sorted(matches, key=lambda m: m.created_at, reverse=True)[:10]
        for match in recent:
            league = session.query(League).filter(League.id == match.league_id).first()
            league_name = league.name if league else "N/A"
            print(f"  {match.match_date.date()} - {league_name} (ID: {match.id})")
        
        # Eksik sezonlar
        print("\n" + "-" * 60)
        print("HEDEF SEZONLAR: 2021, 2022, 2023, 2024, 2025")
        print("-" * 60)
        target_seasons = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
        missing = []
        for target in target_seasons:
            if target not in seasons:
                missing.append(target)
            else:
                print(f"  [OK] {target}: {seasons[target]} mac")
        
        if missing:
            print(f"\n  [EKSIK] Sezonlar: {', '.join(missing)}")
        
        print("\n" + "=" * 60)
        
    finally:
        session.close()







