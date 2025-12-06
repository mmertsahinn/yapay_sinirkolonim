"""2025-11-03 maçlarını kontrol et"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.schema import Match
from sqlalchemy import extract

session = get_session()
try:
    # 2025-11-03 maçları
    nov3 = session.query(Match).filter(
        extract('year', Match.match_date) == 2025,
        extract('month', Match.match_date) == 11,
        extract('day', Match.match_date) == 3
    ).all()
    
    print(f"2025-11-03 maç sayısı: {len(nov3)}")
    
    if len(nov3) > 0:
        print("\n2025-11-03 maçları:")
        for m in nov3:
            home = m.home_team.name if m.home_team else "N/A"
            away = m.away_team.name if m.away_team else "N/A"
            league = m.league.name if m.league else "N/A"
            print(f"  {m.match_date.strftime('%Y-%m-%d %H:%M')} - {league}: {home} vs {away}")
    else:
        print("2025-11-03 maçları bulunamadı!")
        
        # Türkiye ligindeki Kasım maçlarını kontrol et
        from src.db.repositories import LeagueRepository
        turkish_league = LeagueRepository.get_by_name(session, "Süper Lig")
        if turkish_league:
            nov_matches = session.query(Match).filter(
                Match.league_id == turkish_league.id,
                extract('year', Match.match_date) == 2025,
                extract('month', Match.match_date) == 11
            ).order_by(Match.match_date).all()
            
            print(f"\nTürkiye Süper Lig - 2025 Kasım maçları: {len(nov_matches)}")
            for m in nov_matches[:10]:
                home = m.home_team.name if m.home_team else "N/A"
                away = m.away_team.name if m.away_team else "N/A"
                print(f"  {m.match_date.strftime('%Y-%m-%d %H:%M')} - {home} vs {away}")
        
finally:
    session.close()






