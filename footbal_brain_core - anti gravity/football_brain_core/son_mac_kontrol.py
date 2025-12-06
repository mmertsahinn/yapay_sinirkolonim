"""Son maç tarihini kontrol et"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match
from sqlalchemy import desc

session = get_session()
try:
    last_match = session.query(Match).order_by(desc(Match.match_date)).first()
    
    if last_match:
        print(f"Son mac tarihi: {last_match.match_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"Mac: {last_match.home_team.name if last_match.home_team else 'N/A'} vs {last_match.away_team.name if last_match.away_team else 'N/A'}")
        print(f"Skor: {last_match.home_score}-{last_match.away_score}")
        print(f"Lig: {last_match.league.name if last_match.league else 'N/A'}")
        
        # Son 5 maç
        print("\nSon 5 mac:")
        last_5 = session.query(Match).order_by(desc(Match.match_date)).limit(5).all()
        for m in last_5:
            print(f"  {m.match_date.strftime('%Y-%m-%d')} - {m.home_team.name if m.home_team else 'N/A'} vs {m.away_team.name if m.away_team else 'N/A'} ({m.home_score}-{m.away_score})")
    else:
        print("Mac bulunamadi")
finally:
    session.close()






