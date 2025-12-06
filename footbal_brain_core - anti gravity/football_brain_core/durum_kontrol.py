"""Mevcut durumu kontrol et"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.schema import Match, League
from datetime import datetime

session = get_session()

try:
    # Toplam maç
    total = session.query(Match).count()
    print(f"Toplam mac: {total}")
    
    # Son maç
    last = session.query(Match).order_by(Match.match_date.desc()).first()
    if last:
        print(f"Son mac tarihi: {last.match_date.strftime('%Y-%m-%d')}")
        print(f"Son mac: {last.home_team.name if last.home_team else 'N/A'} vs {last.away_team.name if last.away_team else 'N/A'}")
    
    # Lig bazında
    print("\nLig bazinda mac sayilari:")
    leagues = session.query(League).all()
    for l in leagues:
        count = session.query(Match).filter(Match.league_id == l.id).count()
        print(f"  {l.name}: {count}")
    
    # 2024 sezonu son maçlar
    print("\n2024 sezonu son 5 mac:")
    matches_2024 = session.query(Match).filter(
        Match.match_date >= datetime(2024, 1, 1),
        Match.match_date < datetime(2025, 1, 1)
    ).order_by(Match.match_date.desc()).limit(5).all()
    
    for m in matches_2024:
        home = m.home_team.name if m.home_team else "N/A"
        away = m.away_team.name if m.away_team else "N/A"
        print(f"  {m.match_date.strftime('%Y-%m-%d')} - {home} vs {away}")
    
finally:
    session.close()






