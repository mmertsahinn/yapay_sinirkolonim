"""2025 Kasım maçlarını kontrol et"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.schema import Match
from sqlalchemy import func, extract

session = get_session()
try:
    # 2025 Kasım maçları
    nov_2025 = session.query(Match).filter(
        extract('year', Match.match_date) == 2025,
        extract('month', Match.match_date) == 11
    ).count()
    
    print(f"2025 Kasım maç sayısı: {nov_2025}")
    
    if nov_2025 == 0:
        # En yeni maç tarihi
        max_date = session.query(func.max(Match.match_date)).scalar()
        print(f"\nEn yeni maç tarihi: {max_date}")
        
        # 2025 yılındaki tüm maçlar
        matches_2025 = session.query(Match).filter(
            extract('year', Match.match_date) == 2025
        ).order_by(Match.match_date.desc()).limit(10).all()
        
        print(f"\n2025 yılındaki son 10 maç:")
        for m in matches_2025:
            print(f"  {m.match_date.strftime('%Y-%m-%d')} - {m.home_team.name if m.home_team else 'N/A'} vs {m.away_team.name if m.away_team else 'N/A'}")
    else:
        # Kasım maçlarını göster
        nov_matches = session.query(Match).filter(
            extract('year', Match.match_date) == 2025,
            extract('month', Match.match_date) == 11
        ).order_by(Match.match_date.desc()).limit(10).all()
        
        print(f"\n2025 Kasım son 10 maç:")
        for m in nov_matches:
            print(f"  {m.match_date.strftime('%Y-%m-%d')} - {m.home_team.name if m.home_team else 'N/A'} vs {m.away_team.name if m.away_team else 'N/A'}")
        
finally:
    session.close()






