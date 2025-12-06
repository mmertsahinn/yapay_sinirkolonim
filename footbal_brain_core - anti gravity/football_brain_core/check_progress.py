"""
Veri yükleme ilerlemesini kontrol et
"""
import sys
from pathlib import Path

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match, Team, League

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    session = get_session()
    try:
        match_count = session.query(Match).count()
        team_count = session.query(Team).count()
        league_count = session.query(League).count()
        
        # Tahmini toplam (5 sezon x 7 lig x ~380 maç = ~13,300)
        estimated_total = 5 * 7 * 380  # ~13,300 maç
        progress = (match_count / estimated_total) * 100 if estimated_total > 0 else 0
        
        print("=" * 60)
        print("VERI YUKLEME ILERLEMESI")
        print("=" * 60)
        print(f"Ligler: {league_count}")
        print(f"Takimlar: {team_count}")
        print(f"Maclar: {match_count}")
        print(f"Ilerleme: {progress:.1f}%")
        print("=" * 60)
        
        if match_count > 0:
            # Son yüklenen maçları göster
            from datetime import datetime
            recent_matches = session.query(Match).order_by(Match.created_at.desc()).limit(5).all()
            print("\nSon yuklenen maclar:")
            for match in recent_matches:
                print(f"  - {match.match_date.date()} (ID: {match.id})")
        
    finally:
        session.close()







