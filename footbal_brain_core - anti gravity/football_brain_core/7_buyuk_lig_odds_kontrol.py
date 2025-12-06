"""
7 BÜYÜK LİG - ODDS DURUMU KONTROLÜ
Eğitim için kullanılan 7 büyük lig için odds durumunu gösterir
"""
import sys
from pathlib import Path

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League
from sqlalchemy import and_, extract

def check_7_big_leagues_odds():
    """7 büyük lig için odds durumunu kontrol eder"""
    session = get_session()
    
    # 7 büyük lig (config'den)
    big_leagues = [
        'Premier League',
        'La Liga',
        'Serie A',
        'Bundesliga',
        'Ligue 1',
        'Liga Portugal',
        'Süper Lig'
    ]
    
    print("=" * 70)
    print("7 BÜYÜK LİG - ODDS DURUMU (EĞİTİM İÇİN)")
    print("=" * 70)
    print()
    
    total_all = 0
    odds_all = 0
    
    league_stats = []
    
    for league_name in big_leagues:
        league = session.query(League).filter(League.name == league_name).first()
        
        if not league:
            print(f"⚠️  {league_name} bulunamadı!")
            continue
        
        # Toplam maç
        total = session.query(Match).filter(
            and_(
                Match.league_id == league.id,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        # Odds'ı olan maç
        with_odds = session.query(Match).join(MatchOdds).filter(
            and_(
                Match.league_id == league.id,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        without_odds = total - with_odds
        percentage = (with_odds / total * 100) if total > 0 else 0
        
        status = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"
        
        print(f"{status} {league_name:<20} {with_odds:>5}/{total:>5} ({percentage:>5.1f}%) - Eksik: {without_odds:>5}")
        
        total_all += total
        odds_all += with_odds
        
        league_stats.append({
            'league': league_name,
            'total': total,
            'with_odds': with_odds,
            'without_odds': without_odds,
            'percentage': percentage
        })
    
    print()
    print("=" * 70)
    overall_pct = (odds_all / total_all * 100) if total_all > 0 else 0
    print(f"📊 TOPLAM: {odds_all:>5}/{total_all:>5} ({overall_pct:>5.1f}%)")
    print(f"❌ Eksik: {total_all - odds_all:>5} maç")
    print("=" * 70)
    print()
    
    # Yıl bazında detay
    print("=" * 70)
    print("📅 YIL BAZINDA DURUM (7 BÜYÜK LİG)")
    print("=" * 70)
    
    for year in range(2020, 2026):
        year_total = 0
        year_odds = 0
        
        for league_name in big_leagues:
            league = session.query(League).filter(League.name == league_name).first()
            if not league:
                continue
            
            year_matches = session.query(Match).filter(
                and_(
                    Match.league_id == league.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) == year
                )
            ).count()
            
            year_matches_with_odds = session.query(Match).join(MatchOdds).filter(
                and_(
                    Match.league_id == league.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) == year
                )
            ).count()
            
            year_total += year_matches
            year_odds += year_matches_with_odds
        
        if year_total > 0:
            year_pct = (year_odds / year_total * 100)
            status = "✅" if year_pct >= 80 else "⚠️" if year_pct >= 50 else "❌"
            print(f"{status} {year}: {year_odds:>5}/{year_total:>5} ({year_pct:>5.1f}%)")
    
    print("=" * 70)
    
    session.close()

if __name__ == "__main__":
    check_7_big_leagues_odds()





