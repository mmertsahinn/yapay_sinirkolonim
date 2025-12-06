"""
Eğitim maçlarında (2020-2022) odds verisi kontrolü
"""
import sys
import io
from pathlib import Path

# Windows encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract
from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League

session = get_session()

try:
    # 2020-2022 arası eğitim maçları
    training_matches = session.query(Match).filter(
        and_(
            Match.home_score.isnot(None),
            Match.away_score.isnot(None),
            extract('year', Match.match_date) >= 2020,
            extract('year', Match.match_date) <= 2022
        )
    ).all()
    
    total_training = len(training_matches)
    
    # Odds'ı olan eğitim maçları
    matches_with_odds = session.query(Match).join(MatchOdds).filter(
        and_(
            Match.home_score.isnot(None),
            Match.away_score.isnot(None),
            extract('year', Match.match_date) >= 2020,
            extract('year', Match.match_date) <= 2022
        )
    ).all()
    
    matches_with_odds_count = len(matches_with_odds)
    
    print("=" * 80)
    print("📊 EĞİTİM MAÇLARI (2020-2022) ODDS DURUMU")
    print("=" * 80)
    print(f"\n✅ Toplam eğitim maçı: {total_training:,}")
    print(f"🎲 Odds'ı olan maç: {matches_with_odds_count:,}")
    print(f"❌ Odds'ı olmayan maç: {total_training - matches_with_odds_count:,}")
    
    if total_training > 0:
        percentage = (matches_with_odds_count / total_training) * 100
        print(f"📈 Yüzde: {percentage:.1f}%")
    
    # Lig bazında detay
    print("\n" + "=" * 80)
    print("📋 LİG BAZINDA EĞİTİM MAÇLARI ODDS DURUMU")
    print("=" * 80)
    
    leagues = session.query(League).all()
    league_stats = []
    
    for league in leagues:
        total = session.query(Match).filter(
            and_(
                Match.league_id == league.id,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2022
            )
        ).count()
        
        with_odds = session.query(Match).join(MatchOdds).filter(
            and_(
                Match.league_id == league.id,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2022
            )
        ).count()
        
        if total > 0:
            pct = (with_odds / total) * 100
            league_stats.append({
                'name': league.name,
                'total': total,
                'with_odds': with_odds,
                'without_odds': total - with_odds,
                'percentage': pct
            })
    
    # Sırala (toplam maça göre)
    league_stats.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n{'Lig':<30} {'Toplam':<10} {'Odds Var':<12} {'Odds Yok':<12} {'%':<8}")
    print("-" * 80)
    
    for stat in league_stats:
        status = "✅" if stat['percentage'] >= 50 else "⚠️" if stat['percentage'] > 0 else "❌"
        print(f"{status} {stat['name']:<28} {stat['total']:<10} {stat['with_odds']:<12} "
              f"{stat['without_odds']:<12} {stat['percentage']:.1f}%")
    
    # Yıl bazında detay
    print("\n" + "=" * 80)
    print("📅 YIL BAZINDA EĞİTİM MAÇLARI ODDS DURUMU")
    print("=" * 80)
    
    for year in [2020, 2021, 2022]:
        total = session.query(Match).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) == year
            )
        ).count()
        
        with_odds = session.query(Match).join(MatchOdds).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) == year
            )
        ).count()
        
        if total > 0:
            pct = (with_odds / total) * 100
            status = "✅" if pct >= 50 else "⚠️" if pct > 0 else "❌"
            print(f"{status} {year}: {with_odds:,}/{total:,} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    
finally:
    session.close()

