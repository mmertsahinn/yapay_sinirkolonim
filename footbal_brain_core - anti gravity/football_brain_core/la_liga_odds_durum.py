"""
LA LIGA ODDS DURUM KONTROLÜ
- Kaç maçta odds var?
- Kaç maçta odds yok?
- Eksik maçların detayları
"""
import sys
from pathlib import Path
from datetime import datetime

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import LeagueRepository, TeamRepository

def check_la_liga_odds_status():
    """La Liga odds durumunu kontrol et"""
    session = get_session()
    
    try:
        la_liga = LeagueRepository.get_by_name(session, "La Liga")
        if not la_liga:
            print("❌ La Liga bulunamadı!")
            return
        
        print("=" * 80)
        print("LA LIGA ODDS DURUM RAPORU")
        print("=" * 80)
        print()
        
        # Genel durum
        total_matches = session.query(Match).filter(
            and_(
                Match.league_id == la_liga.id,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        matches_with_odds = session.query(Match).join(MatchOdds).filter(
            and_(
                Match.league_id == la_liga.id,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        matches_without_odds = total_matches - matches_with_odds
        percentage = (matches_with_odds / total_matches * 100) if total_matches > 0 else 0
        
        print("📊 GENEL DURUM")
        print("-" * 80)
        print(f"   Toplam maç (2020-2025): {total_matches:,}")
        print(f"   Odds olan maç: {matches_with_odds:,} ({percentage:.1f}%)")
        print(f"   Odds olmayan maç: {matches_without_odds:,} ({100-percentage:.1f}%)")
        print()
        
        # Yıl bazında durum
        print("=" * 80)
        print("📅 YIL BAZINDA DURUM")
        print("=" * 80)
        
        for year in range(2020, 2026):
            year_total = session.query(Match).filter(
                and_(
                    Match.league_id == la_liga.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) == year
                )
            ).count()
            
            year_with_odds = session.query(Match).join(MatchOdds).filter(
                and_(
                    Match.league_id == la_liga.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) == year
                )
            ).count()
            
            year_without_odds = year_total - year_with_odds
            year_pct = (year_with_odds / year_total * 100) if year_total > 0 else 0
            
            status = "✅" if year_pct >= 80 else "⚠️" if year_pct >= 50 else "❌"
            print(f"{status} {year}: {year_with_odds:>4}/{year_total:>4} ({year_pct:>5.1f}%) - Eksik: {year_without_odds:>4}")
        
        print()
        
        # Eksik maçların detayları (ilk 20)
        if matches_without_odds > 0:
            print("=" * 80)
            print("❌ ODDS OLMAYAN MAÇLAR (İlk 20)")
            print("=" * 80)
            
            matches_no_odds = session.query(Match).filter(
                and_(
                    Match.league_id == la_liga.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025,
                    ~Match.id.in_(
                        session.query(MatchOdds.match_id).subquery()
                    )
                )
            ).order_by(Match.match_date.asc()).limit(20).all()
            
            for match in matches_no_odds:
                home_team = TeamRepository.get_by_id(session, match.home_team_id)
                away_team = TeamRepository.get_by_id(session, match.away_team_id)
                
                home_name = home_team.name if home_team else "N/A"
                away_name = away_team.name if away_team else "N/A"
                
                print(f"   {match.match_date.strftime('%Y-%m-%d')} | {home_name:<25} vs {away_name:<25} | {match.home_score}-{match.away_score}")
            
            if matches_without_odds > 20:
                print(f"   ... ve {matches_without_odds - 20} maç daha")
            print()
        
        # Takım bazında durum (en çok eksik olan takımlar)
        print("=" * 80)
        print("📋 TAKIM BAZINDA DURUM (En çok eksik olan 10 takım)")
        print("=" * 80)
        
        all_teams = TeamRepository.get_by_league(session, la_liga.id)
        team_stats = []
        
        for team in all_teams:
            # Bu takımın home maçları
            home_total = session.query(Match).filter(
                and_(
                    Match.league_id == la_liga.id,
                    Match.home_team_id == team.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            home_with_odds = session.query(Match).join(MatchOdds).filter(
                and_(
                    Match.league_id == la_liga.id,
                    Match.home_team_id == team.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            # Bu takımın away maçları
            away_total = session.query(Match).filter(
                and_(
                    Match.league_id == la_liga.id,
                    Match.away_team_id == team.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            away_with_odds = session.query(Match).join(MatchOdds).filter(
                and_(
                    Match.league_id == la_liga.id,
                    Match.away_team_id == team.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            team_total = home_total + away_total
            team_with_odds = home_with_odds + away_with_odds
            team_without_odds = team_total - team_with_odds
            
            if team_total > 0:
                team_pct = (team_with_odds / team_total * 100)
                team_stats.append({
                    'team': team.name,
                    'total': team_total,
                    'with_odds': team_with_odds,
                    'without_odds': team_without_odds,
                    'pct': team_pct
                })
        
        # En çok eksik olan takımları sırala
        team_stats.sort(key=lambda x: x['without_odds'], reverse=True)
        
        for i, stat in enumerate(team_stats[:10], 1):
            status = "✅" if stat['pct'] >= 80 else "⚠️" if stat['pct'] >= 50 else "❌"
            print(f"{status} {i:>2}. {stat['team']:<30} | {stat['with_odds']:>3}/{stat['total']:>3} ({stat['pct']:>5.1f}%) | Eksik: {stat['without_odds']:>3}")
        
        print()
        print("=" * 80)
        print(f"📊 ÖZET: {matches_with_odds:,}/{total_matches:,} maçta odds var ({percentage:.1f}%)")
        print(f"❌ {matches_without_odds:,} maçta odds eksik")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    check_la_liga_odds_status()





