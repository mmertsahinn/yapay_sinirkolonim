"""Maç verilerini analiz et - duplicate ve lig bazında"""
import sys
from pathlib import Path
from collections import Counter

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.schema import Match, League
from sqlalchemy import func, extract

session = get_session()
try:
    print("=" * 80)
    print("MAC VERILERI ANALIZI")
    print("=" * 80)
    
    # 1. Toplam maç sayısı
    total_matches = session.query(Match).count()
    print(f"\n1. TOPLAM MAC SAYISI: {total_matches:,}")
    
    # 2. Benzersiz match_id kontrolü
    unique_match_ids = session.query(func.count(func.distinct(Match.match_id))).scalar()
    print(f"2. BENZERSIZ match_id SAYISI: {unique_match_ids:,}")
    
    if total_matches != unique_match_ids:
        duplicates = total_matches - unique_match_ids
        print(f"   [UYARI] {duplicates:,} duplicate mac kaydi var!")
        
        # Duplicate match_id'leri bul
        duplicate_ids = session.query(
            Match.match_id,
            func.count(Match.id).label('count')
        ).group_by(Match.match_id).having(func.count(Match.id) > 1).limit(10).all()
        
        if duplicate_ids:
            print(f"\n   Ilk 10 duplicate match_id:")
            for match_id, count in duplicate_ids:
                print(f"     {match_id}: {count} kez")
    else:
        print("   [OK] Duplicate mac yok")
    
    # 3. Lig bazında dağılım
    print(f"\n3. LIG BAZINDA MAC SAYILARI:")
    print("-" * 80)
    league_counts = session.query(
        League.name,
        func.count(Match.id).label('count')
    ).join(Match, League.id == Match.league_id).group_by(League.name).order_by(func.count(Match.id).desc()).all()
    
    for league_name, count in league_counts:
        print(f"   {league_name:30s}: {count:6,} mac")
    
    # 4. Yıl bazında dağılım
    print(f"\n4. YIL BAZINDA MAC SAYILARI:")
    print("-" * 80)
    year_counts = session.query(
        extract('year', Match.match_date).label('year'),
        func.count(Match.id).label('count')
    ).group_by('year').order_by('year').all()
    
    for year, count in year_counts:
        if year:
            print(f"   {int(year)}: {count:6,} mac")
    
    # 5. Aynı takımlar arası maç sayısı (duplicate kontrolü)
    print(f"\n5. AYNI TAKIMLAR ARASI MAC KONTROLU:")
    print("-" * 80)
    same_teams = session.query(
        Match.home_team_id,
        Match.away_team_id,
        func.count(Match.id).label('count')
    ).group_by(Match.home_team_id, Match.away_team_id).having(func.count(Match.id) > 5).order_by(func.count(Match.id).desc()).limit(10).all()
    
    if same_teams:
        print(f"   En cok tekrar eden takim eslesmeleri (5'ten fazla):")
        for home_id, away_id, count in same_teams:
            home_team = session.query(Match).filter(Match.home_team_id == home_id).first()
            away_team = session.query(Match).filter(Match.away_team_id == away_id).first()
            if home_team and away_team:
                home_name = home_team.home_team.name if home_team.home_team else "N/A"
                away_name = away_team.away_team.name if away_team.away_team else "N/A"
                print(f"     {home_name} vs {away_name}: {count} mac")
    
    # 6. match_id NULL kontrolü
    null_match_ids = session.query(Match).filter(Match.match_id == None).count()
    if null_match_ids > 0:
        print(f"\n6. [UYARI] match_id NULL olan mac sayisi: {null_match_ids:,}")
    
    print("\n" + "=" * 80)
    
finally:
    session.close()






