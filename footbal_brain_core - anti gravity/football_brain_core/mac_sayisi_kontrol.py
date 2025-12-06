"""
Veritabanındaki maç sayısını kontrol eder
"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import extract, func
from src.db.connection import get_session
from src.db.schema import Match, League

session = get_session()

try:
    print("=" * 80)
    print("📊 VERİTABANI MAÇ SAYISI ANALİZİ")
    print("=" * 80)
    
    # Toplam maç sayısı
    total_matches = session.query(Match).count()
    print(f"\n✅ Toplam maç sayısı: {total_matches:,}")
    
    # Sonuç bilgisi olan maçlar
    matches_with_result = session.query(Match).filter(
        Match.home_score.isnot(None),
        Match.away_score.isnot(None)
    ).count()
    print(f"🎯 Sonuç bilgisi olan maç: {matches_with_result:,}")
    
    # Yıl bazında dağılım
    print("\n" + "=" * 80)
    print("📅 YIL BAZINDA MAÇ SAYISI")
    print("=" * 80)
    
    year_stats = session.query(
        extract('year', Match.match_date).label('year'),
        func.count(Match.id).label('count')
    ).group_by(extract('year', Match.match_date)).order_by('year').all()
    
    print(f"\n{'Yıl':<10} {'Toplam Maç':<15} {'Sonuç Var':<15} {'Sonuç Yok':<15}")
    print("-" * 60)
    
    for year, count in year_stats:
        if year:
            with_result = session.query(Match).filter(
                extract('year', Match.match_date) == year,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None)
            ).count()
            without_result = count - with_result
            print(f"{int(year):<10} {count:<15,} {with_result:<15,} {without_result:<15,}")
    
    # 2020-2022 arası detay
    print("\n" + "=" * 80)
    print("📊 2020-2022 ARASI DETAY")
    print("=" * 80)
    
    matches_2020_2022 = session.query(Match).filter(
        extract('year', Match.match_date) >= 2020,
        extract('year', Match.match_date) <= 2022
    ).count()
    
    matches_2020_2022_with_result = session.query(Match).filter(
        extract('year', Match.match_date) >= 2020,
        extract('year', Match.match_date) <= 2022,
        Match.home_score.isnot(None),
        Match.away_score.isnot(None)
    ).count()
    
    print(f"\nToplam maç (2020-2022): {matches_2020_2022:,}")
    print(f"Sonuç bilgisi olan (2020-2022): {matches_2020_2022_with_result:,}")
    
    # Lig bazında 2020-2022
    print("\n" + "=" * 80)
    print("📋 LİG BAZINDA 2020-2022 MAÇ SAYISI")
    print("=" * 80)
    
    leagues = session.query(League).all()
    league_counts = []
    
    for league in leagues:
        count = session.query(Match).filter(
            Match.league_id == league.id,
            extract('year', Match.match_date) >= 2020,
            extract('year', Match.match_date) <= 2022,
            Match.home_score.isnot(None),
            Match.away_score.isnot(None)
        ).count()
        
        if count > 0:
            league_counts.append((league.name, count))
    
    league_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Lig':<30} {'Maç Sayısı':<15}")
    print("-" * 50)
    for name, count in league_counts:
        print(f"{name:<30} {count:<15,}")
    
    # Toplam lig sayısı
    total_leagues = session.query(League).count()
    print(f"\n📌 Toplam lig sayısı: {total_leagues}")
    
    # Sezon bazında kontrol (eğer sezon bilgisi varsa)
    print("\n" + "=" * 80)
    print("💡 NOTLAR")
    print("=" * 80)
    print("""
- Bir sezonda (yılda) her lig için yaklaşık 300-400 maç olur
- 20+ lig varsa: 20 lig × 350 maç = 7,000 maç/yıl
- 3 yılda (2020-2022): 7,000 × 3 = 21,000 maç normal
- Eğer daha fazla lig veya alt ligler varsa sayı artabilir
    """)
    
finally:
    session.close()
