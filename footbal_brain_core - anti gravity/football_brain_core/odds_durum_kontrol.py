"""
Odds verilerinin durumunu kontrol eder ve eksikleri gösterir
"""
import sys
from pathlib import Path
from datetime import datetime

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League
from sqlalchemy import and_, extract, func

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_odds_status():
    """Odds verilerinin durumunu kontrol eder"""
    session = get_session()
    
    try:
        # Önce match_odds tablosunun var olup olmadığını kontrol et
        try:
            test_query = session.query(MatchOdds).limit(1).all()
            table_exists = True
        except Exception as e:
            table_exists = False
            print(f"⚠️  match_odds tablosu henüz oluşturulmamış: {type(e).__name__}")
            print("   Önce odds_tablo_olustur.py çalıştırılmalı!")
            return
        
        print("=" * 80)
        print("📊 ODDS VERİ DURUMU KONTROLÜ")
        print("=" * 80)
        print()
        
        # 2020-2025 arası toplam maç sayısı
        total_matches = session.query(Match).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        # Odds'ı olan maç sayısı
        matches_with_odds = session.query(Match).join(MatchOdds).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        print(f"📊 Genel Durum:")
        print(f"   Toplam maç (2020-2025): {total_matches:,}")
        print(f"   Odds'ı olan maç: {matches_with_odds:,}")
        print(f"   Odds'ı olmayan maç: {total_matches - matches_with_odds:,}")
        print(f"   Yüzde: {(matches_with_odds/total_matches*100) if total_matches > 0 else 0:.1f}%")
        print()
        
        # Lig bazında kontrol
        print("=" * 80)
        print("📋 LİG BAZINDA DURUM")
        print("=" * 80)
        
        leagues = session.query(League).all()
        
        league_stats = []
        for league in leagues:
            # Bu ligdeki toplam maç
            league_matches = session.query(Match).filter(
                and_(
                    Match.league_id == league.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            # Bu ligdeki odds'ı olan maç
            league_matches_with_odds = session.query(Match).join(MatchOdds).filter(
                and_(
                    Match.league_id == league.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            if league_matches > 0:
                percentage = (league_matches_with_odds / league_matches) * 100
                league_stats.append({
                    "league": league.name,
                    "total": league_matches,
                    "with_odds": league_matches_with_odds,
                    "without_odds": league_matches - league_matches_with_odds,
                    "percentage": percentage
                })
        
        # Yüzdeye göre sırala (en eksik olanlar üstte)
        league_stats.sort(key=lambda x: x["percentage"])
        
        print(f"{'Lig':<30} {'Toplam':<10} {'Odds Var':<12} {'Odds Yok':<12} {'%':<10}")
        print("-" * 80)
        
        for stat in league_stats:
            status = "✅" if stat["percentage"] >= 80 else "⚠️" if stat["percentage"] >= 50 else "❌"
            print(f"{status} {stat['league']:<28} {stat['total']:<10} {stat['with_odds']:<12} {stat['without_odds']:<12} {stat['percentage']:.1f}%")
        
        print()
        
        # Yıl bazında kontrol
        print("=" * 80)
        print("📅 YIL BAZINDA DURUM")
        print("=" * 80)
        
        for year in range(2020, 2026):
            year_matches = session.query(Match).filter(
                and_(
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) == year
                )
            ).count()
            
            year_matches_with_odds = session.query(Match).join(MatchOdds).filter(
                and_(
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) == year
                )
            ).count()
            
            if year_matches > 0:
                percentage = (year_matches_with_odds / year_matches) * 100
                status = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"
                print(f"{status} {year}: {year_matches_with_odds:,}/{year_matches:,} ({percentage:.1f}%)")
        
        print()
        print("=" * 80)
        print("💡 ÖNERİLER")
        print("=" * 80)
        
        # Eksik olan ligleri bul
        missing_leagues = [s for s in league_stats if s["percentage"] < 80]
        if missing_leagues:
            print(f"⚠️  {len(missing_leagues)} ligde odds verisi eksik:")
            for stat in missing_leagues[:10]:  # İlk 10'unu göster
                print(f"   - {stat['league']}: {stat['without_odds']:,} maç eksik")
            if len(missing_leagues) > 10:
                print(f"   ... ve {len(missing_leagues) - 10} lig daha")
            print()
            print("🔧 Odds yüklemek için: python odds_yukle.py")
        else:
            print("✅ Tüm liglerde odds verisi yeterli seviyede!")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    check_odds_status()
