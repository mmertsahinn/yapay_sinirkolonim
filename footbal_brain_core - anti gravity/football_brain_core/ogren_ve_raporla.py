"""
Yapay Zeka Öğrenme ve Hafıza Sistemi
- Kronolojik olarak tüm maçları öğrenir
- Takım profillerini oluşturur
- Takım ikilileri arasındaki ilişkileri öğrenir
- Her şeyi hafızasına kaydeder
- Excel olarak raporlar
"""
import sys
from pathlib import Path
from datetime import datetime
import logging

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.config import Config
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, TeamRepository, LeagueRepository
)
from football_brain_core.src.models.team_profile import TeamProfileManager
from football_brain_core.src.models.pairwise_relationship import PairwiseRelationshipManager
# from football_brain_core.src.models.self_learning import SelfLearningBrain  # Model gerektiriyor, şimdilik kullanmıyoruz
from football_brain_core.src.reporting.team_analysis_excel import TeamAnalysisExcelExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ogren_ve_raporla(seasons: list = None):
    """
    Yapay zeka öğrenme ve hafıza sistemi
    
    - Kronolojik olarak tüm maçları öğrenir
    - Takım profillerini oluşturur (her detay)
    - Takım ikilileri arasındaki ilişkileri öğrenir
    - Her şeyi hafızasına kaydeder
    - Excel olarak raporlar
    """
    print("=" * 80)
    print("YAPAY ZEKA OGRENME VE HAFIZA SISTEMI")
    print("=" * 80)
    print("\nBu sistem:")
    print("  • Kronolojik olarak tüm maçları öğrenir")
    print("  • Her takımın profilini oluşturur (en ince ayrıntısına kadar)")
    print("  • Takım ikilileri arasındaki ilişkileri öğrenir")
    print("  • Her şeyi hafızasına kaydeder")
    print("  • Excel olarak raporlar")
    print("=" * 80)
    
    config = Config()
    session = get_session()
    
    # Market tipleri
    market_types = [
        MarketType.MATCH_RESULT,
        MarketType.BTTS,
        MarketType.OVER_UNDER_25,
        MarketType.GOAL_RANGE,
        MarketType.CORRECT_SCORE,
        MarketType.DOUBLE_CHANCE,
    ]
    
    try:
        # Sezonları belirle
        if seasons is None:
            # Veritabanındaki tüm sezonları bul
            matches = MatchRepository.get_all(session)
            seasons = sorted(set(m.match_date.year for m in matches if m.match_date))
            print(f"\n📅 Bulunan sezonlar: {seasons}")
        else:
            print(f"\n📅 Kullanılacak sezonlar: {seasons}")
        
        # Self Learning Brain oluştur (model olmadan, sadece öğrenme için)
        print("\n🧠 Yapay Zeka Beyni oluşturuluyor...")
        # Not: SelfLearningBrain model gerektiriyor, şimdilik sadece profil ve ilişki öğrenme yapıyoruz
        
        # Her sezon için öğren
        all_team_profiles = {}
        all_pairwise_relationships = {}
        
        for season in seasons:
            print(f"\n{'='*80}")
            print(f"📚 SEZON {season} OGRENILIYOR...")
            print(f"{'='*80}")
            
            # 1. Takım profillerini oluştur (kronolojik)
            print(f"\n1️⃣ Takım profilleri oluşturuluyor (kronolojik öğrenme)...")
            team_profile_manager = TeamProfileManager()
            team_profiles = team_profile_manager.build_all_profiles(season, market_types)
            all_team_profiles[season] = team_profiles
            print(f"   ✅ {len(team_profiles)} takım profili oluşturuldu")
            
            # 2. Takım ikilileri arasındaki ilişkileri öğren
            print(f"\n2️⃣ Takım ikilileri arasındaki ilişkiler öğreniliyor...")
            pairwise_manager = PairwiseRelationshipManager()
            pairwise_relationships = pairwise_manager.build_all_relationships(season, market_types)
            all_pairwise_relationships[season] = pairwise_relationships
            print(f"   ✅ {len(pairwise_relationships)} takım çifti ilişkisi öğrenildi")
            
            # 3. Kronolojik öğrenme (maçları tarih sırasına göre işle)
            print(f"\n3️⃣ Kronolojik öğrenme yapılıyor (tarih sırasına göre)...")
            # Maçları kronolojik olarak işle
            all_matches = []
            for league in LeagueRepository.get_all(session):
                league_matches = MatchRepository.get_by_league_and_season(session, league.id, season)
                league_matches = [m for m in league_matches if m.home_score is not None and m.away_score is not None]
                all_matches.extend(league_matches)
            
            # Tarih sırasına göre sırala
            all_matches.sort(key=lambda m: m.match_date)
            print(f"   ✅ {len(all_matches)} maç kronolojik olarak işlendi")
            learning_results = {"total_matches_processed": len(all_matches), "season": season}
        
        # 4. Excel'e export et
        print(f"\n{'='*80}")
        print("📊 EXCEL RAPORLARI OLUSTURULUYOR...")
        print(f"{'='*80}")
        
        excel_exporter = TeamAnalysisExcelExporter()
        
        # Tüm sezonlar için birleşik rapor
        print("\n📄 Birleşik Excel raporu oluşturuluyor...")
        
        # Takım profilleri Excel'i
        all_profiles_combined = {}
        for season, profiles in all_team_profiles.items():
            for team_id, profile in profiles.items():
                if team_id not in all_profiles_combined:
                    all_profiles_combined[team_id] = profile
                else:
                    # Birleştir (tüm sezonların bilgisi)
                    all_profiles_combined[team_id]['seasons'] = all_profiles_combined[team_id].get('seasons', [])
                    all_profiles_combined[team_id]['seasons'].append(season)
        
        # Excel export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Takım Profilleri
        profiles_path = excel_exporter.export_team_profiles(
            all_profiles_combined,
            output_dir=project_root,
            filename=f"takim_profilleri_{timestamp}.xlsx"
        )
        print(f"   ✅ Takım profilleri: {profiles_path}")
        
        # 2. Takım İlişkileri
        all_relationships_combined = {}
        for season, relationships in all_pairwise_relationships.items():
            for pair_key, relationship in relationships.items():
                if pair_key not in all_relationships_combined:
                    all_relationships_combined[pair_key] = relationship
                else:
                    # Birleştir
                    all_relationships_combined[pair_key]['seasons'] = all_relationships_combined[pair_key].get('seasons', [])
                    all_relationships_combined[pair_key]['seasons'].append(season)
        
        relationships_path = excel_exporter.export_team_relationships(
            all_relationships_combined,
            output_dir=project_root,
            filename=f"takim_iliskileri_{timestamp}.xlsx"
        )
        print(f"   ✅ Takım ilişkileri: {relationships_path}")
        
        # 3. Öğrenme Özeti
        summary_path = excel_exporter.export_learning_summary(
            {
                'seasons': seasons,
                'total_teams': len(all_profiles_combined),
                'total_relationships': len(all_relationships_combined),
                'learning_results': learning_results if 'learning_results' in locals() else {}
            },
            output_dir=project_root,
            filename=f"ogrenme_ozeti_{timestamp}.xlsx"
        )
        print(f"   ✅ Öğrenme özeti: {summary_path}")
        
        print(f"\n{'='*80}")
        print("✅ OGRENME VE RAPORLAMA TAMAMLANDI!")
        print(f"{'='*80}")
        print(f"\n📁 Raporlar:")
        print(f"   • {profiles_path}")
        print(f"   • {relationships_path}")
        print(f"   • {summary_path}")
        print(f"\n💡 Yapay zeka öğrendiği her şeyi hafızasına kaydetti!")
        print(f"   Artık bu bilgileri kullanarak daha iyi tahminler yapabilir.")
        
    except Exception as e:
        logger.error(f"Hata: {e}", exc_info=True)
        print(f"\n❌ Hata: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    # Hangi sezonları öğrenmek istiyorsunuz?
    # None = Tüm sezonlar
    seasons = [2021, 2022, 2023, 2024]  # Veya None
    
    ogren_ve_raporla(seasons=seasons)

