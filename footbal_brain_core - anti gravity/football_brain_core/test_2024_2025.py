"""
2024-2025 verileriyle test yap (denemeleri gerisinden)
Eğitilmiş model ile 2024-2025 maçlarını tahmin et ve sonuçlarla karşılaştır
"""
import sys
from pathlib import Path
from datetime import date
import torch

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.config import Config
from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.features.feature_builder import FeatureBuilder
from football_brain_core.src.features.market_targets import MarketType, MARKET_OUTCOMES
from football_brain_core.src.inference.backtest import Backtester
from football_brain_core.src.reporting.backtest_excel import BacktestExcelExporter
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, LeagueRepository

def test_2024_2025():
    """2024-2025 verileriyle test yap"""
    print("=" * 80)
    print("2024-2025 TEST (DENEMELERI GERISINDEN)")
    print("=" * 80)
    
    # Model yükle
    model_path = "model_prd_v1.0.pth"
    print(f"\n📦 Model yükleniyor: {model_path}")
    
    market_types = [
        MarketType.MATCH_RESULT,
        MarketType.BTTS,
        MarketType.OVER_UNDER_25,
        MarketType.GOAL_RANGE,
        MarketType.CORRECT_SCORE,
        MarketType.DOUBLE_CHANCE,
    ]
    
    config = Config()
    feature_builder = FeatureBuilder()
    session = get_session()
    
    try:
        # Model yükle
        sample_match = MatchRepository.get_by_id(session, 1)
        if sample_match:
            sample_features = feature_builder.build_match_features(
                sample_match.home_team_id,
                sample_match.away_team_id,
                sample_match.match_date,
                sample_match.league_id,
                session
            )
            input_size = len(sample_features)
        else:
            input_size = 50
        
        model = MultiTaskModel(
            input_size=input_size,
            market_types=market_types,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("✅ Model yüklendi!")
        
        # Backtester oluştur
        backtester = Backtester(model, market_types, feature_builder)
        
        # 2024-2025 maçlarını bul
        print("\n📅 2024-2025 maçları bulunuyor...")
        leagues = LeagueRepository.get_all(session)
        league_ids = [l.id for l in leagues]
        
        all_test_matches = []
        for league_id in league_ids:
            # 2024 sezonu
            matches_2024 = MatchRepository.get_by_league_and_season(session, league_id, 2024)
            all_test_matches.extend(matches_2024)
            
            # 2025 sezonu
            try:
                matches_2025 = MatchRepository.get_by_league_and_season(session, league_id, 2025)
                all_test_matches.extend(matches_2025)
            except:
                pass
        
        # Sadece skoru olan maçlar
        test_matches = [m for m in all_test_matches if m.home_score is not None and m.away_score is not None]
        print(f"   ✅ {len(test_matches)} test maçı bulundu")
        
        if not test_matches:
            print("⚠️  Test için maç bulunamadı!")
            return
        
        # Backtest yap
        print("\n🧪 Backtest yapılıyor (tahminler vs gerçek sonuçlar)...")
        backtest_results = backtester.backtest_matches(test_matches)
        
        print(f"\n📊 Sonuçlar:")
        print(f"   Toplam maç: {backtest_results['total_matches']}")
        
        for market_type in market_types:
            correct = backtest_results['correct_predictions'].get(market_type.value, 0)
            incorrect = backtest_results['incorrect_predictions'].get(market_type.value, 0)
            total = correct + incorrect
            accuracy = backtest_results['accuracy_by_market'].get(market_type.value, 0)
            
            print(f"   {market_type.value}: {accuracy:.1%} ({correct}/{total})")
        
        # Excel'e export et
        print("\n📊 Excel raporu oluşturuluyor...")
        exporter = BacktestExcelExporter(config)
        
        excel_path = exporter.export_backtest_results(
            backtest_results,
            date_from=date(2024, 1, 1),
            date_to=date(2025, 12, 31)
        )
        
        print(f"\n✅ Excel raporu oluşturuldu: {excel_path}")
        print(f"\n📋 Rapor içeriği:")
        print(f"   • Test Results: Her maç için tahminler ve gerçek sonuçlar")
        print(f"   • Summary: Doğruluk metrikleri")
        print(f"\n💡 Excel'de:")
        print(f"   • Önce tahminler görünür")
        print(f"   • Sonra gerçek sonuçlar")
        print(f"   • Correct kolonu: Yes/No (doğru/yanlış)")
        print(f"   • En az hatası olan maçlar (Correct=Yes) öğrenme için kullanılabilir")
        
    except FileNotFoundError:
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print("💡 Önce model eğitmelisiniz: python quick_test.py")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    test_2024_2025()






