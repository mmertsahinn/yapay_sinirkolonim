"""
Test sonuçlarını Excel'e export et - PRD'ye uygun
Backtest sonuçları, model performans, hata analizleri
"""
import sys
from pathlib import Path
from datetime import date, timedelta

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.reporting.export_excel import ExcelExporter
from football_brain_core.src.reporting.backtest_excel import BacktestExcelExporter
from football_brain_core.src.inference.backtest import Backtester
from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.models.evaluate import ModelEvaluator
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    ModelVersionRepository, MatchRepository, LeagueRepository
)
from football_brain_core.src.features.feature_builder import FeatureBuilder
import torch

def export_test_results(
    date_from: date = None,
    date_to: date = None,
    season: int = None
):
    """
    Test sonuçlarını Excel'e export et.
    
    Args:
        date_from: Başlangıç tarihi
        date_to: Bitiş tarihi
        season: Sezon (date_from/to yerine)
    """
    print("Test sonuclari Excel'e export ediliyor...")
    print("=" * 60)
    
    session = get_session()
    try:
        # Aktif modeli al
        active_model = ModelVersionRepository.get_active(session)
        if not active_model:
            print("[HATA] Aktif model bulunamadi! Once model egitmelisin.")
            return
        
        print(f"Kullanilan model: {active_model.version}")
        
        # Model yükleme kısmı buraya eklenecek
        # Şimdilik placeholder - gerçek implementasyon model dosyasından yüklemeli
        
        market_types = [
            MarketType.MATCH_RESULT,
            MarketType.BTTS,
            MarketType.OVER_UNDER_25,
        ]
        
        # 1. Tahminler ve sonuçlar Excel'i
        print("\n1. Tahminler ve sonuclar export ediliyor...")
        exporter = ExcelExporter()
        
        if season:
            # Sezon bazlı
            leagues = LeagueRepository.get_all(session)
            league_ids = [l.id for l in leagues]
            
            # Sezon tarihleri
            date_from = date(season, 8, 1)
            date_to = date(season + 1, 7, 31)
        else:
            if not date_from:
                date_from = date.today() - timedelta(days=30)
            if not date_to:
                date_to = date.today()
            league_ids = None
        
        predictions_path = exporter.export_predictions(
            date_from=date_from,
            date_to=date_to,
            league_ids=league_ids,
            model_version_id=active_model.id
        )
        print(f"[OK] Tahminler export edildi: {predictions_path}")
        
        # 2. Backtest sonuçları (eğer model yüklüyse)
        print("\n2. Backtest sonuclari hazirlaniyor...")
        print("[NOT] Model yukleme implementasyonu eklendiginde backtest calisacak")
        
        # Placeholder - gerçek implementasyonda:
        # model = load_model(...)
        # backtester = Backtester(model, market_types)
        # backtest_results = backtester.backtest_by_date_range(date_from, date_to, league_ids)
        # backtest_exporter = BacktestExcelExporter()
        # backtest_path = backtest_exporter.export_backtest_results(backtest_results, date_from, date_to)
        
        # 3. Model performans metrikleri
        print("\n3. Model performans metrikleri hazirlaniyor...")
        print("[NOT] Model yukleme implementasyonu eklendiginde metrikler hesaplanacak")
        
        print("\n" + "=" * 60)
        print("EXPORT TAMAMLANDI!")
        print("=" * 60)
        print(f"Tahminler: {predictions_path}")
        print("\nExcel dosyalari 'reports/' klasorunde!")
        
    except Exception as e:
        print(f"[HATA] Export hatasi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test sonuclarini Excel'e export et")
    parser.add_argument("--date-from", type=str, help="Baslangic tarihi (YYYY-MM-DD)")
    parser.add_argument("--date-to", type=str, help="Bitis tarihi (YYYY-MM-DD)")
    parser.add_argument("--season", type=int, help="Sezon (ornek: 2023)")
    
    args = parser.parse_args()
    
    date_from = None
    date_to = None
    
    if args.date_from:
        date_from = date.fromisoformat(args.date_from)
    if args.date_to:
        date_to = date.fromisoformat(args.date_to)
    
    export_test_results(
        date_from=date_from,
        date_to=date_to,
        season=args.season
    )







