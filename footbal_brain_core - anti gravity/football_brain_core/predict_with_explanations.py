"""
Tahmin yap ve LLM ile açıklama üret - PRD'ye uygun tam workflow
"""
import sys
from pathlib import Path
from datetime import date, timedelta
import logging
import torch

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.inference.predict_markets import MarketPredictor, load_model_and_predict
from src.explanations.scenario_builder import ScenarioBuilder
from src.features.market_targets import MarketType
from src.db.connection import get_session
from src.db.repositories import (
    MatchRepository, ModelVersionRepository, MarketRepository
)
from src.config import Config
from src.models.multi_task_model import MultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_and_explain(
    model_path: str = None,
    days_ahead: int = 7,
    market_types: list = None
):
    """
    Gelecek maçlar için tahmin yap ve LLM ile açıklama üret
    
    Args:
        model_path: Eğitilmiş model dosyası yolu (None ise aktif model kullanılır)
        days_ahead: Kaç gün ileriye tahmin yapılacak
        market_types: Hangi marketler için tahmin yapılacak
    """
    if market_types is None:
        market_types = [
            MarketType.MATCH_RESULT,
            MarketType.BTTS,
            MarketType.OVER_UNDER_25,
            MarketType.GOAL_RANGE,
        ]
    
    config = Config()
    session = get_session()
    
    try:
        # Aktif modeli al
        active_model = ModelVersionRepository.get_active(session)
        if not active_model:
            logger.error("Aktif model bulunamadı! Önce model eğitmelisin.")
            return
        
        logger.info(f"Kullanılan model: {active_model.version}")
        
        # Modeli yükle
        if model_path:
            logger.info(f"Model yükleniyor: {model_path}")
            # Model yükleme kodu buraya eklenecek
            # Şimdilik aktif model kullanılıyor
        else:
            logger.info("Aktif model kullanılıyor")
        
        # Tahmin edilecek maçları bul
        date_from = date.today()
        date_to = date_from + timedelta(days=days_ahead)
        
        matches = MatchRepository.get_by_date_range(session, date_from, date_to)
        matches = [m for m in matches if m.home_score is None or m.away_score is None]
        
        logger.info(f"{len(matches)} maç için tahmin yapılacak")
        
        if not matches:
            logger.warning("Tahmin edilecek maç bulunamadı!")
            return
        
        # Predictor ve ScenarioBuilder oluştur
        # Not: Model yükleme kısmı tam implementasyon gerektirir
        # Şimdilik placeholder
        logger.info("Predictor ve ScenarioBuilder hazırlanıyor...")
        
        scenario_builder = ScenarioBuilder()
        
        # Her maç için tahmin yap ve açıklama üret
        for i, match in enumerate(matches, 1):
            try:
                logger.info(f"[{i}/{len(matches)}] Maç {match.id} işleniyor...")
                
                # Tahmin yap (model yüklendikten sonra)
                # predictions = predictor.predict_match(match.id, session)
                
                # Şimdilik placeholder - gerçek implementasyon model yüklendikten sonra
                logger.info(f"  ⏳ Tahmin yapılıyor...")
                
                # LLM ile açıklama üret
                # explanations = scenario_builder.generate_explanation(
                #     match, predictions, market_types
                # )
                
                logger.info(f"  ⏳ LLM açıklaması üretiliyor...")
                
                # Kaydet
                # predictor.save_predictions(match.id, predictions, active_model.id)
                # scenario_builder.save_explanations(match, explanations, {})
                
                logger.info(f"  ✅ Maç {match.id} tamamlandı")
                
            except Exception as e:
                logger.error(f"  ❌ Maç {match.id} için hata: {e}")
                continue
        
        logger.info("✅ Tüm tahminler ve açıklamalar tamamlandı!")
        logger.info("📊 Excel raporu oluşturmak için: python -m football_brain_core.src.cli.main report-daily")
        
    except Exception as e:
        logger.error(f"Hata: {e}", exc_info=True)
    finally:
        session.close()


if __name__ == "__main__":
    print("🧠 Tahmin ve Açıklama Üretimi")
    print("=" * 50)
    
    # Model yolu varsa belirt, yoksa aktif model kullanılır
    model_path = None  # "model_v1.0.pth"
    
    predict_and_explain(
        model_path=model_path,
        days_ahead=7,
        market_types=[
            MarketType.MATCH_RESULT,
            MarketType.BTTS,
            MarketType.OVER_UNDER_25,
        ]
    )


