from datetime import date, timedelta
from typing import Optional, List
import logging

from football_brain_core.src.reporting.export_excel import ExcelExporter
from football_brain_core.src.inference.predict_markets import MarketPredictor
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import ModelVersionRepository
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyReporter:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.exporter = ExcelExporter(config)
    
    def generate_daily_report(
        self,
        predictor: Optional[MarketPredictor] = None,
        model_version_id: Optional[int] = None
    ) -> str:
        logger.info("Generating daily report")
        
        today = date.today()
        date_from = today
        date_to = today + timedelta(days=7)
        
        if predictor and not model_version_id:
            session = get_session()
            try:
                active_model = ModelVersionRepository.get_active(session)
                if active_model:
                    model_version_id = active_model.id
            finally:
                session.close()
        
        output_path = self.exporter.export_predictions(
            date_from=date_from,
            date_to=date_to,
            model_version_id=model_version_id
        )
        
        logger.info(f"Daily report generated: {output_path}")
        return output_path







