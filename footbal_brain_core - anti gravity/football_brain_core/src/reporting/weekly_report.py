from datetime import date, timedelta
from typing import Optional, Dict, List
import logging

from football_brain_core.src.reporting.export_excel import ExcelExporter
from football_brain_core.src.inference.backtest import Backtester
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import ModelVersionRepository, LeagueRepository
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeeklyReporter:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.exporter = ExcelExporter(config)
    
    def generate_weekly_report(
        self,
        backtester: Optional[Backtester] = None,
        model_version_id: Optional[int] = None
    ) -> Dict[str, str]:
        logger.info("Generating weekly report")
        
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        
        results = {}
        
        if backtester:
            session = get_session()
            try:
                leagues = LeagueRepository.get_all(session)
                league_ids = [l.id for l in leagues]
                
                backtest_results = backtester.backtest_by_date_range(
                    week_start, week_end, league_ids
                )
                
                results["backtest"] = backtest_results
                logger.info(f"Backtest accuracy: {backtest_results.get('accuracy_by_market', {})}")
            finally:
                session.close()
        
        output_path = self.exporter.export_predictions(
            date_from=week_start,
            date_to=week_end,
            model_version_id=model_version_id
        )
        
        results["excel_path"] = output_path
        logger.info(f"Weekly report generated: {output_path}")
        
        return results







