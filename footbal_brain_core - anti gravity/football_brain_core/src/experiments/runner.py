from typing import Dict, Any, List
import json
import uuid
from datetime import datetime
import logging

from football_brain_core.src.models.train_offline import OfflineTrainer
from football_brain_core.src.models.evaluate import ModelEvaluator
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.features.feature_builder import FeatureBuilder
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import ExperimentRepository, LeagueRepository
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.feature_builder = FeatureBuilder(
            sequence_length=self.config.SEQUENCE_LENGTH
        )
    
    def run_experiment(
        self,
        experiment_config: Dict[str, Any],
        train_seasons: List[int],
        val_seasons: List[int],
        market_types: List[MarketType]
    ) -> Dict[str, Any]:
        experiment_id = str(uuid.uuid4())
        logger.info(f"Starting experiment {experiment_id}")
        
        session = get_session()
        try:
            league_ids = [
                LeagueRepository.get_or_create(session, league.name).id
                for league in self.config.TARGET_LEAGUES
            ]
            
            model_config = experiment_config.get("model_config", {})
            trainer = OfflineTrainer(market_types, self.config, model_config)
            
            model = trainer.train(
                train_seasons,
                val_seasons,
                league_ids,
                epochs=experiment_config.get("epochs", self.config.MODEL_CONFIG.epochs)
            )
            
            evaluator = ModelEvaluator(self.feature_builder)
            metrics = {}
            
            for league_id in league_ids:
                league_metrics = evaluator.evaluate_by_league(
                    model, league_id, val_seasons[-1], market_types
                )
                metrics[f"league_{league_id}"] = league_metrics
            
            overall_metrics = {}
            for market_type in market_types:
                market_metrics = []
                for league_metrics in metrics.values():
                    if market_type.value in league_metrics:
                        market_metrics.append(league_metrics[market_type.value])
                
                if market_metrics:
                    overall_metrics[market_type.value] = {
                        "avg_accuracy": sum(m["accuracy"] for m in market_metrics) / len(market_metrics),
                        "avg_brier_score": sum(m["brier_score"] for m in market_metrics) / len(market_metrics),
                        "avg_log_loss": sum(m["log_loss"] for m in market_metrics) / len(market_metrics),
                    }
            
            period_start = datetime(train_seasons[0], 8, 1)
            period_end = datetime(val_seasons[-1] + 1, 7, 31)
            
            experiment = ExperimentRepository.create(
                session,
                experiment_id=experiment_id,
                config=experiment_config,
                period_start=period_start,
                period_end=period_end,
                metrics=overall_metrics
            )
            
            session.commit()
            
            logger.info(f"Experiment {experiment_id} completed")
            
            return {
                "experiment_id": experiment_id,
                "metrics": overall_metrics,
                "config": experiment_config
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Experiment {experiment_id} failed: {e}")
            raise
        finally:
            session.close()







