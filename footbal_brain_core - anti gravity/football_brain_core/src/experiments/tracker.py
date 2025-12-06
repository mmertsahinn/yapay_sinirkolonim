from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import ExperimentRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    @staticmethod
    def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
        session = get_session()
        try:
            experiment = ExperimentRepository.get_by_id(session, experiment_id)
            if experiment:
                return {
                    "experiment_id": experiment.experiment_id,
                    "config": experiment.config,
                    "period_start": experiment.period_start.isoformat() if experiment.period_start else None,
                    "period_end": experiment.period_end.isoformat() if experiment.period_end else None,
                    "metrics": experiment.metrics,
                    "created_at": experiment.created_at.isoformat()
                }
            return None
        finally:
            session.close()
    
    @staticmethod
    def list_experiments(limit: int = 50) -> List[Dict[str, Any]]:
        session = get_session()
        try:
            experiments = ExperimentRepository.get_all(session)
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "config": exp.config,
                    "metrics": exp.metrics,
                    "created_at": exp.created_at.isoformat()
                }
                for exp in experiments[:limit]
            ]
        finally:
            session.close()
    
    @staticmethod
    def compare_experiments(experiment_ids: List[str]) -> Dict[str, Any]:
        session = get_session()
        try:
            experiments = [
                ExperimentRepository.get_by_id(session, exp_id)
                for exp_id in experiment_ids
            ]
            experiments = [e for e in experiments if e is not None]
            
            comparison = {}
            for exp in experiments:
                comparison[exp.experiment_id] = {
                    "config": exp.config,
                    "metrics": exp.metrics,
                    "created_at": exp.created_at.isoformat()
                }
            
            return comparison
        finally:
            session.close()
    
    @staticmethod
    def get_best_experiment(metric: str = "avg_accuracy") -> Optional[Dict[str, Any]]:
        session = get_session()
        try:
            experiments = ExperimentRepository.get_all(session)
            
            best_exp = None
            best_value = -1
            
            for exp in experiments:
                if exp.metrics:
                    for market_metrics in exp.metrics.values():
                        if isinstance(market_metrics, dict) and metric in market_metrics:
                            value = market_metrics[metric]
                            if value > best_value:
                                best_value = value
                                best_exp = exp
            
            if best_exp:
                return {
                    "experiment_id": best_exp.experiment_id,
                    "config": best_exp.config,
                    "metrics": best_exp.metrics,
                    "created_at": best_exp.created_at.isoformat()
                }
            return None
        finally:
            session.close()







