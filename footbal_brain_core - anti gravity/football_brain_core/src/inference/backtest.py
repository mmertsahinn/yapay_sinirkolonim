from datetime import datetime, date
from typing import List, Dict, Optional, Any
import logging

from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.inference.predict_markets import MarketPredictor
from football_brain_core.src.models.evaluate import ModelEvaluator
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, ResultRepository, MarketRepository
from football_brain_core.src.features.feature_builder import FeatureBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    def __init__(
        self,
        model: MultiTaskModel,
        market_types: List[MarketType],
        feature_builder: Optional[FeatureBuilder] = None
    ):
        self.predictor = MarketPredictor(model, market_types, feature_builder)
        self.evaluator = ModelEvaluator(feature_builder or FeatureBuilder())
        self.market_types = market_types
    
    def backtest_matches(
        self,
        matches: List,
        cutoff_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        session = get_session()
        try:
            results = {
                "total_matches": 0,
                "correct_predictions": {},
                "incorrect_predictions": {},
                "accuracy_by_market": {},
                "match_results": []
            }
            
            for market_type in self.market_types:
                results["correct_predictions"][market_type.value] = 0
                results["incorrect_predictions"][market_type.value] = 0
            
            for match in matches:
                if cutoff_date and match.match_date >= cutoff_date:
                    continue
                
                if match.home_score is None or match.away_score is None:
                    continue
                
                try:
                    predictions = self.predictor.predict_match(match.id, session)
                    
                    match_result = {
                        "match_id": match.id,
                        "match_date": match.match_date.isoformat(),
                        "predictions": {},
                        "actuals": {},
                        "correct": {}
                    }
                    
                    for market_type in self.market_types:
                        market = MarketRepository.get_or_create(
                            session, name=market_type.value
                        )
                        result = ResultRepository.get_by_match(session, match.id)
                        market_result = next(
                            (r for r in result if r.market_id == market.id), None
                        )
                        
                        if market_result:
                            actual_outcome = market_result.actual_outcome
                            pred_outcome = predictions[market_type]["outcome"]
                            
                            match_result["predictions"][market_type.value] = pred_outcome
                            match_result["actuals"][market_type.value] = actual_outcome
                            
                            is_correct = pred_outcome == actual_outcome
                            match_result["correct"][market_type.value] = is_correct
                            
                            if is_correct:
                                results["correct_predictions"][market_type.value] += 1
                            else:
                                results["incorrect_predictions"][market_type.value] += 1
                    
                    results["match_results"].append(match_result)
                    results["total_matches"] += 1
                
                except Exception as e:
                    logger.error(f"Error backtesting match {match.id}: {e}")
            
            for market_type in self.market_types:
                correct = results["correct_predictions"][market_type.value]
                incorrect = results["incorrect_predictions"][market_type.value]
                total = correct + incorrect
                
                if total > 0:
                    results["accuracy_by_market"][market_type.value] = correct / total
                else:
                    results["accuracy_by_market"][market_type.value] = 0.0
            
            return results
        finally:
            session.close()
    
    def backtest_by_date_range(
        self,
        date_from: date,
        date_to: date,
        league_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        session = get_session()
        try:
            matches = MatchRepository.get_by_date_range(session, date_from, date_to)
            
            if league_ids:
                matches = [m for m in matches if m.league_id in league_ids]
            
            return self.backtest_matches(matches)
        finally:
            session.close()
    
    def backtest_by_season(
        self,
        league_id: int,
        season: int
    ) -> Dict[str, any]:
        session = get_session()
        try:
            matches = MatchRepository.get_by_league_and_season(
                session, league_id, season
            )
            
            cutoff_date = datetime(season, 8, 1)
            
            return self.backtest_matches(matches, cutoff_date=cutoff_date)
        finally:
            session.close()

