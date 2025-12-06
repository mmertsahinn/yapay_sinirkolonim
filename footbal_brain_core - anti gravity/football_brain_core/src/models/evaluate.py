from typing import List, Dict, Tuple
import torch
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import logging

from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.features.market_targets import MarketType, MARKET_OUTCOMES
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, ResultRepository, MarketRepository
from football_brain_core.src.features.feature_builder import FeatureBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(y_proba.shape) == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])
    
    y_one_hot = np.zeros_like(y_proba)
    y_one_hot[np.arange(len(y_true)), y_true] = 1
    
    return np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1))


class ModelEvaluator:
    def __init__(self, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_model(
        self,
        model: MultiTaskModel,
        matches: List,
        market_types: List[MarketType]
    ) -> Dict[str, Dict[str, float]]:
        session = get_session()
        try:
            model.eval()
            results = {}
            
            for market_type in market_types:
                predictions = []
                actuals = []
                probabilities = []
                
                market = MarketRepository.get_or_create(
                    session, name=market_type.value
                )
                
                for match in matches:
                    if match.home_score is None or match.away_score is None:
                        continue
                    
                    features = self.feature_builder.build_match_features(
                        match.home_team_id,
                        match.away_team_id,
                        match.match_date,
                        match.league_id,
                        session
                    )
                    
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(features_tensor)
                        probas = torch.softmax(outputs[market_type.value], dim=-1)
                        pred = torch.argmax(probas, dim=-1).cpu().numpy()[0]
                    
                    result = ResultRepository.get_by_match(session, match.id)
                    market_result = next(
                        (r for r in result if r.market_id == market.id), None
                    )
                    
                    if market_result:
                        actual_outcome = market_result.actual_outcome
                        if actual_outcome in MARKET_OUTCOMES[market_type]:
                            actual_idx = MARKET_OUTCOMES[market_type].index(actual_outcome)
                            predictions.append(pred)
                            actuals.append(actual_idx)
                            probabilities.append(probas.cpu().numpy()[0])
                
                if len(actuals) > 0:
                    accuracy = accuracy_score(actuals, predictions)
                    
                    probas_array = np.array(probabilities)
                    actuals_array = np.array(actuals)
                    
                    try:
                        brier = brier_score(actuals_array, probas_array)
                    except:
                        brier = 0.0
                    
                    try:
                        log_loss_val = log_loss(actuals_array, probas_array)
                    except:
                        log_loss_val = 0.0
                    
                    results[market_type.value] = {
                        "accuracy": accuracy,
                        "brier_score": brier,
                        "log_loss": log_loss_val,
                        "num_samples": len(actuals)
                    }
                else:
                    results[market_type.value] = {
                        "accuracy": 0.0,
                        "brier_score": 0.0,
                        "log_loss": 0.0,
                        "num_samples": 0
                    }
            
            return results
        finally:
            session.close()
    
    def evaluate_by_league(
        self,
        model: MultiTaskModel,
        league_id: int,
        season: int,
        market_types: List[MarketType]
    ) -> Dict[str, Dict[str, float]]:
        session = get_session()
        try:
            matches = MatchRepository.get_by_league_and_season(
                session, league_id, season
            )
            return self.evaluate_model(model, matches, market_types)
        finally:
            session.close()







