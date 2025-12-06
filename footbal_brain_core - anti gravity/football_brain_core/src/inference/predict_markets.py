from datetime import datetime, date
from typing import List, Dict, Optional, Any
import torch
import logging

from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.features.feature_builder import FeatureBuilder
from football_brain_core.src.features.market_targets import MarketType, MARKET_OUTCOMES
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, PredictionRepository, ModelVersionRepository, MarketRepository
)
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketPredictor:
    def __init__(
        self,
        model: MultiTaskModel,
        market_types: List[MarketType],
        feature_builder: Optional[FeatureBuilder] = None
    ):
        self.model = model
        self.market_types = market_types
        self.feature_builder = feature_builder or FeatureBuilder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_match(
        self,
        match_id: int,
        session
    ) -> Dict[MarketType, Dict[str, Any]]:
        match = MatchRepository.get_by_id(session, match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")
        
        features = self.feature_builder.build_match_features(
            match.home_team_id,
            match.away_team_id,
            match.match_date,
            match.league_id,
            session
        )
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probas = self.model.predict_proba(features_tensor)
            predictions = self.model.predict(features_tensor)
        
        results = {}
        for market_type in self.market_types:
            pred_idx = predictions[market_type.value].cpu().numpy()[0]
            outcome = MARKET_OUTCOMES[market_type][pred_idx]
            probability = probas[market_type.value].cpu().numpy()[0][pred_idx]
            
            results[market_type] = {
                "outcome": outcome,
                "probability": float(probability),
                "all_probabilities": probas[market_type.value].cpu().numpy()[0].tolist()
            }
        
        return results
    
    def predict_upcoming_matches(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        league_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        session = get_session()
        try:
            if not date_from:
                date_from = date.today()
            if not date_to:
                date_to = date_from
            
            matches = MatchRepository.get_by_date_range(session, date_from, date_to)
            
            if league_ids:
                matches = [m for m in matches if m.league_id in league_ids]
            
            matches = [m for m in matches if m.home_score is None or m.away_score is None]
            
            predictions = []
            for match in matches:
                try:
                    match_predictions = self.predict_match(match.id, session)
                    predictions.append({
                        "match_id": match.id,
                        "match_date": match.match_date.isoformat(),
                        "predictions": {
                            market.value: result
                            for market, result in match_predictions.items()
                        }
                    })
                except Exception as e:
                    logger.error(f"Error predicting match {match.id}: {e}")
            
            return predictions
        finally:
            session.close()
    
    def save_predictions(
        self,
        match_id: int,
        predictions: Dict[MarketType, Dict[str, Any]],
        model_version_id: int
    ) -> None:
        session = get_session()
        try:
            for market_type, result in predictions.items():
                market = MarketRepository.get_or_create(
                    session, name=market_type.value
                )
                
                PredictionRepository.create(
                    session,
                    match_id=match_id,
                    market_id=market.id,
                    predicted_outcome=result["outcome"],
                    model_version_id=model_version_id,
                    p_hat=result.get("probability")
                )
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving predictions: {e}")
            raise
        finally:
            session.close()


def load_model_and_predict(
    model_path: str,
    market_types: List[MarketType],
    input_size: int,
    model_config: Dict
) -> MarketPredictor:
    model = MultiTaskModel(
        input_size=input_size,
        market_types=market_types,
        **model_config
    )
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    return MarketPredictor(model, market_types)

