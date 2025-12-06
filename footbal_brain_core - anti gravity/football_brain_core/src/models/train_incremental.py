from datetime import datetime
from typing import List, Optional, Dict
import torch
import logging

from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.models.train_offline import OfflineTrainer
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, ModelVersionRepository
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncrementalTrainer:
    def __init__(
        self,
        market_types: List[MarketType],
        config: Optional[Config] = None
    ):
        self.config = config or Config()
        self.market_types = market_types
        self.trainer = OfflineTrainer(market_types, config)
    
    def get_new_matches_since_last_training(
        self,
        last_training_date: Optional[datetime],
        league_ids: List[int]
    ) -> List:
        session = get_session()
        try:
            if last_training_date:
                matches = []
                for league_id in league_ids:
                    all_matches = MatchRepository.get_by_league_and_season(
                        session, league_id, datetime.now().year
                    )
                    new_matches = [
                        m for m in all_matches
                        if m.match_date > last_training_date
                        and m.home_score is not None
                        and m.away_score is not None
                    ]
                    matches.extend(new_matches)
                return matches
            return []
        finally:
            session.close()
    
    def retrain(
        self,
        base_model: MultiTaskModel,
        new_matches: List,
        train_seasons: List[int],
        val_seasons: List[int],
        league_ids: List[int],
        epochs: int = 10
    ) -> MultiTaskModel:
        logger.info(f"Retraining with {len(new_matches)} new matches")
        
        session = get_session()
        try:
            existing_train_matches = []
            for league_id in league_ids:
                for season in train_seasons:
                    matches = MatchRepository.get_by_league_and_season(
                        session, league_id, season
                    )
                    existing_train_matches.extend(matches)
            
            all_train_matches = existing_train_matches + new_matches
            
            train_loader, val_loader = self.trainer.prepare_data(
                train_seasons, val_seasons, league_ids
            )
            
            model = base_model
            model.train()
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.MODEL_CONFIG.learning_rate * 0.1
            )
            criterion = torch.nn.CrossEntropyLoss()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for epoch in range(epochs):
                train_metrics = self.trainer.train_epoch(
                    model, train_loader, optimizer, criterion
                )
                val_metrics = self.trainer.validate(model, val_loader, criterion)
                
                logger.info(
                    f"Incremental Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['total_loss']:.4f}"
                )
            
            return model
        finally:
            session.close()
    
    def should_update_model(
        self,
        new_model: MultiTaskModel,
        current_model: MultiTaskModel,
        val_loader
    ) -> bool:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.CrossEntropyLoss()
        
        new_metrics = self.trainer.validate(new_model, val_loader, criterion)
        current_metrics = self.trainer.validate(current_model, val_loader, criterion)
        
        improvement_threshold = 0.02
        
        if new_metrics["total_loss"] < current_metrics["total_loss"] * (1 - improvement_threshold):
            logger.info("New model shows significant improvement")
            return True
        
        logger.info("New model does not show significant improvement")
        return False

