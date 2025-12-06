from datetime import datetime
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.models.datasets import create_data_loaders
from football_brain_core.src.features.feature_builder import FeatureBuilder
from football_brain_core.src.features.market_targets import MarketType, MARKET_OUTCOMES
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfflineTrainer:
    def __init__(
        self,
        market_types: List[MarketType],
        config: Optional[Config] = None,
        model_config: Optional[Dict] = None
    ):
        self.config = config or Config()
        self.market_types = market_types
        self.model_config = model_config or {
            "hidden_size": self.config.MODEL_CONFIG.hidden_size,
            "num_layers": self.config.MODEL_CONFIG.num_layers,
            "dropout": self.config.MODEL_CONFIG.dropout,
        }
        self.feature_builder = FeatureBuilder(
            sequence_length=self.config.SEQUENCE_LENGTH
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def prepare_data(
        self,
        train_seasons: List[int],
        val_seasons: List[int],
        league_ids: List[int]
    ) -> tuple:
        session = get_session()
        try:
            train_matches = []
            val_matches = []
            
            for league_id in league_ids:
                for season in train_seasons:
                    matches = MatchRepository.get_by_league_and_season(
                        session, league_id, season
                    )
                    train_matches.extend(matches)
                
                for season in val_seasons:
                    matches = MatchRepository.get_by_league_and_season(
                        session, league_id, season
                    )
                    val_matches.extend(matches)
            
            train_loader, val_loader = create_data_loaders(
                train_matches,
                val_matches,
                self.feature_builder,
                self.market_types,
                batch_size=self.config.MODEL_CONFIG.batch_size,
                session=session
            )
            
            return train_loader, val_loader
        finally:
            session.close()
    
    def train_epoch(
        self,
        model: MultiTaskModel,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        model.train()
        total_loss = 0.0
        market_losses = {market.value: 0.0 for market in self.market_types}
        
        for features, labels in train_loader:
            features = features.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = 0.0
            for market_type in self.market_types:
                # squeeze() yerine view(-1) kullan - batch boyutunu koru
                labels_tensor = labels[market_type].view(-1).to(self.device)
                # Boş batch kontrolü
                if labels_tensor.numel() == 0:
                    continue
                market_loss = criterion(outputs[market_type.value], labels_tensor)
                loss += market_loss
                market_losses[market_type.value] += market_loss.item()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_market_losses = {
            market: loss / len(train_loader)
            for market, loss in market_losses.items()
        }
        
        return {"total_loss": avg_loss, **avg_market_losses}
    
    def validate(
        self,
        model: MultiTaskModel,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        market_losses = {market.value: 0.0 for market in self.market_types}
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                outputs = model(features)
                
                loss = 0.0
                for market_type in self.market_types:
                    # squeeze() yerine view(-1) kullan - batch boyutunu koru
                    labels_tensor = labels[market_type].view(-1).to(self.device)
                    # Boş batch kontrolü
                    if labels_tensor.numel() == 0:
                        continue
                    market_loss = criterion(outputs[market_type.value], labels_tensor)
                    loss += market_loss
                    market_losses[market_type.value] += market_loss.item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_market_losses = {
            market: loss / len(val_loader)
            for market, loss in market_losses.items()
        }
        
        return {"total_loss": avg_loss, **avg_market_losses}
    
    def train(
        self,
        train_seasons: List[int],
        val_seasons: List[int],
        league_ids: List[int],
        epochs: Optional[int] = None
    ) -> MultiTaskModel:
        epochs = epochs or self.config.MODEL_CONFIG.epochs
        
        logger.info("Preparing data...")
        train_loader, val_loader = self.prepare_data(
            train_seasons, val_seasons, league_ids
        )
        
        input_size = next(iter(train_loader))[0].shape[1]
        model = MultiTaskModel(
            input_size=input_size,
            market_types=self.market_types,
            **self.model_config
        ).to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.MODEL_CONFIG.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.validate(model, val_loader, criterion)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}"
            )
            
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed")
        return model


