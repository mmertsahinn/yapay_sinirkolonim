from datetime import datetime
from typing import List, Tuple, Optional, Dict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository
from football_brain_core.src.db.schema import Match
from football_brain_core.src.features.feature_builder import FeatureBuilder
from football_brain_core.src.features.market_targets import (
    MarketType, MARKET_OUTCOMES, calculate_all_market_outcomes
)


class FootballDataset(Dataset):
    def __init__(
        self,
        matches: List[Match],
        feature_builder: FeatureBuilder,
        market_types: List[MarketType],
        session
    ):
        self.matches = matches
        self.feature_builder = feature_builder
        self.market_types = market_types
        self.session = session
        
        self.features = []
        self.labels = {}
        
        for market_type in market_types:
            self.labels[market_type] = []
        
        self._build_dataset()
    
    def _build_dataset(self):
        for match in self.matches:
            if match.home_score is None or match.away_score is None:
                continue
            
            features = self.feature_builder.build_match_features(
                match.home_team_id,
                match.away_team_id,
                match.match_date,
                match.league_id,
                self.session
            )
            
            self.features.append(features)
            
            outcomes = calculate_all_market_outcomes(
                match.home_score,
                match.away_score
            )
            
            for market_type in self.market_types:
                outcome = outcomes.get(market_type, "")
                if outcome:
                    outcome_idx = MARKET_OUTCOMES[market_type].index(outcome)
                else:
                    outcome_idx = 0
                self.labels[market_type].append(outcome_idx)
        
        self.features = np.array(self.features)
        for market_type in self.market_types:
            self.labels[market_type] = np.array(self.labels[market_type])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        labels = {
            market_type: torch.LongTensor([self.labels[market_type][idx]])
            for market_type in self.market_types
        }
        return features, labels


def create_data_loaders(
    train_matches: List[Match],
    val_matches: List[Match],
    feature_builder: FeatureBuilder,
    market_types: List[MarketType],
    batch_size: int = 32,
    session=None
) -> Tuple[DataLoader, DataLoader]:
    if session is None:
        session = get_session()
    
    train_dataset = FootballDataset(
        train_matches, feature_builder, market_types, session
    )
    val_dataset = FootballDataset(
        val_matches, feature_builder, market_types, session
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader







