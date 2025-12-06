from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from football_brain_core.src.features.market_targets import MarketType, get_num_classes_for_market


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        market_types: List[MarketType],
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(MultiTaskModel, self).__init__()
        
        self.market_types = market_types
        self.input_size = input_size
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        for i in range(num_layers - 1):
            self.shared_layers.add_module(
                f"layer_{i}",
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        
        self.market_heads = nn.ModuleDict()
        for market_type in market_types:
            num_classes = get_num_classes_for_market(market_type)
            self.market_heads[market_type.value] = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_output = self.shared_layers(x)
        
        outputs = {}
        for market_type in self.market_types:
            logits = self.market_heads[market_type.value](shared_output)
            outputs[market_type.value] = logits
        
        return outputs
    
    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probas = {
                market_type.value: F.softmax(outputs[market_type.value], dim=-1)
                for market_type in self.market_types
            }
        return probas
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = {
                market_type.value: torch.argmax(outputs[market_type.value], dim=-1)
                for market_type in self.market_types
            }
        return predictions







