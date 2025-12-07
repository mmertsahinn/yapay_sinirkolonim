"""
🧠 DEEP LEARNING KOLEKTİF ZEKA - TEMPERAMENT ENCODER
====================================================

Mizacı bir embedding space'e çeviriyoruz!

Bilimsel temel: Personality embeddings research (2024)
- Benzer mizaçlar embedding'de yakın olacak
- Neural network mizacı öğrenebilir representation'a çevirir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemperamentEncoder(nn.Module):
    """
    Mizaç vektörünü neural embedding'e çevir
    
    Input: [openness, contrarian, independence, risk, hype_sens, ...]
    Output: Dense embedding vector
    
    Neden deep learning?
    - Mizaç kombinasyonlarını öğrenir
    - Non-linear relationships yakalar
    - Transfer learning imkanı
    """
    
    def __init__(self, temperament_dim=10, embed_dim=128, dropout=0.2):
        """
        Args:
            temperament_dim: Kaç temperament trait var (default: 10)
            embed_dim: Embedding boyutu (default: 128)
            dropout: Regularization (default: 0.2)
        """
        super().__init__()
        
        self.temperament_dim = temperament_dim
        self.embed_dim = embed_dim
        
        # Multi-layer encoder
        self.encoder = nn.Sequential(
            # Layer 1: Temperament → Hidden
            nn.Linear(temperament_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: Hidden → Hidden
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3: Hidden → Embedding
            nn.Linear(192, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        print(f"✅ TemperamentEncoder initialized ({temperament_dim} → {embed_dim})")
    
    def forward(self, temperament_dict):
        """
        Mizaç dict'ini embedding'e çevir
        
        Args:
            temperament_dict: {
                'openness': 0.7,
                'contrarian_score': 0.3,
                'independence': 0.5,
                ...
            }
            
        Returns:
            embedding: [embed_dim] tensor
        """
        # Dict → Tensor
        temperament_vector = self._dict_to_tensor(temperament_dict)
        
        # Encode
        embedding = self.encoder(temperament_vector)
        
        return embedding
    
    def _dict_to_tensor(self, temp_dict):
        """Temperament dict'ini tensor'e çevir"""
        # Standard temperament features
        features = [
            temp_dict.get('openness', 0.5),
            temp_dict.get('contrarian_score', 0.5),
            temp_dict.get('independence', 0.5),
            temp_dict.get('risk_tolerance', 0.5),
            temp_dict.get('hype_sensitivity', 0.5),
            temp_dict.get('confidence', 0.5),
            temp_dict.get('stress_tolerance', 0.5),
            temp_dict.get('social_intelligence', 0.5),
            temp_dict.get('curiosity', 0.5),
            temp_dict.get('patience', 0.5)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def compute_similarity(self, emb1, emb2):
        """İki mizaç embedding'i ne kadar benzer?"""
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()


class PersonalityModulatedAttention(nn.Module):
    """
    Mizaç attention'ı nasıl etkiler?
    
    Örnek:
    - Açık mizaç (openness=0.9) → Herkese attention ver
    - Kapalı mizaç (openness=0.2) → Az attention ver
    - Karşıt mizaç (contrarian=0.9) → Ters yorumla
    
    Bilimsel: Individual differences in attention networks
    """
    
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # Attention gate: Temperament + Neighbor info → Gate value [0, 1]
        self.attention_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 arası gate
        )
        
        # Information filter: Ne kadar bilgi geçsin?
        self.info_filter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh()
        )
        
        print("✅ PersonalityModulatedAttention initialized")
    
    def forward(self, my_temp_emb, neighbor_emb):
        """
        Mizaca göre komşudan ne kadar bilgi alınacak?
        
        Args:
            my_temp_emb: Benim mizaç embedding'im [embed_dim]
            neighbor_emb: Komşunun bilgi embedding'i [embed_dim]
            
        Returns:
            filtered_info: Mizaca göre filtrelenmiş bilgi [embed_dim]
            gate_value: Kapı değeri (loglama için)
        """
        # Gate hesapla
        combined = torch.cat([my_temp_emb, neighbor_emb], dim=-1)
        gate = self.attention_gate(combined)
        
        # Bilgiyi filtrele
        filtered = self.info_filter(neighbor_emb)
        
        # Gate uygula
        filtered_info = gate * filtered
        
        return filtered_info, gate.item()


# Global instance (lazy initialization için)
_global_temp_encoder = None
_global_personality_attn = None


def get_temperament_encoder(embed_dim=128):
    """Global temperament encoder instance"""
    global _global_temp_encoder
    if _global_temp_encoder is None:
        _global_temp_encoder = TemperamentEncoder(
            temperament_dim=10,
            embed_dim=embed_dim
        )
    return _global_temp_encoder


def get_personality_attention(embed_dim=128):
    """Global personality-modulated attention instance"""
    global _global_personality_attn
    if _global_personality_attn is None:
        _global_personality_attn = PersonalityModulatedAttention(embed_dim)
    return _global_personality_attn
