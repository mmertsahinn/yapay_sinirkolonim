"""
🕸️ GRAPH ATTENTION NETWORK - SOSYAL BAĞLAR
============================================

GAT ile LoRAlar birbirlerine "dikkat" ediyor!

Bilimsel temel: Graph Attention Networks (Veličković et al., 2018)
- Her LoRA komşularına farklı attention weights veriyor
- Dinamik: Attention öğrenilebilir
- Scalable: 100+ LoRA ile çalışır

Neden GAT?
✅ Sosyal bağlar bir graf (nodes = LoRAlar, edges = bağlar)
✅ Attention mechanism: Önemli komşulara focus
✅ Multi-head: Farklı perspektifler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class SocialAttentionLayer(nn.Module):
    """
    Graph Attention Network for LoRA social bonds
    
    Her LoRA, komşularına öğrenilebilir attention ağırlıkları verir
    
    Formula:
    α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))
    h'_i = σ(Σ α_ij W h_j)
    """
    
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        """
        Args:
            embed_dim: LoRA embedding boyutu
            num_heads: Multi-head attention sayısı
            dropout: Regularization
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ SocialAttentionLayer initialized (dim={embed_dim}, heads={num_heads})")
    
    def forward(self, lora_embeddings, adjacency_matrix=None):
        """
        Sosyal attention uygula
        
        Args:
            lora_embeddings: [N x embed_dim] Her LoRA'nın embedding'i
            adjacency_matrix: [N x N] Sosyal bağ matrisi (optional)
                             1.0 = güçlü bağ, 0.0 = bağ yok
                             
        Returns:
            updated_embeddings: [N x embed_dim] Sosyal context ile güncellenmiş
            attention_weights: [N x N] Öğrenilmiş attention ağırlıkları
        """
        # Add batch dimension if needed
        if lora_embeddings.dim() == 2:
            lora_embeddings = lora_embeddings.unsqueeze(0)  # [1 x N x embed_dim]
        
        # Attention mask from adjacency (optional)
        attn_mask = None
        if adjacency_matrix is not None:
            # Mask out weak bonds (< 0.3)
            attn_mask = adjacency_matrix < 0.3
            # Convert to attention mask format (-inf for masked positions)
            attn_mask = attn_mask.float() * -1e9
        
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(
            lora_embeddings,
            lora_embeddings,
            lora_embeddings,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        # Residual + Norm
        x = self.norm1(lora_embeddings + self.dropout(attn_output))
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        # Remove batch dimension
        updated_embeddings = x.squeeze(0)  # [N x embed_dim]
        
        return updated_embeddings, attn_weights.squeeze(0)
    
    def get_attention_statistics(self, attn_weights, lora_ids):
        """
        Attention istatistikleri (debugging/logging için)
        
        Returns:
            Dict with attention statistics
        """
        stats = {
            'mean_attention': attn_weights.mean().item(),
            'max_attention': attn_weights.max().item(),
            'min_attention': attn_weights.min().item(),
            'sparsity': (attn_weights < 0.1).float().mean().item()
        }
        
        # En güçlü attention'lar
        flat_attn = attn_weights.flatten()
        top_k = min(5, len(flat_attn))
        top_values, top_indices = torch.topk(flat_attn, top_k)
        
        stats['top_attentions'] = []
        for val, idx in zip(top_values, top_indices):
            i = idx // len(lora_ids)
            j = idx % len(lora_ids)
            stats['top_attentions'].append({
                'from': lora_ids[i],
                'to': lora_ids[j],
                'weight': val.item()
            })
        
        return stats


class SocialGraphBuilder:
    """
    Sosyal bağ matrisini oluştur
    
    Bond strength'i hesapla:
    - Prediction similarity
    - Success correlation  
    - Shared experiences
    """
    
    @staticmethod
    def build_adjacency_matrix(population: List, social_network) -> torch.Tensor:
        """
        Popülasyondan adjacency matrix oluştur
        
        Args:
            population: LoRA listesi
            social_network: SocialNetwork instance
            
        Returns:
            adjacency: [N x N] tensor, bond strengths
        """
        N = len(population)
        adjacency = torch.zeros(N, N)
        
        # LoRA ID → index mapping
        id_to_idx = {lora.id: i for i, lora in enumerate(population)}
        
        # Sosyal ağdan bond'ları al
        if hasattr(social_network, 'bonds'):
            for (id1, id2), strength in social_network.bonds.items():
                if id1 in id_to_idx and id2 in id_to_idx:
                    i = id_to_idx[id1]
                    j = id_to_idx[id2]
                    adjacency[i, j] = strength
                    adjacency[j, i] = strength  # Symmetric
        
        # Self-loops (her LoRA kendine bağlı)
        for i in range(N):
            adjacency[i, i] = 1.0
        
        return adjacency


# Global instance
_global_social_attention = None


def get_social_attention(embed_dim=128, num_heads=4):
    """Global social attention layer instance"""
    global _global_social_attention
    if _global_social_attention is None:
        _global_social_attention = SocialAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
    return _global_social_attention
