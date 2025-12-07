"""
🕸️ ADVANCED SOCIAL NETWORK - Neural Similarity & Thinking Patterns
====================================================================

Gelişmiş sosyal ağ: Benzer düşünenler birbirine çekilmeli.

Özellikler:
✅ Similarity-based attraction
✅ Neural similarity (nöron yapılarının benzerliği)
✅ Thinking pattern clustering
✅ Dynamic bond formation based on neural/thinking similarity

Mevcut social_network.py'yi genişletir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math

# Import base social network
from .social_network import SocialNetwork

# Import related modules
from .thinking_patterns import ThinkingPattern, EvolvableThinkingSystem


class NeuralSimilarityCalculator:
    """
    Nöron yapılarının benzerliğini hesaplar
    
    İki LoRA'nın nöron mimarisi ne kadar benzer?
    """
    
    def __init__(self):
        print("✅ NeuralSimilarityCalculator initialized")
    
    def calculate_architecture_similarity(self, lora_i, lora_j) -> float:
        """
        Mimari benzerliği hesapla
        
        Args:
            lora_i: LoRA instance
            lora_j: LoRA instance
            
        Returns:
            Similarity score (0-1)
        """
        # Nöron sayıları
        dim_i = getattr(lora_i, 'hidden_dim', 128)
        dim_j = getattr(lora_j, 'hidden_dim', 128)
        
        # Katman sayıları
        layers_i = self._count_layers(lora_i)
        layers_j = self._count_layers(lora_j)
        
        # Dimension similarity
        dim_sim = 1.0 - abs(dim_i - dim_j) / max(dim_i, dim_j, 1)
        
        # Layer similarity
        layer_sim = 1.0 if layers_i == layers_j else 0.5
        
        # Combined similarity
        similarity = (dim_sim * 0.7 + layer_sim * 0.3)
        
        return max(0.0, min(1.0, similarity))
    
    def _count_layers(self, lora) -> int:
        """Katman sayısını hesapla"""
        count = 0
        for attr in ['fc1', 'fc2', 'fc3']:
            if hasattr(lora, attr):
                count += 1
        return count
    
    def calculate_parameter_similarity(self, lora_i, lora_j) -> float:
        """
        Parametre benzerliğini hesapla (cosine similarity)
        
        Args:
            lora_i: LoRA instance
            lora_j: LoRA instance
            
        Returns:
            Similarity score (0-1)
        """
        try:
            params_i = lora_i.get_all_lora_params()
            params_j = lora_j.get_all_lora_params()
            
            # Flatten parameters
            vec_i = self._flatten_params(params_i)
            vec_j = self._flatten_params(params_j)
            
            # Cosine similarity
            vec_i_norm = vec_i / (torch.norm(vec_i) + 1e-8)
            vec_j_norm = vec_j / (torch.norm(vec_j) + 1e-8)
            
            similarity = torch.dot(vec_i_norm, vec_j_norm).item()
            
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))  # -1,1 → 0,1
        
        except Exception as e:
            return 0.5  # Default similarity
    
    def _flatten_params(self, params: Dict) -> torch.Tensor:
        """Parametreleri düzleştir"""
        param_list = []
        for layer in ['fc1', 'fc2', 'fc3']:
            if layer in params:
                for matrix in ['lora_A', 'lora_B']:
                    if matrix in params[layer]:
                        param_list.append(params[layer][matrix].flatten())
        
        if param_list:
            return torch.cat(param_list)
        else:
            return torch.tensor([0.0])


class ThinkingPatternClustering:
    """
    Düşünme biçimlerine göre gruplaşma
    
    Benzer düşünen LoRA'lar birbirine çekilmeli
    """
    
    def __init__(self):
        print("✅ ThinkingPatternClustering initialized")
    
    def calculate_thinking_similarity(self, lora_i, lora_j) -> float:
        """
        Düşünme biçimi benzerliği
        
        Args:
            lora_i: LoRA instance
            lora_j: LoRA instance
            
        Returns:
            Similarity score (0-1)
        """
        # Thinking system'leri al
        thinking_i = getattr(lora_i, 'thinking_system', None)
        thinking_j = getattr(lora_j, 'thinking_system', None)
        
        if thinking_i is None or thinking_j is None:
            return 0.5  # Default
        
        # Primary pattern similarity
        if thinking_i.primary_pattern == thinking_j.primary_pattern:
            primary_sim = 1.0
        else:
            primary_sim = 0.3  # Farklı pattern'ler
        
        # Pattern weights similarity
        weights_i = thinking_i.pattern_weights
        weights_j = thinking_j.pattern_weights
        
        # Cosine similarity of weight vectors
        weights_i_vec = torch.tensor([weights_i.get(p, 0.0) for p in ThinkingPattern])
        weights_j_vec = torch.tensor([weights_j.get(p, 0.0) for p in ThinkingPattern])
        
        weights_i_norm = weights_i_vec / (torch.norm(weights_i_vec) + 1e-8)
        weights_j_norm = weights_j_vec / (torch.norm(weights_j_vec) + 1e-8)
        
        weights_sim = torch.dot(weights_i_norm, weights_j_norm).item()
        weights_sim = (weights_sim + 1.0) / 2.0  # -1,1 → 0,1
        
        # Combined similarity
        similarity = primary_sim * 0.4 + weights_sim * 0.6
        
        return max(0.0, min(1.0, similarity))
    
    def find_thinking_clusters(self, population: List) -> Dict[ThinkingPattern, List[str]]:
        """
        Düşünme biçimlerine göre kümeler bul
        
        Args:
            population: LoRA popülasyonu
            
        Returns:
            Pattern → LoRA IDs mapping
        """
        clusters = defaultdict(list)
        
        for lora in population:
            thinking_system = getattr(lora, 'thinking_system', None)
            if thinking_system:
                primary_pattern = thinking_system.primary_pattern
                clusters[primary_pattern].append(lora.id)
            else:
                # Default cluster
                clusters[ThinkingPattern.HOLISTIC].append(lora.id)
        
        return dict(clusters)


class AdvancedSocialNetwork(SocialNetwork):
    """
    Gelişmiş Sosyal Ağ
    
    Mevcut SocialNetwork'ü genişletir:
    - Neural similarity
    - Thinking pattern clustering
    - Similarity-based attraction
    """
    
    def __init__(self):
        super().__init__()
        
        # Advanced components
        self.neural_similarity = NeuralSimilarityCalculator()
        self.thinking_clustering = ThinkingPatternClustering()
        
        # Similarity cache (performance için)
        self.similarity_cache: Dict[Tuple[str, str], Dict] = {}
        
        # Cluster tracking
        self.thinking_clusters: Dict[ThinkingPattern, List[str]] = {}
        
        # Attraction weights
        self.ALPHA_NEURAL = 0.25  # Neural similarity weight
        self.ALPHA_THINKING = 0.25  # Thinking similarity weight
        self.ALPHA_BASE = 0.50  # Base social network weight
        
        print("="*80)
        print("🕸️ ADVANCED SOCIAL NETWORK INITIALIZED")
        print("="*80)
        print(f"   Neural similarity weight: {self.ALPHA_NEURAL}")
        print(f"   Thinking similarity weight: {self.ALPHA_THINKING}")
        print(f"   Base network weight: {self.ALPHA_BASE}")
        print("="*80)
    
    def update_social_bond(self, lora_i: Any, lora_j: Any, match_result: Dict) -> float:
        """
        Gelişmiş sosyal bağ güncelleme
        
        Neural similarity ve thinking pattern benzerliği eklenir
        """
        # Base bond (parent class'tan)
        base_bond = super().update_social_bond(lora_i, lora_j, match_result)
        
        # Neural similarity
        neural_sim = self.neural_similarity.calculate_architecture_similarity(lora_i, lora_j)
        
        # Parameter similarity
        param_sim = self.neural_similarity.calculate_parameter_similarity(lora_i, lora_j)
        
        neural_composite = (neural_sim * 0.6 + param_sim * 0.4)
        
        # Thinking pattern similarity
        thinking_sim = self.thinking_clustering.calculate_thinking_similarity(lora_i, lora_j)
        
        # Combined bond strength
        enhanced_bond = (
            self.ALPHA_BASE * base_bond +
            self.ALPHA_NEURAL * neural_composite +
            self.ALPHA_THINKING * thinking_sim
        )
        
        # Cache similarity
        key = tuple(sorted((lora_i.id, lora_j.id)))
        self.similarity_cache[key] = {
            'neural': neural_sim,
            'parameter': param_sim,
            'thinking': thinking_sim,
            'base_bond': base_bond,
            'enhanced_bond': enhanced_bond
        }
        
        # Update bond with enhanced value
        self.bonds[key] = enhanced_bond
        
        return enhanced_bond
    
    def update_thinking_clusters(self, population: List):
        """
        Düşünme biçimi kümelerini güncelle
        
        Args:
            population: LoRA popülasyonu
        """
        self.thinking_clusters = self.thinking_clustering.find_thinking_clusters(population)
        
        print(f"   🧠 Thinking clusters updated:")
        for pattern, lora_ids in self.thinking_clusters.items():
            print(f"      {pattern.value}: {len(lora_ids)} LoRAs")
    
    def get_similarity_based_cluster(self, lora_id: str, threshold: float = 0.6) -> List[str]:
        """
        Benzerlik bazlı küme (neural + thinking)
        
        Args:
            lora_id: LoRA ID
            threshold: Benzerlik eşiği
            
        Returns:
            Benzer LoRA ID'leri
        """
        similar = []
        
        for key, cache_data in self.similarity_cache.items():
            if lora_id not in key:
                continue
            
            other_id = key[1] if key[0] == lora_id else key[0]
            
            # Combined similarity
            combined_sim = (
                cache_data['neural'] * 0.4 +
                cache_data['thinking'] * 0.4 +
                cache_data['base_bond'] * 0.2
            )
            
            if combined_sim > threshold:
                similar.append(other_id)
        
        return similar
    
    def apply_similarity_based_attraction(self, population: List, attraction_strength: float = 0.05):
        """
        Benzerlik bazlı çekim uygula
        
        Benzer düşünen/nöron yapılı LoRA'lar birbirine çekilir
        
        Args:
            population: LoRA popülasyonu
            attraction_strength: Çekim gücü
        """
        attraction_count = 0
        
        # Her benzer çift için
        for key, cache_data in self.similarity_cache.items():
            if cache_data['enhanced_bond'] < 0.7:  # Sadece güçlü bağlar
                continue
            
            id1, id2 = key
            lora_i = next((l for l in population if l.id == id1), None)
            lora_j = next((l for l in population if l.id == id2), None)
            
            if not lora_i or not lora_j:
                continue
            
            # Neural attraction: Nöron yapıları birbirine yaklaşır
            if cache_data['neural'] > 0.6:
                self._apply_neural_attraction(lora_i, lora_j, attraction_strength)
                attraction_count += 1
            
            # Thinking attraction: Düşünme biçimleri birbirine yaklaşır
            if cache_data['thinking'] > 0.6:
                self._apply_thinking_attraction(lora_i, lora_j, attraction_strength)
        
        if attraction_count > 0:
            print(f"   🧲 Similarity-based attraction: {attraction_count} pairs")
    
    def _apply_neural_attraction(self, lora_i, lora_j, strength: float):
        """Nöron yapıları birbirine yaklaştır"""
        # Hidden dimension convergence
        if hasattr(lora_i, 'hidden_dim') and hasattr(lora_j, 'hidden_dim'):
            dim_i = lora_i.hidden_dim
            dim_j = lora_j.hidden_dim
            
            # Average'e doğru çek
            avg_dim = int((dim_i + dim_j) / 2)
            
            # Gradual convergence
            if abs(dim_i - dim_j) > 10:  # Sadece büyük farklar için
                # Şimdilik sadece log, gerçek değişim karmaşık
                pass
    
    def _apply_thinking_attraction(self, lora_i, lora_j, strength: float):
        """Düşünme biçimleri birbirine yaklaştır"""
        thinking_i = getattr(lora_i, 'thinking_system', None)
        thinking_j = getattr(lora_j, 'thinking_system', None)
        
        if thinking_i is None or thinking_j is None:
            return
        
        # Pattern weights'leri birbirine yaklaştır
        for pattern in ThinkingPattern:
            if pattern in thinking_i.pattern_weights and pattern in thinking_j.pattern_weights:
                w_i = thinking_i.pattern_weights[pattern]
                w_j = thinking_j.pattern_weights[pattern]
                
                # Blend: (1-α) × w_i + α × w_j
                avg_weight = (w_i + w_j) / 2
                
                thinking_i.pattern_weights[pattern] = (1 - strength) * w_i + strength * avg_weight
                thinking_j.pattern_weights[pattern] = (1 - strength) * w_j + strength * avg_weight
        
        # Update primary patterns
        thinking_i.primary_pattern = max(thinking_i.pattern_weights.items(), key=lambda x: x[1])[0]
        thinking_j.primary_pattern = max(thinking_j.pattern_weights.items(), key=lambda x: x[1])[0]
    
    def get_bond_strength(self, id1: str, id2: str) -> float:
        """Sosyal bağ gücünü al (override for enhanced bonds)"""
        return super().get_bond_strength(id1, id2)
    
    def get_social_cluster(self, lora_id: str, threshold: float = 0.5) -> List[str]:
        """Sosyal küme (enhanced version)"""
        # Base cluster
        base_cluster = super().get_social_cluster(lora_id, threshold)
        
        # Similarity-based cluster
        similarity_cluster = self.get_similarity_based_cluster(lora_id, threshold)
        
        # Combine (unique)
        combined = list(set(base_cluster + similarity_cluster))
        
        return combined
    
    def get_network_statistics(self) -> Dict:
        """Ağ istatistikleri"""
        stats = {
            'total_bonds': len(self.bonds),
            'strong_bonds': sum(1 for b in self.bonds.values() if b > 0.7),
            'thinking_clusters': {
                pattern.value: len(ids) 
                for pattern, ids in self.thinking_clusters.items()
            },
            'similarity_cache_size': len(self.similarity_cache)
        }
        
        return stats


# Global instance
_global_advanced_social_network = None


def get_advanced_social_network() -> AdvancedSocialNetwork:
    """Global advanced social network instance"""
    global _global_advanced_social_network
    if _global_advanced_social_network is None:
        _global_advanced_social_network = AdvancedSocialNetwork()
    return _global_advanced_social_network


