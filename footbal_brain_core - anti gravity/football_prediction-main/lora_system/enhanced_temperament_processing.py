"""
🎭 ENHANCED TEMPERAMENT PROCESSING
====================================

Mizaca göre bilgi işleme: Açık mizaç yeni bilgilere açık, bağımsız kendi yorumunu yapar,
risk toleransı ne kadar adapte olacağını belirler.

Bilimsel Temel:
- Personality-modulated attention
- Information filtering based on temperament
- Adaptive information processing

Özellikler:
✅ Mizaca göre bilgi işleme
✅ Personality-modulated knowledge transfer
✅ Attention mechanisms with temperament influence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .temperament_encoder import TemperamentEncoder, PersonalityModulatedAttention


class TemperamentBasedFilter(nn.Module):
    """
    Mizaca göre bilgi filtreleme
    
    Her LoRA mizacına göre bilgiyi farklı şekilde işler
    """
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Filter gates: Mizaç → Information gate
        self.openness_gate = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),  # info + openness
            nn.Sigmoid()
        )
        
        self.contrarian_gate = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),  # info + contrarian
            nn.Tanh()  # -1 ile +1 arası (ters yorumlama)
        )
        
        self.independence_gate = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),
            nn.Sigmoid()
        )
        
        print(f"✅ TemperamentBasedFilter initialized")
    
    def filter_information(self,
                          information: torch.Tensor,
                          temperament: Dict) -> torch.Tensor:
        """
        Bilgiyi mizaca göre filtrele
        
        Args:
            information: Information embedding [embed_dim]
            temperament: Temperament dict
            
        Returns:
            Filtered information
        """
        # Get temperament values
        openness = temperament.get('openness', temperament.get('risk_tolerance', 0.5))
        contrarian = temperament.get('contrarian_score', 0.3)
        independence = temperament.get('independence', 0.5)
        
        # Openness gate: Açık mizaç daha fazla bilgi alır
        openness_input = torch.cat([information, torch.tensor([openness], device=information.device)])
        openness_filter = self.openness_gate(openness_input)
        filtered = information * openness_filter
        
        # Contrarian gate: Karşıt mizaç ters yorumlar
        contrarian_input = torch.cat([information, torch.tensor([contrarian], device=information.device)])
        contrarian_filter = self.contrarian_gate(contrarian_input)
        filtered = filtered * (1 + contrarian_filter * contrarian)  # Contrarian yüksekse ters etki
        
        # Independence gate: Bağımsız mizaç kendi yorumunu yapar
        independence_input = torch.cat([information, torch.tensor([independence], device=information.device)])
        independence_filter = self.independence_gate(independence_input)
        filtered = filtered * (0.5 + 0.5 * (1 - independence_filter))  # Independence yüksekse az etki
        
        return filtered


class PersonalityModulatedKnowledgeTransfer(nn.Module):
    """
    Personality-modulated knowledge transfer
    
    Bilgi transferinde mizacın etkisi
    """
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Compatibility scorer: İki LoRA'nın mizaç uyumluluğu
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),  # temp1 + temp2
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Transfer strength modulator
        self.transfer_modulator = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),  # knowledge + compatibility
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Temperament encoder
        self.temp_encoder = TemperamentEncoder(embed_dim=embed_dim)
        
        print(f"✅ PersonalityModulatedKnowledgeTransfer initialized")
    
    def compute_transfer_strength(self,
                                  source_temp: Dict,
                                  target_temp: Dict) -> float:
        """
        Transfer gücünü hesapla (mizaca göre)
        
        Args:
            source_temp: Kaynak LoRA'nın mizacı
            target_temp: Hedef LoRA'nın mizacı
            
        Returns:
            Transfer strength (0-1)
        """
        # Encode temperaments
        source_emb = self.temp_encoder(source_temp)
        target_emb = self.temp_encoder(target_temp)
        
        # Compatibility
        combined = torch.cat([source_emb, target_emb], dim=-1)
        compatibility = self.compatibility_scorer(combined)
        
        return compatibility.item()
    
    def modulate_transfer(self,
                         knowledge: torch.Tensor,
                         source_temp: Dict,
                         target_temp: Dict) -> torch.Tensor:
        """
        Bilgi transferini mizaca göre modüle et
        
        Args:
            knowledge: Knowledge embedding
            source_temp: Kaynak mizaç
            target_temp: Hedef mizaç
            
        Returns:
            Modulated knowledge
        """
        # Transfer strength
        transfer_strength = self.compute_transfer_strength(source_temp, target_temp)
        
        # Target temperament values
        target_openness = target_temp.get('openness', target_temp.get('risk_tolerance', 0.5))
        target_independence = target_temp.get('independence', 0.5)
        
        # Modulate based on target temperament
        # Açık mizaç → Daha fazla alır
        # Bağımsız mizaç → Daha az alır, kendi yorumunu yapar
        modulation_factor = target_openness * (1 - target_independence * 0.5)
        modulation_factor *= transfer_strength
        
        modulated = knowledge * modulation_factor
        
        return modulated


class EnhancedTemperamentProcessing:
    """
    Enhanced Temperament Processing System
    
    Tüm mizaç bazlı işlemleri yönetir
    """
    
    def __init__(self, embed_dim: int = 128, device='cpu'):
        """
        Args:
            embed_dim: Embedding boyutu
            device: Device
        """
        self.embed_dim = embed_dim
        self.device = device
        
        # Components
        self.temperament_encoder = TemperamentEncoder(embed_dim=embed_dim).to(device)
        self.personality_attention = PersonalityModulatedAttention(embed_dim=embed_dim).to(device)
        self.temperament_filter = TemperamentBasedFilter(embed_dim=embed_dim).to(device)
        self.knowledge_transfer = PersonalityModulatedKnowledgeTransfer(embed_dim=embed_dim).to(device)
        
        print("="*80)
        print("🎭 ENHANCED TEMPERAMENT PROCESSING INITIALIZED")
        print("="*80)
    
    def process_information_with_temperament(self,
                                            information: torch.Tensor,
                                            lora_temperament: Dict) -> torch.Tensor:
        """
        Bilgiyi mizaca göre işle
        
        Args:
            information: Information embedding
            lora_temperament: LoRA'nın mizacı
            
        Returns:
            Processed information
        """
        # Filter based on temperament
        filtered = self.temperament_filter.filter_information(
            information, lora_temperament
        )
        
        return filtered
    
    def transfer_knowledge_with_temperament(self,
                                           knowledge: torch.Tensor,
                                           source_lora,
                                           target_lora) -> Tuple[torch.Tensor, float]:
        """
        Bilgiyi mizaca göre transfer et
        
        Args:
            knowledge: Knowledge embedding
            source_lora: Kaynak LoRA
            target_lora: Hedef LoRA
            
        Returns:
            (transferred_knowledge, transfer_strength)
        """
        source_temp = getattr(source_lora, 'temperament', {})
        target_temp = getattr(target_lora, 'temperament', {})
        
        # Modulate transfer
        modulated = self.knowledge_transfer.modulate_transfer(
            knowledge, source_temp, target_temp
        )
        
        # Transfer strength
        transfer_strength = self.knowledge_transfer.compute_transfer_strength(
            source_temp, target_temp
        )
        
        return modulated, transfer_strength
    
    def compute_attention_with_temperament(self,
                                          my_temperament: Dict,
                                          neighbor_info: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Mizaca göre attention hesapla
        
        Args:
            my_temperament: Benim mizacım
            neighbor_info: Komşu bilgisi
            
        Returns:
            (attention_output, gate_value)
        """
        # Encode temperament
        my_temp_emb = self.temperament_encoder(my_temperament)
        
        # Personality-modulated attention
        filtered_info, gate = self.personality_attention(
            my_temp_emb, neighbor_info
        )
        
        return filtered_info, gate
    
    def get_adoption_rate(self,
                         discovery_temperament: Dict,
                         target_temperament: Dict,
                         discovery_accuracy: float) -> float:
        """
        Keşfi kabul oranını hesapla (mizaca göre)
        
        Args:
            discovery_temperament: Keşfeden LoRA'nın mizacı
            target_temperament: Hedef LoRA'nın mizacı
            discovery_accuracy: Keşif doğruluğu
            
        Returns:
            Adoption rate (0-1)
        """
        # Base adoption from accuracy
        base_adoption = discovery_accuracy
        
        # Temperament factors
        target_openness = target_temperament.get('openness', target_temperament.get('risk_tolerance', 0.5))
        target_contrarian = target_temperament.get('contrarian_score', 0.3)
        target_independence = target_temperament.get('independence', 0.5)
        
        # Compatibility
        compatibility = self.knowledge_transfer.compute_transfer_strength(
            discovery_temperament, target_temperament
        )
        
        # Final adoption rate
        adoption = base_adoption * target_openness * (1 - target_contrarian * 0.5) * (1 - target_independence * 0.3)
        adoption *= compatibility
        
        return min(1.0, max(0.0, adoption))
    
    def evolve_temperament_processing(self,
                                     lora,
                                     performance: float,
                                     is_success: bool):
        """
        Mizaç işlemeyi evril (performansa göre)
        
        Args:
            lora: LoRA instance
            performance: Performans skoru
            is_success: Başarılı mı?
        """
        if not hasattr(lora, 'temperament'):
            return
        
        # Performansa göre mizaç ayarı (basit örnek)
        if is_success and performance > 0.8:
            # Başarılı → Açıklık artabilir
            if 'openness' in lora.temperament:
                lora.temperament['openness'] = min(1.0, lora.temperament['openness'] + 0.05)
        elif not is_success and performance < 0.3:
            # Başarısız → Daha kapalı olabilir veya daha bağımsız
            if 'independence' in lora.temperament:
                lora.temperament['independence'] = min(1.0, lora.temperament['independence'] + 0.05)


# Global instance
_global_temperament_processing = None


def get_enhanced_temperament_processing(embed_dim: int = 128, device='cpu') -> EnhancedTemperamentProcessing:
    """Global enhanced temperament processing instance"""
    global _global_temperament_processing
    if _global_temperament_processing is None:
        _global_temperament_processing = EnhancedTemperamentProcessing(embed_dim, device)
    return _global_temperament_processing


