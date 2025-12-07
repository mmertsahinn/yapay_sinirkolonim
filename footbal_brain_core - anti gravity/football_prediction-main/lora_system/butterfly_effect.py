"""
🦋 KELEBEK ETKİSİ (Kaotik Determinizm Kırıcı)
==============================================

Plan'dan:
"Bir LoRA'nın küçük bir ağırlık değişimi, sosyal ağdaki komşularında 
dalgalanma (noise injection) yaratacak."

Mantık:
- LoRA öğrenir → Ağırlıkları değişir
- Komşularına küçük noise injection yapılır
- Bu noise, komşuların da öğrenmesini tetikler
- Kaotik determinizm kırılır (sürpriz keşifler!)
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import random


class ButterflyEffect:
    """
    Kelebek Etkisi: Bir LoRA'nın değişimi komşularını etkiler
    """
    
    def __init__(self, noise_strength: float = 0.01, propagation_depth: int = 1):
        """
        Args:
            noise_strength: Noise injection gücü (0.01 = %1)
            propagation_depth: Kaç seviye komşuya yayılacak (1 = sadece direkt komşular)
        """
        self.noise_strength = noise_strength
        self.propagation_depth = propagation_depth
        
        print(f"🦋 Butterfly Effect başlatıldı (noise={noise_strength}, depth={propagation_depth})")
    
    def apply_butterfly_effect(self,
                               changed_lora,
                               social_network,
                               population: List,
                               change_magnitude: float = None) -> Dict:
        """
        Bir LoRA değiştiğinde komşularına noise injection yap
        
        Args:
            changed_lora: Değişen LoRA
            social_network: Sosyal ağ (komşuları bulmak için)
            population: Tüm popülasyon
            change_magnitude: Değişim büyüklüğü (None ise otomatik hesaplanır)
        
        Returns:
            {
                'affected_loras': [lora1, lora2, ...],
                'noise_injected': True/False,
                'propagation_count': int
            }
        """
        if social_network is None:
            return {'affected_loras': [], 'noise_injected': False, 'propagation_count': 0}
        
        # 1. Komşuları bul
        neighbors = self._get_neighbors(changed_lora, social_network, population)
        
        if not neighbors:
            return {'affected_loras': [], 'noise_injected': False, 'propagation_count': 0}
        
        # 2. Değişim büyüklüğünü hesapla (yoksa otomatik)
        if change_magnitude is None:
            change_magnitude = self._calculate_change_magnitude(changed_lora)
        
        # 3. Noise injection gücünü ayarla (değişim büyüklüğüne göre)
        effective_noise = self.noise_strength * min(change_magnitude, 1.0)
        
        # 4. Komşulara noise injection yap
        affected_loras = []
        
        for neighbor in neighbors:
            try:
                self._inject_noise(neighbor, effective_noise)
                affected_loras.append(neighbor)
            except Exception as e:
                # Hata varsa devam et
                continue
        
        # 5. Derinlik > 1 ise, komşuların komşularına da yayıl
        if self.propagation_depth > 1 and affected_loras:
            for neighbor in affected_loras:
                # Komşunun komşularına da noise injection (daha az güçlü)
                sub_neighbors = self._get_neighbors(neighbor, social_network, population)
                for sub_neighbor in sub_neighbors:
                    if sub_neighbor.id != changed_lora.id and sub_neighbor not in affected_loras:
                        try:
                            # Daha az güçlü noise (derinlik arttıkça azalır)
                            sub_noise = effective_noise * (0.5 ** (self.propagation_depth - 1))
                            self._inject_noise(sub_neighbor, sub_noise)
                            affected_loras.append(sub_neighbor)
                        except:
                            continue
        
        return {
            'affected_loras': affected_loras,
            'noise_injected': len(affected_loras) > 0,
            'propagation_count': len(affected_loras)
        }
    
    def _get_neighbors(self, lora, social_network, population: List) -> List:
        """Sosyal ağdan komşuları bul"""
        neighbors = []
        
        try:
            # Sosyal bağları kontrol et
            if hasattr(lora, 'social_bonds') and lora.social_bonds:
                for neighbor_id, bond_strength in lora.social_bonds.items():
                    # Güçlü bağlar öncelikli (bond_strength > 0.3)
                    if bond_strength > 0.3:
                        neighbor = next((l for l in population if l.id == neighbor_id), None)
                        if neighbor:
                            neighbors.append(neighbor)
            
            # Sosyal ağ sisteminden de komşuları al (varsa)
            if hasattr(social_network, 'get_neighbors'):
                network_neighbors = social_network.get_neighbors(lora.id)
                for neighbor_id in network_neighbors:
                    neighbor = next((l for l in population if l.id == neighbor_id), None)
                    if neighbor and neighbor not in neighbors:
                        neighbors.append(neighbor)
        except Exception:
            # Hata varsa boş liste dön
            pass
        
        return neighbors
    
    def _calculate_change_magnitude(self, lora) -> float:
        """LoRA'nın son değişim büyüklüğünü hesapla"""
        try:
            # Son parametre değişimini kontrol et
            if hasattr(lora, '_last_param_change'):
                return min(lora._last_param_change, 1.0)
            
            # Alternatif: Son loss değişimi
            if hasattr(lora, '_last_loss') and hasattr(lora, '_previous_loss'):
                loss_change = abs(lora._last_loss - lora._previous_loss)
                return min(loss_change, 1.0)
        except:
            pass
        
        # Varsayılan: Orta seviye değişim
        return 0.5
    
    def _inject_noise(self, lora, noise_strength: float):
        """
        LoRA'ya noise injection yap (ağırlıklara küçük rastgele değişim)
        
        Args:
            lora: Noise injection yapılacak LoRA
            noise_strength: Noise gücü (0.01 = %1)
        """
        with torch.no_grad():
            for name, param in lora.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    # LoRA parametrelerine noise ekle
                    noise = torch.randn_like(param) * noise_strength
                    param.data.add_(noise)
                    
                    # Clamp (çok büyük değerler olmasın)
                    if 'lora_A' in name:
                        param.data.clamp_(-2.0, 2.0)
                    elif 'lora_B' in name:
                        param.data.clamp_(-2.0, 2.0)
    
    def apply_learning_trigger(self,
                              learning_lora,
                              social_network,
                              population: List) -> Dict:
        """
        Bir LoRA öğrendiğinde komşularını da tetikle (öğrenmeye teşvik et)
        
        Bu, Kelebek Etkisi'nin öğrenme versiyonu:
        - LoRA öğrenir → Komşularına "sen de öğren" sinyali gönder
        - Komşuların learning rate'i geçici olarak artar
        """
        neighbors = self._get_neighbors(learning_lora, social_network, population)
        
        if not neighbors:
            return {'triggered': 0}
        
        triggered_count = 0
        
        for neighbor in neighbors:
            try:
                # Komşunun learning rate'ini geçici olarak artır
                if hasattr(neighbor, '_base_learning_rate'):
                    # %10 artır (geçici)
                    neighbor._temporary_lr_boost = neighbor._base_learning_rate * 1.1
                    neighbor._lr_boost_remaining = 3  # 3 maç süreyle
                    triggered_count += 1
            except:
                continue
        
        return {'triggered': triggered_count}


# Global instance
butterfly_effect = ButterflyEffect()

