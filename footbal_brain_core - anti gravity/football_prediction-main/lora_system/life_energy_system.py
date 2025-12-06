"""
⚡ LIFE ENERGY SYSTEM (Yaşam Enerjisi!)
========================================

Her LoRA bir parçacık gibi!
Yaşam enerjisi var, enerji biterse sönümlenir!

ENERJI KAYNAKLARI:
+ Darwin (Popülasyona katkı)
+ Einstein (Sürpriz başarısı)
- Newton (Öğrenme maliyeti)
+ Sosyal bağlar
- Travma

DOĞAL ÖLÜM: Energy <= 0
"""

import numpy as np
from typing import Dict, List


class LifeEnergySystem:
    """
    Yaşam enerjisi sistemi (Termodinamik!)
    """
    
    def __init__(self):
        # Lambda parametreleri (Ağırlıklar)
        self.λ_einstein = 1.0   # Einstein terimi ağırlığı
        self.λ_newton = 0.5     # Newton terimi ağırlığı (ceza!)
        self.λ_social = 0.3     # Sosyal bağ bonusu
        self.λ_trauma = 0.4     # Travma cezası
        
        print("⚡ Life Energy System başlatıldı")
    
    def initialize_life_energy(self, lora):
        """
        LoRA'ya başlangıç enerjisi ver
        
        Mizaç bazlı başlangıç:
        - Will to live yüksek → Daha fazla enerji
        - Resilience yüksek → Daha fazla enerji
        """
        temp = lora.temperament
        
        base_energy = 1.0
        
        # Mizaç bonusu
        will_bonus = temp.get('will_to_live', 0.5) * 0.3
        resilience_bonus = temp.get('resilience', 0.5) * 0.2
        
        initial_energy = base_energy + will_bonus + resilience_bonus
        
        lora.life_energy = initial_energy
        lora._last_kl = 0.0  # Einstein terimi için
        
        return initial_energy
    
    def calculate_energy_change(self, lora, population: List, 
                                darwin_term: float, einstein_term: float, 
                                newton_term: float, dt: float = 1.0) -> Dict:
        """
        Enerji değişimini hesapla (Master Flux!)
        
        dE = Darwin + λ₁×Einstein - λ₂×Newton + Sosyal - Travma
        
        Args:
            darwin_term: Popülasyona katkı
            einstein_term: Sürpriz başarısı
            newton_term: Öğrenme maliyeti
            dt: Zaman adımı (genelde 1.0)
        
        Returns:
            {
                'dE': Enerji değişimi,
                'new_energy': Yeni enerji,
                'status': 'alive' / 'natural_death'
            }
        """
        # TEMEL TERIMLER
        dE_darwin = darwin_term
        dE_einstein = einstein_term * self.λ_einstein
        dE_newton = newton_term * self.λ_newton  # Ceza!
        
        # SOSYAL BONUS
        dE_social = 0.0
        if hasattr(lora, 'social_bonds') and len(lora.social_bonds) > 0:
            # Güçlü bağlar → Enerji bonusu!
            max_bond = max(lora.social_bonds.values())
            dE_social = max_bond * self.λ_social
        
        # TRAVMA CEZASI
        dE_trauma = 0.0
        if hasattr(lora, 'trauma_history'):
            recent_trauma = [t for t in lora.trauma_history[-10:] if t.get('severity', 0) > 0.3]
            trauma_penalty = len(recent_trauma) * 0.05
            dE_trauma = trauma_penalty * self.λ_trauma
        
        # TOPLAM ENERJİ DEĞİŞİMİ
        dE_total = (dE_darwin + dE_einstein + dE_social) - (dE_newton + dE_trauma)
        dE_total = dE_total * dt
        
        # Yeni enerji
        current_energy = getattr(lora, 'life_energy', 1.0)
        new_energy = current_energy + dE_total
        
        # 0-2 arası sınırla
        new_energy = max(0.0, min(2.0, new_energy))
        
        lora.life_energy = new_energy
        
        # DURUM
        if new_energy <= 0:
            status = 'natural_death'
        elif new_energy < 0.3:
            status = 'critical'
        elif new_energy > 1.5:
            status = 'thriving'
        else:
            status = 'alive'
        
        return {
            'dE': dE_total,
            'new_energy': new_energy,
            'status': status,
            'breakdown': {
                'darwin': dE_darwin,
                'einstein': dE_einstein,
                'newton': -dE_newton,
                'social': dE_social,
                'trauma': -dE_trauma
            }
        }
    
    def get_energy_status(self, lora) -> Dict:
        """
        LoRA'nın enerji durumu
        """
        energy = getattr(lora, 'life_energy', 1.0)
        
        if energy >= 1.5:
            tier = "Yüksek Enerji"
            emoji = "⚡⚡"
        elif energy >= 1.0:
            tier = "Normal"
            emoji = "⚡"
        elif energy >= 0.5:
            tier = "Düşük"
            emoji = "🔋"
        elif energy > 0:
            tier = "Kritik"
            emoji = "⚠️"
        else:
            tier = "Tükenmiş"
            emoji = "💀"
        
        return {
            'energy': energy,
            'tier': tier,
            'emoji': emoji
        }


# Global instance
life_energy_system = LifeEnergySystem()



