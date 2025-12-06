"""
👻 GHOST FIELDS (Hayalet Alanlar!)
====================================

Ölü LoRA'ların parametreleri "hayalet alan" oluşturur!
Yeni nesil bu alanlardan etkilenir!

ATAYA SAYGI TERİMİ:
L_total = L_match + γ × ||θ_child - θ_ancestor||²

Mantık:
- Einstein öldü ama parametreleri hayalet olarak kalıyor
- Yeni nesil Einstein'dan çok sapamaz (bağışıklık!)
- Ama taklit de etmez (özgürlük!)

γ = 0.1 (Hafif bağ)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class GhostFields:
    """
    Hayalet alanlar (Ölü LoRA'ların etkisi!)
    """
    
    def __init__(self, γ: float = 0.1):
        self.γ = γ  # Ataya saygı katsayısı
        
        # Hayalet parametreler (Ölü LoRA'lar)
        self.ghost_parameters = {}  # lora_id -> params
        self.ghost_influence = {}   # lora_id -> influence_score
        
        # Hayalet kayıt defteri (Registry)
        self.ghost_registry = {}  # {ghost_id: {'params': tensor, 'tes_score': float}}
        
        print(f"👻 Ghost Fields başlatıldı (γ={γ})")
    
    def calculate_ghost_potential(
        self,
        lora_params: torch.Tensor,
        sigma: float = 0.5
    ) -> float:
        """
        Hayalet potansiyel alanını hesapla!
        
        U_ghost(θ) = Σ_i w_i × exp(-||θ - θ_ancestor_i||^2 / σ^2)
        
        Nerede:
          • θ: Mevcut parametreler
          • θ_ancestor_i: i. ata parametreleri
          • w_i: Ata ağırlığı (TES skoruna göre!)
          • σ: Alan genişliği
        
        Returns:
            Potansiyel enerji (Yüksek = Atalardan uzak!)
        """
        if len(self.ghost_registry) == 0:
            return 0.0
        
        total_potential = 0.0
        sigma_squared = sigma ** 2
        
        for ghost_id, ghost_data in self.ghost_registry.items():
            ghost_params = ghost_data['params']
            ghost_weight = ghost_data.get('tes_score', 0.5)
            
            try:
                # Parametre farkı
                diff = lora_params - ghost_params
                distance_squared = torch.sum(diff ** 2).item()
                
                # Gaussian potansiyel
                potential_i = ghost_weight * torch.exp(
                    torch.tensor(-distance_squared / sigma_squared)
                ).item()
                
                total_potential += potential_i
            except:
                pass
        
        return total_potential
    
    def register_ghost(self, dead_lora, influence_score: float = None, tes_score: float = None):
        """
        Ölen LoRA'yı hayalet olarak kaydet!
        
        Args:
            dead_lora: Ölen LoRA
            influence_score: Etki skoru (fitness bazlı)
            tes_score: TES skoru (fiziksel ağırlık için!)
        """
        # Parametreleri kaydet
        params = dead_lora.get_all_lora_params()
        
        # Detach ve CPU'ya taşı (hafıza için)
        ghost_params = {}
        
        # Registry'ye de ekle (Potansiyel bariyer için!)
        self.ghost_registry[dead_lora.id] = {
            'params': params.detach().cpu() if isinstance(params, torch.Tensor) else params,
            'tes_score': tes_score if tes_score is not None else getattr(dead_lora, 'tes_scores', {}).get('total_tes', 0.5)
        }
        for layer in ['fc1', 'fc2', 'fc3']:
            ghost_params[layer] = {}
            for matrix in ['lora_A', 'lora_B']:
                ghost_params[layer][matrix] = params[layer][matrix].detach().cpu().clone()
        
        self.ghost_parameters[dead_lora.id] = ghost_params
        
        # Etki skoru (Fitness bazlı)
        if influence_score is None:
            influence_score = dead_lora.get_recent_fitness()
        
        self.ghost_influence[dead_lora.id] = influence_score
        
        print(f"   👻 {dead_lora.name} hayalet oldu (Etki: {influence_score:.3f})")
    
    def calculate_ancestor_respect_loss(self, child_lora, ancestor_ids: List[str] = None) -> float:
        """
        ATAYA SAYGI TERİMİ!
        
        L_respect = γ × ||θ_child - θ_ancestor||²
        
        Args:
            child_lora: Çocuk LoRA
            ancestor_ids: Hangi ataları dinle? (None = En güçlü 3)
        
        Returns:
            Respect loss (0+)
        """
        if len(self.ghost_parameters) == 0:
            return 0.0  # Hayalet yok
        
        # En güçlü hayaletleri seç
        if ancestor_ids is None:
            # En yüksek influence'a sahip 3 hayalet
            sorted_ghosts = sorted(
                self.ghost_influence.items(),
                key=lambda x: x[1],
                reverse=True
            )
            ancestor_ids = [ghost_id for ghost_id, _ in sorted_ghosts[:3]]
        
        # Çocuğun parametreleri
        child_params = child_lora.get_all_lora_params()
        
        total_distance = 0.0
        count = 0
        
        for ancestor_id in ancestor_ids:
            if ancestor_id not in self.ghost_parameters:
                continue
            
            ghost_params = self.ghost_parameters[ancestor_id]
            
            # Her layer için mesafe hesapla
            for layer in ['fc1', 'fc2', 'fc3']:
                for matrix in ['lora_A', 'lora_B']:
                    child_tensor = child_params[layer][matrix]
                    ghost_tensor = ghost_params[layer][matrix].to(child_tensor.device)
                    
                    # L2 mesafe
                    distance = torch.norm(child_tensor - ghost_tensor).item()
                    total_distance += distance
                    count += 1
        
        # Ortalama mesafe
        avg_distance = total_distance / count if count > 0 else 0.0
        
        # Respect loss
        respect_loss = self.γ * (avg_distance ** 2)
        
        return respect_loss
    
    def get_closest_ancestor(self, lora) -> Optional[Tuple[str, float]]:
        """
        En yakın hayalet atayı bul!
        
        Returns:
            (ancestor_id, distance)
        """
        if len(self.ghost_parameters) == 0:
            return None
        
        lora_params = lora.get_all_lora_params()
        
        min_distance = float('inf')
        closest_ancestor = None
        
        for ghost_id, ghost_params in self.ghost_parameters.items():
            total_dist = 0.0
            count = 0
            
            for layer in ['fc1', 'fc2', 'fc3']:
                for matrix in ['lora_A', 'lora_B']:
                    lora_tensor = lora_params[layer][matrix]
                    ghost_tensor = ghost_params[layer][matrix].to(lora_tensor.device)
                    
                    dist = torch.norm(lora_tensor - ghost_tensor).item()
                    total_dist += dist
                    count += 1
            
            avg_dist = total_dist / count
            
            if avg_dist < min_distance:
                min_distance = avg_dist
                closest_ancestor = ghost_id
        
        return (closest_ancestor, min_distance) if closest_ancestor else None
    
    def prune_weak_ghosts(self, max_ghosts: int = 50):
        """
        Zayıf hayaletleri temizle (Hafıza için!)
        
        En güçlü max_ghosts kadar tut
        """
        if len(self.ghost_parameters) <= max_ghosts:
            return
        
        # Influence'a göre sırala
        sorted_ghosts = sorted(
            self.ghost_influence.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # İlk max_ghosts'u tut
        keep_ids = {ghost_id for ghost_id, _ in sorted_ghosts[:max_ghosts]}
        
        # Geri kalanları sil
        removed = 0
        for ghost_id in list(self.ghost_parameters.keys()):
            if ghost_id not in keep_ids:
                del self.ghost_parameters[ghost_id]
                del self.ghost_influence[ghost_id]
                removed += 1
        
        if removed > 0:
            print(f"   👻 {removed} zayıf hayalet temizlendi (RAM optimizasyonu)")

    def apply_temperament_perturbation(self, population: List, intensity: float = 0.1):
        """
        👻 HAYALET ETKİSİ (Resonance & Dissonance)
        
        Hayaletler rastgele saldırmaz!
        Kendi mizaçlarına uygun olanları "Gaza getirir" (Rezonans),
        Zıt olanları "Kafasını karıştırır" (Dissonance).
        """
        import random
        import numpy as np
        
        if not self.ghost_parameters:
            return
            
        # Rastgele bir hayalet seç (O maçın ruhu)
        ghost_id = random.choice(list(self.ghost_parameters.keys()))
        # Hayaletin mizacını bul (yoksa rastgele varsay)
        # (Basitlik için hayaletin mizacını parametrelerinden türetebiliriz veya rastgele atayabiliriz)
        ghost_temperament_vec = np.random.rand(5) # [Conf, Amb, Stress, Pat, Risk]
        
        affected_count = 0
        
        for lora in population:
            # LoRA mizaç vektörü
            temp = lora.temperament
            
            # 🛡️ GÜVENLİK KONTROLÜ: Temperament bozuksa düzelt!
            if not isinstance(temp, dict):
                print(f"⚠️ UYARI: {lora.name} mizaç verisi bozuk! (Tip: {type(temp)}) -> Onarılıyor...")
                # Varsayılan mizaç ata
                temp = {
                    'confidence_level': 0.5, 'ambition': 0.5, 'stress_tolerance': 0.5,
                    'patience': 0.5, 'risk_appetite': 0.5, 'social_intelligence': 0.5,
                    'independence': 0.5, 'contrarian_score': 0.5, 'herd_tendency': 0.5
                }
                lora.temperament = temp
            lora_vec = np.array([
                temp.get('confidence_level', 0.5),
                temp.get('ambition', 0.5),
                temp.get('stress_tolerance', 0.5),
                temp.get('patience', 0.5),
                temp.get('risk_appetite', 0.5)
            ])
            
            # 1. REZONANS HESAPLA (Cosine Similarity)
            # Dot product / magnitudes
            dot_product = np.dot(ghost_temperament_vec, lora_vec)
            norm_a = np.linalg.norm(ghost_temperament_vec)
            norm_b = np.linalg.norm(lora_vec)
            
            similarity = dot_product / (norm_a * norm_b + 1e-10) # -1 ile 1 arası (ama burada 0-1 çünkü değerler pozitif)
            
            # 2. ETKİ TÜRÜ
            if similarity > 0.8:
                # 🔥 REZONANS (Amplification)
                # Hayalet ve LoRA aynı kafada! Özellikleri uçur!
                # Agresifse daha agresif, sakinse daha sakin.
                direction = 1.0
                impact_type = "Rezonans"
            elif similarity < 0.4:
                # 🌪️ DISSONANCE (Conflict)
                # Hayalet ve LoRA zıt! Kafası karışır.
                # Özellikleri merkeze (0.5) çeker veya rastgele bozar.
                direction = -1.0 # Mevcut durumun tersine it
                impact_type = "Dissonance"
            else:
                continue # Nötr etki (Pas geç)
                
            # 3. UYGULA
            if random.random() < intensity:
                # Etkilenecek özelliği seç
                trait_idx = random.randint(0, 4)
                trait_keys = ['confidence_level', 'ambition', 'stress_tolerance', 'patience', 'risk_appetite']
                trait = trait_keys[trait_idx]
                
                # Değişim
                change = 0.15 * direction
                
                # Eğer Dissonance ise ve direction -1 ise, bu şu demek:
                # Yüksekse düşür, düşükse yükselt (Kaos/Denge)
                if impact_type == "Dissonance":
                    current = temp.get(trait, 0.5)
                    if current > 0.5: change = -0.15
                    else: change = 0.15
                
                # Kalıcı değişim (Maçlık delilik!)
                current_val = temp.get(trait, 0.5)
                temp[trait] = max(0.0, min(1.0, current_val + change))
                
                # Log (Opsiyonel)
                # print(f"👻 {lora.name[:10]} -> {impact_type} ({trait} {change:+.2f})")
                
                affected_count += 1
                
            lora.temperament = temp
        
        if affected_count > 0:
            print(f"   👻 Ghost Field dalgalanması: {affected_count} LoRA'nın mizacı değişti!")


# Global instance
ghost_fields = GhostFields(γ=0.1)

