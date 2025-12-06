"""
🌊 MASTER FLUX EQUATION (Ana Yaşam Akışı!)
============================================

Termodinamik Evrimsel Eylem (TES):

S_i(t) = ∫ (Darwin + Einstein_flux - Newton_cost) dτ

BİLEŞENLER:
1. DARWIN: Price katkısı (Popülasyona katkı)
2. EINSTEIN: KL sürpriz akışı (Haklı farklılık)
3. NEWTON: Onsager-Machlup maliyeti (İstikrar)

Fizik motoru! If/else YOK!
"""

import torch
import numpy as np
from typing import Dict, List
import torch.nn.functional as F


class MasterFluxEquation:
    """
    Termodinamik Evrimsel Skor sistemi
    """
    
    def __init__(self):
        # Lambda parametreleri
        self.λ_einstein = 1.0
        self.λ_newton = 0.5
        self.λ_social = 0.1  # Sosyal bağ bonusu
        self.λ_trauma = 0.2  # Travma cezası
        
        print("🌊 Master Flux Equation başlatıldı")
    
    def calculate_darwin_term(self, lora, population: List) -> float:
        """
        DARWIN TERİMİ: Price Denklemi
        
        D_i = Cov(w, z) / Var(z)
        
        w = Fitness (Başarı)
        z = Character (Mizaç vektörü - 15 boyut!)
        
        Mantık: LoRA'nın başarısı popülasyonun karakteri ile uyumlu mu?
        """
        if len(population) < 3:
            return 0.0  # Yetersiz veri
        
        # w (Fitness vector - tüm popülasyon)
        w_values = []
        z_vectors = []
        
        for other_lora in population:
            w_values.append(other_lora.get_recent_fitness())
            
            # z (Character): Mizaç vektörü (15 boyut)
            temp = other_lora.temperament
            z_vec = [
                temp.get('independence', 0.5),
                temp.get('social_intelligence', 0.5),
                temp.get('herd_tendency', 0.5),
                temp.get('contrarian_score', 0.5),
                temp.get('emotional_depth', 0.5),
                temp.get('empathy', 0.5),
                temp.get('anger_tendency', 0.5),
                temp.get('ambition', 0.5),
                temp.get('competitiveness', 0.5),
                temp.get('resilience', 0.5),
                temp.get('will_to_live', 0.5),
                temp.get('patience', 0.5),
                temp.get('impulsiveness', 0.5),
                temp.get('stress_tolerance', 0.5),
                temp.get('risk_appetite', 0.5)
            ]
            z_vectors.append(z_vec)
        
        w_values = np.array(w_values)
        z_vectors = np.array(z_vectors)  # (N, 15)
        
        # Bu LoRA'nın değerleri
        lora_w = lora.get_recent_fitness()
        lora_z = np.array([
            lora.temperament.get('independence', 0.5),
            lora.temperament.get('social_intelligence', 0.5),
            lora.temperament.get('herd_tendency', 0.5),
            lora.temperament.get('contrarian_score', 0.5),
            lora.temperament.get('emotional_depth', 0.5),
            lora.temperament.get('empathy', 0.5),
            lora.temperament.get('anger_tendency', 0.5),
            lora.temperament.get('ambition', 0.5),
            lora.temperament.get('competitiveness', 0.5),
            lora.temperament.get('resilience', 0.5),
            lora.temperament.get('will_to_live', 0.5),
            lora.temperament.get('patience', 0.5),
            lora.temperament.get('impulsiveness', 0.5),
            lora.temperament.get('stress_tolerance', 0.5),
            lora.temperament.get('risk_appetite', 0.5)
        ])
        
        # KOVARYANS hesapla (Her boyut için)
        covariances = []
        for dim in range(15):
            z_dim = z_vectors[:, dim]
            
            # Cov(w, z_dim)
            cov = np.cov(w_values, z_dim)[0, 1]
            var_z = np.var(z_dim)
            
            if var_z > 0.001:  # Sıfıra bölme kontrolü
                contribution = cov / var_z
                covariances.append(contribution)
        
        # Ortalama katkı
        darwin_score = np.mean(covariances) if covariances else 0.0
        
        return darwin_score
    
    def calculate_einstein_term(self, lora, lora_proba: np.ndarray, 
                                population_proba: np.ndarray, correct: bool) -> float:
        """
        EINSTEIN TERİMİ: KL-Divergence Surprisal
        
        E_i = D_KL(P_i || P_pop) × I_success
        
        P_i: LoRA'nın tahmini
        P_pop: Popülasyon ortalaması
        I_success: Doğru mu? (0/1)
        
        Mantık: Herkes yanılırken o bildi mi?
        """
        # KL-Divergence hesapla
        # D_KL(P || Q) = Σ P(i) log(P(i) / Q(i))
        
        # Güvenlik: 0'a bölme önle
        lora_proba = np.clip(lora_proba, 1e-10, 1.0)
        population_proba = np.clip(population_proba, 1e-10, 1.0)
        
        # Normalize (güvenlik)
        lora_proba = lora_proba / lora_proba.sum()
        population_proba = population_proba / population_proba.sum()
        
        # KL-Divergence
        kl_div = np.sum(lora_proba * np.log(lora_proba / population_proba))
        
        # Başarılıysa puan al!
        if correct:
            einstein_score = kl_div
        else:
            einstein_score = 0.0  # Sadece aykırı olmak yetmez, haklı olmak gerekir!
        
        # Kaydet (flux için)
        if not hasattr(lora, '_last_kl'):
            lora._last_kl = 0.0
        lora._last_kl = kl_div
        
        return einstein_score
    
    def calculate_newton_term(self, lora, fisher_data: Dict = None) -> float:
        """
        NEWTON TERİMİ: Onsager-Machlup (K-FAC ile!)
        
        N_i = exp(-S_OM)
        
        S_OM ≈ Flat Minima Score (K-FAC'dan!)
        
        Düşük action = İstikrarlı = Yüksek puan!
        """
        if fisher_data is None:
            from lora_system.kfac_fisher import kfac_fisher
            fisher_data = kfac_fisher.compute_fisher_kfac(lora)
        
        # Flat minima skoru (0-1)
        flat_score = fisher_data['flat_minima_score']
        
        # Action (Enerji maliyeti)
        # Düz minimumda = Düşük action!
        action_cost = 1.0 - flat_score
        
        # Consistency penalty (Ek!)
        if len(lora.fitness_history) >= 10:
            variance = np.var(lora.fitness_history[-10:])
            consistency_penalty = variance * 2.0  # Yüksek variance = yüksek maliyet
            action_cost += consistency_penalty
        
        # Newton puanı (Düşük maliyet = Yüksek puan!)
        newton_score = np.exp(-action_cost)
        
        return newton_score
    
    def update_life_energy(self, lora, population: List, lora_proba: np.ndarray,
                          population_proba: np.ndarray, correct: bool, 
                          fisher_data: Dict = None, dt: float = 1.0,
                          top_5_cache: Dict = None) -> Dict:
        """
        MASTER FLUX ile yaşam enerjisini güncelle!
        
        S_i(t) = ∫ (Darwin + λ₁×Einstein_flux - λ₂×Newton_cost) dτ
        
        🛡️ ÖLÜMSÜZLÜK KORUMASI: Çoklu uzman LoRA'lar daha az enerji kaybeder!
        
        Args:
            top_5_cache: Takım uzmanlık Top 5 listeleri (ölümsüzlük için!)
        
        Returns:
            Enerji durumu
        """
        # 1) DARWIN
        darwin = self.calculate_darwin_term(lora, population)
        
        # 2) EINSTEIN
        einstein = self.calculate_einstein_term(lora, lora_proba, population_proba, correct)
        
        # 3) NEWTON
        newton = self.calculate_newton_term(lora, fisher_data)
        
        # 4) ENERJI DEĞİŞİMİ
        dE = (darwin + self.λ_einstein * einstein) - (self.λ_newton * (1.0 - newton))
        
        # Sosyal ve travma bonusları
        if hasattr(lora, 'social_bonds') and len(lora.social_bonds) > 0:
            social_bonus = max(lora.social_bonds.values()) * self.λ_social
            dE += social_bonus
        
        if hasattr(lora, 'trauma_history'):
            # TraumaEvent objesi, dict değil!
            trauma_count = len([t for t in lora.trauma_history[-10:] if getattr(t, 'severity', 0) > 0.3])
            trauma_penalty = trauma_count * 0.05 * self.λ_trauma
            dE -= trauma_penalty
        
        # 🛡️ ÖLÜMSÜZLÜK KORUMASI! (Çoklu uzman LoRA'lar korunur!)
        if dE < 0 and top_5_cache is not None:  # Sadece enerji kaybında
            from lora_system.death_immunity_system import apply_death_immunity_to_energy_loss
            dE = apply_death_immunity_to_energy_loss(lora, dE, top_5_cache)
        
        # GÜNCELLE
        dE = dE * dt
        
        current_energy = getattr(lora, 'life_energy', 1.0)
        new_energy = current_energy + dE
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
            'dE': dE,
            'energy': new_energy,
            'status': status,
            'darwin': darwin,
            'einstein': einstein,
            'newton': newton
        }


# Global instance
master_flux = MasterFluxEquation()


