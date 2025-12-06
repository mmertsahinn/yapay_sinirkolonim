"""
🌊 FLUID TEMPERAMENT (Akışkan Mizaç!)
=======================================

Mizaç sabit değil, sinüs dalgası gibi evrimleşir!

temperament(t) = base + amplitude × sin(frequency × t + phase)

OLAYLARA GÖRE DEĞİŞİR:
- Travma → Frekans düşer (yavaşlar)
- Başarı → Amplitude artar (daha dalgalı!)
- Sosyal kayıp → Base değişir (kalıcı!)
- Scoreboard yükseliş → Phase kayar!

Karakter gelişimi! Matematiksel!
"""

import numpy as np
from typing import Dict, List
from math import pi, sin


class FluidTemperament:
    """
    Akışkan mizaç sistemi (Sinüsoidal evolution!)
    """
    
    def __init__(self, σ: float = 0.03):
        # Her LoRA'nın sinüs parametreleri
        self.temperament_dynamics = {}  # lora_id -> dynamics
        
        # Gürültü şiddeti (Brownian motion!)
        self.σ = σ  # Stokastik terim!
        
        print(f"🌊 Fluid Temperament System başlatıldı (σ={σ} gürültü!)")
    
    def initialize_dynamics(self, lora):
        """
        LoRA'ya sinüsoidal dinamikler ver
        
        Her mizaç özelliği için:
        - base (Ortalama)
        - amplitude (Salınım genişliği)
        - frequency (Frekans)
        - phase (Faz)
        """
        dynamics = {}
        
        for trait, value in lora.temperament.items():
            dynamics[trait] = {
                'base': value,                    # Mevcut değer
                'amplitude': 0.10,                # Başlangıç: %10 salınım
                'frequency': 0.05,                # Yavaş dalga
                'phase': np.random.uniform(0, 2*pi)  # Rastgele faz
            }
        
        self.temperament_dynamics[lora.id] = dynamics
    
    def evolve_temperament(self, lora, match_count: int, events: List[str]) -> Dict:
        """
        Mizacı evrimleştir! (Olaylara göre!)
        
        Args:
            lora: LoRA instance
            match_count: Zaman (t)
            events: Bu maçtaki olaylar
                ['trauma', 'success_streak', 'social_loss', 'rank_rise', ...]
        
        Returns:
            Yeni mizaç değerleri
        """
        if lora.id not in self.temperament_dynamics:
            self.initialize_dynamics(lora)
        
        dynamics = self.temperament_dynamics[lora.id]
        new_temperament = {}
        
        for trait, params in dynamics.items():
            base = params['base']
            amplitude = params['amplitude']
            frequency = params['frequency']
            phase = params['phase']
            
            # ============================================
            # OLAYLARA GÖRE PARAMETRELERİ DEĞİŞTİR!
            # ============================================
            
            for event in events:
                # TRAVMA → Frekans düşer (Yavaşlar, donuklaşır!)
                if event == 'trauma':
                    frequency *= 0.90
                    amplitude *= 0.95  # Hafif azalır
                
                # BAŞARI → Amplitude artar (Daha canlı!)
                elif event == 'success_streak':
                    amplitude *= 1.15
                    if trait == 'confidence_level':
                        base += 0.02  # Özgüven kalıcı artar!
                
                # SOSYAL KAYIP → Base değişir (Kalıcı etki!)
                elif event == 'social_loss':
                    if trait == 'emotional_depth':
                        base += 0.03  # Daha duygusal!
                    if trait == 'resilience':
                        base -= 0.02  # Daha kırılgan!
                
                # RANK YÜKSELİŞİ → Faz kayması!
                elif event == 'rank_rise':
                    phase += pi/6  # 30 derece kayma!
                    if trait == 'ambition':
                        amplitude *= 1.10
                
                # RANK DÜŞÜŞÜ → Frekans ve base değişir!
                elif event == 'rank_drop':
                    if trait == 'anger_tendency':
                        base += 0.05  # Daha sinirli!
                    if trait == 'resilience':
                        # Hırslıysa direnir!
                        if lora.temperament.get('ambition', 0.5) > 0.7:
                            base += 0.03  # Daha dayanıklı!
                
                # FELAKET → Tüm parametreler değişir!
                elif event in ['disaster', 'kara_veba']:
                    frequency *= 0.70  # Çok yavaşlar
                    amplitude *= 0.80
                    if trait == 'stress_tolerance':
                        base -= 0.10  # Kalıcı stres!
            
            # ========================================
            # ORNSTEIN-UHLENBECK SÜRECİ!
            # dT = -θ(T - T_base) dt + σ dW
            # ========================================
            
            # Mevcut değer
            current_value = lora.temperament.get(trait, base)
            
            # 1) ORTALAMAYA DÖNÜŞ (Mean Reversion!)
            theta_return = 0.15  # Dönüş hızı (0.15 = orta hız)
            drift_term = -theta_return * (current_value - base)
            
            # 2) SİNÜSOİDAL MODÜLASobject
            # (Uzun vadeli salınım - deterministik!)
            t = match_count / 10.0
            sine_modulation = amplitude * sin(frequency * t + phase)
            
            # 3) BROWNIAN GÜRÜLTÜ (Wiener Process!)
            # Her maç rastgele dW ~ N(0, σ)
            brownian_noise = np.random.normal(0, self.σ)
            
            # 4) TOPLAM DİNAMİK (Ornstein-Uhlenbeck + Sinüs!)
            # dT = drift + sine + noise
            stochastic_value = current_value + drift_term + sine_modulation * 0.1 + brownian_noise
            
            # 0-1 arası sınırla
            stochastic_value = max(0.0, min(1.0, stochastic_value))
            
            new_temperament[trait] = stochastic_value
            
            # Parametreleri güncelle
            params['base'] = base
            params['amplitude'] = amplitude
            params['frequency'] = frequency
            params['phase'] = phase
        
        # Mizacı güncelle
        lora.temperament = new_temperament
        
        return new_temperament
    
    def get_temperament_trajectory(self, lora, trait: str, future_matches: int = 50) -> List[float]:
        """
        Gelecekteki mizaç yörüngesini tahmin et!
        
        Args:
            lora: LoRA
            trait: Hangi özellik? (örn: 'independence')
            future_matches: Kaç maç ileri?
        
        Returns:
            Gelecek değerler listesi
        """
        if lora.id not in self.temperament_dynamics:
            return []
        
        dynamics = self.temperament_dynamics[lora.id].get(trait, {})
        if not dynamics:
            return []
        
        base = dynamics['base']
        amplitude = dynamics['amplitude']
        frequency = dynamics['frequency']
        phase = dynamics['phase']
        
        trajectory = []
        current_match = len(lora.fitness_history)
        
        for i in range(future_matches):
            t = (current_match + i) / 10.0
            value = base + amplitude * sin(frequency * t + phase)
            value = max(0.0, min(1.0, value))
            trajectory.append(value)
        
        return trajectory


# Global instance
fluid_temperament = FluidTemperament()

