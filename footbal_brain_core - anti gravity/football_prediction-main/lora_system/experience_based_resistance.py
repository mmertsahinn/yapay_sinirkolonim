"""
🛡️ DENEYİM BAZLI DİRENÇ SİSTEMİ
=================================

LoRA'lar deneyimlerinden ÖLÜM DİRENCİ kazanır!

DİRENÇ KAYNAKLARI:
1. Scoreboard düşüşünden kurtulma
2. Doğa felaketinden kurtulma
3. Travmalardan toparlanma
4. Kritik durumlardan dönüş

Her deneyim → Direnç artar!
"""

from typing import Dict
import numpy as np


class ExperienceBasedResistance:
    """
    Deneyim bazlı ölüm direnci sistemi
    """
    
    def __init__(self):
        # Her LoRA'nın direnç değerleri
        self.lora_resistances = {}  # lora_id -> resistance_dict
    
    def get_resistance(self, lora_id: str) -> Dict:
        """
        LoRA'nın direnç değerlerini al
        
        Returns:
            {
                'total_resistance': 0-1 arası,
                'rank_drop_resistance': 0-1,
                'disaster_resistance': 0-1,
                'trauma_resistance': 0-1,
                'critical_survival_bonus': 0-1
            }
        """
        if lora_id not in self.lora_resistances:
            # Yeni LoRA, direnç yok
            return {
                'total_resistance': 0.0,
                'rank_drop_resistance': 0.0,
                'disaster_resistance': 0.0,
                'trauma_resistance': 0.0,
                'critical_survival_bonus': 0.0
            }
        
        return self.lora_resistances[lora_id]
    
    def add_rank_drop_survival(self, lora_id: str, rank_drop: int, survived_how: str = "determination"):
        """
        Scoreboard'dan düşüp hayatta kaldı!
        
        Args:
            rank_drop: Kaç basamak düştü
            survived_how: 'determination', 'motivation', 'luck'
        """
        if lora_id not in self.lora_resistances:
            self.lora_resistances[lora_id] = {
                'total_resistance': 0.0,
                'rank_drop_resistance': 0.0,
                'disaster_resistance': 0.0,
                'trauma_resistance': 0.0,
                'critical_survival_bonus': 0.0
            }
        
        resistance = self.lora_resistances[lora_id]
        
        # Düşüş büyüklüğüne göre direnç
        drop_factor = min(rank_drop / 20.0, 1.0)
        
        if survived_how == "determination":
            gain = 0.08 * drop_factor  # Kararlılıkla → en yüksek
        elif survived_how == "motivation":
            gain = 0.06 * drop_factor  # Motivasyonla → orta
        else:  # luck
            gain = 0.03 * drop_factor  # Şansla → düşük
        
        resistance['rank_drop_resistance'] += gain
        resistance['rank_drop_resistance'] = min(0.60, resistance['rank_drop_resistance'])  # Max 0.60
        
        # Toplam güncelle
        self._update_total_resistance(lora_id)
    
    def add_disaster_survival(self, lora_id: str, disaster_type: str, survived_how: str = "luck"):
        """
        Doğa felaketinden kurtuldu!
        
        Args:
            disaster_type: 'deprem', 'veba', vs.
            survived_how: 'armor', 'adaptation', 'luck'
        """
        if lora_id not in self.lora_resistances:
            self.lora_resistances[lora_id] = {
                'total_resistance': 0.0,
                'rank_drop_resistance': 0.0,
                'disaster_resistance': 0.0,
                'trauma_resistance': 0.0,
                'critical_survival_bonus': 0.0
            }
        
        resistance = self.lora_resistances[lora_id]
        
        # Felaket tipine göre
        disaster_severity = {
            'minor_shake': 0.05,
            'stress_wave': 0.08,
            'quake': 0.10,
            'major_quake': 0.15,
            'mass_extinction': 0.20,
            'kara_veba': 0.30  # En yüksek!
        }.get(disaster_type, 0.10)
        
        # Nasıl kurtulduğuna göre
        if survived_how == "armor":
            multiplier = 1.5  # Zırh ile → yüksek
        elif survived_how == "adaptation":
            multiplier = 2.0  # Adaptasyon → en yüksek!
        else:  # luck
            multiplier = 0.8  # Şans → düşük
        
        gain = disaster_severity * multiplier
        
        resistance['disaster_resistance'] += gain
        resistance['disaster_resistance'] = min(0.70, resistance['disaster_resistance'])  # Max 0.70
        
        # Toplam güncelle
        self._update_total_resistance(lora_id)
    
    def add_trauma_recovery(self, lora_id: str, trauma_count: int):
        """
        Travmalardan toparlandı!
        
        Her travmadan toparlanma → Direnç!
        """
        if lora_id not in self.lora_resistances:
            self.lora_resistances[lora_id] = {
                'total_resistance': 0.0,
                'rank_drop_resistance': 0.0,
                'disaster_resistance': 0.0,
                'trauma_resistance': 0.0,
                'critical_survival_bonus': 0.0
            }
        
        resistance = self.lora_resistances[lora_id]
        
        # Çok travma → Çok direnç (antifrajilite!)
        gain = min(trauma_count * 0.02, 0.30)  # Max 0.30
        
        resistance['trauma_resistance'] += gain
        resistance['trauma_resistance'] = min(0.50, resistance['trauma_resistance'])
        
        # Toplam güncelle
        self._update_total_resistance(lora_id)
    
    def add_critical_survival(self, lora_id: str, fitness_at_survival: float):
        """
        Kritik durumdan kurtuldu! (fitness çok düşükken hayatta kaldı!)
        
        Args:
            fitness_at_survival: Kurtulduğunda fitness ne kadar düşüktü?
        """
        if lora_id not in self.lora_resistances:
            self.lora_resistances[lora_id] = {
                'total_resistance': 0.0,
                'rank_drop_resistance': 0.0,
                'disaster_resistance': 0.0,
                'trauma_resistance': 0.0,
                'critical_survival_bonus': 0.0
            }
        
        resistance = self.lora_resistances[lora_id]
        
        # Ne kadar kritikti? (düşük fitness = kritik!)
        criticality = max(0.0, 0.10 - fitness_at_survival)  # 0.10 altı = kritik
        
        gain = criticality * 2.0  # 0.10'da: 0.00, 0.01'de: 0.18
        
        resistance['critical_survival_bonus'] += gain
        resistance['critical_survival_bonus'] = min(0.40, resistance['critical_survival_bonus'])
        
        # Toplam güncelle
        self._update_total_resistance(lora_id)
    
    def _update_total_resistance(self, lora_id: str):
        """Toplam direnci güncelle"""
        resistance = self.lora_resistances[lora_id]
        
        # Toplam = Ağırlıklı ortalama
        total = (
            resistance['rank_drop_resistance'] * 0.25 +
            resistance['disaster_resistance'] * 0.35 +
            resistance['trauma_resistance'] * 0.20 +
            resistance['critical_survival_bonus'] * 0.20
        )
        
        resistance['total_resistance'] = min(0.80, total)  # Max 0.80
    
    def calculate_death_threshold(self, lora, base_threshold: float = 0.05) -> float:
        """
        AKIŞKAN ÖLÜM EŞİĞİ!
        
        3 faktör:
        1. Mizaç (Hırs, dayanıklılık, yaşam isteği)
        2. Deneyim direnci (Bu metod!)
        3. Psikolojik durum (Motivasyon, travma)
        
        Returns:
            Dinamik threshold (0.01 - 0.12 arası)
        """
        # 1) MİZAÇ FAKTÖRÜ
        from lora_system.psychological_responses import psychological_responses
        temperament_modifier = psychological_responses.calculate_death_threshold_modifier(lora)
        
        # 2) DENEYİM DİRENCİ
        resistance_data = self.get_resistance(lora.id)
        total_resistance = resistance_data['total_resistance']
        
        # Direnç yüksek → threshold düşer (ölmesi zor!)
        resistance_modifier = -total_resistance * 0.05  # Max -0.04
        
        # 3) PSİKOLOJİK DURUM
        # Yüksek motivasyon → threshold düşer
        # Yüksek travma → threshold artar
        motivation_level = getattr(lora, '_current_motivation', 0.0)
        trauma_level = len(getattr(lora, 'trauma_history', [])) / 20.0  # 0-1 normalize
        
        psychological_modifier = (motivation_level * -0.02) + (trauma_level * 0.02)
        
        # TOPLAM THRESHOLD
        final_threshold = base_threshold + temperament_modifier + resistance_modifier + psychological_modifier
        
        # 0.01 - 0.12 arası sınırla
        final_threshold = max(0.01, min(0.12, final_threshold))
        
        return final_threshold


# Global instance
experience_resistance = ExperienceBasedResistance()



