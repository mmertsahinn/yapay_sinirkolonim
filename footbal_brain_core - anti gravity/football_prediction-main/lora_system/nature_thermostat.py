"""
🌡️ NATURE'S THERMOSTAT (Doğanın Termostatı!)
==============================================

Doğa "if event == deprem" demez!
Doğa serbest enerjiyi minimize eder!

dT_nature/dt = α × (Hedef_Entropi - Mevcut_Entropi)

LoRA'lar çok başarılı → Entropi düşer → Doğa ısınır!
Doğa ısınınca → Zorluk artar!

Fizik bazlı! Otomatik!
"""

import numpy as np
from typing import List, Dict


class NatureThermostat:
    """
    Doğanın termostat sistemi (Entropy-based!)
    """
    
    def __init__(self, target_entropy: float = 0.80, α: float = 0.1):
        self.target_entropy = target_entropy  # Hedef entropi
        self.α = α  # Değişim hızı
        
        self.temperature = 0.5  # Başlangıç sıcaklığı (0-1)
        self.entropy_history = []
        
        print(f"🌡️ Nature's Thermostat başlatıldı (Hedef entropi: {target_entropy})")
    
    def calculate_population_entropy(self, lora_predictions: List[np.ndarray]) -> float:
        """
        Popülasyon entropisini hesapla
        
        Entropi = -Σ P(i) log P(i)
        
        Düşük entropi = Herkes aynı şeyi söylüyor! (Tehlike!)
        Yüksek entropi = Çeşitlilik var! (Sağlıklı!)
        
        Args:
            lora_predictions: Her LoRA'nın tahmin dağılımı
        
        Returns:
            Entropi (0-1 normalize)
        """
        if len(lora_predictions) == 0:
            return 0.5
        
        # Popülasyon ortalaması
        pop_avg = np.mean(lora_predictions, axis=0)
        
        # Normalize
        pop_avg = np.clip(pop_avg, 1e-10, 1.0)
        pop_avg = pop_avg / pop_avg.sum()
        
        # Shannon entropisi
        entropy = -np.sum(pop_avg * np.log(pop_avg + 1e-10))
        
        # Normalize (log(3) = max entropy for 3 classes)
        normalized_entropy = entropy / np.log(3)
        
        return normalized_entropy
    
    def update_temperature(self, population_entropy: float, dt: float = 1.0) -> Dict:
        """
        Doğanın sıcaklığını güncelle!
        
        dT/dt = α × (Mevcut - Hedef)  ← DÜZELTİLDİ!
        
        Düşük entropi → Gap negatif → Sıcaklık ARTAR! (Doğa zorlaşır!)
        Yüksek entropi → Gap pozitif → Sıcaklık DÜŞER! (Doğa yumuşar!)
        
        Args:
            population_entropy: Mevcut popülasyon entropisi (0-1)
            dt: Zaman adımı
        
        Returns:
            Sıcaklık bilgisi
        """
        # Entropi farkı (DÜZELTİLDİ!)
        # Düşük entropi (0.30) → Gap: -0.50 (Negatif!)
        # Yüksek entropi (0.90) → Gap: +0.10 (Pozitif!)
        entropy_gap = population_entropy - self.target_entropy  # Ters çevrildi!
        
        # Sıcaklık değişimi
        # Gap negatif → dT negatif → Sıcaklık ARTAR! (Çünkü çıkarıyoruz!)
        # DOĞRU FORMÜL: dT/dt = -α × gap (Eksi işareti!)
        dT = -self.α * entropy_gap * dt
        
        self.temperature += dT
        
        # 0-1 arası sınırla
        self.temperature = max(0.0, min(1.0, self.temperature))
        
        # Geçmişe ekle
        self.entropy_history.append({
            'entropy': population_entropy,
            'temperature': self.temperature,
            'gap': entropy_gap
        })
        
        # YORUM (EĞİTİCİ LOGLAR!)
        if self.temperature > 0.75:
            status = "🔥 SICAK! (Doğa Agresifleşti!)"
            # explanation = "LoRA'lar çok başarılı/benzer. Doğa dengeyi sağlamak için zorluğu artırıyor."
        elif self.temperature > 0.50:
            status = "☀️ Ilık (Normal Dengeli)"
            # explanation = "Sistem dengede. Standart zorluk seviyesi."
        elif self.temperature > 0.25:
            status = "☁️ Serin (Doğa Yumuşak)"
            # explanation = "LoRA'lar biraz zorlanıyor. Doğa baskıyı azalttı."
        else:
            status = "❄️ SOĞUK! (Doğa Pasif)"
            # explanation = "LoRA'lar başarısız veya çeşitlilik çok yüksek. Doğa iyileşmeye izin veriyor."
        
        return {
            'temperature': self.temperature,
            'entropy': population_entropy,
            'gap': entropy_gap,
            'dT': dT,
            'status': status
        }
    
    def get_difficulty_multiplier(self) -> float:
        """
        Sıcaklığa göre zorluk çarpanı
        
        Sıcak → Zorlaşır!
        Soğuk → Kolaylaşır!
        
        Returns:
            Çarpan (0.5 - 2.0)
        """
        # Sıcaklık 0.5: Normal (×1.0)
        # Sıcaklık 1.0: Çok sıcak (×2.0)
        # Sıcaklık 0.0: Çok soğuk (×0.5)
        
        multiplier = 0.5 + (self.temperature * 1.5)
        
        return multiplier
    
    def apply_temperature_effects(self, nature_state) -> Dict:
        """
        Sıcaklık etkilerini doğa durumuna uygula!
        
        Sıcak → Öfke artar, Sağlık azalır!
        """
        difficulty = self.get_difficulty_multiplier()
        
        # Sıcaklık etkileri
        if self.temperature > 0.70:
            # SICAK! Doğa agresif!
            anger_boost = (self.temperature - 0.70) * 0.5
            health_penalty = (self.temperature - 0.70) * 0.3
            
            nature_state.anger = min(1.0, nature_state.anger + anger_boost)
            nature_state.health = max(0.0, nature_state.health - health_penalty)
        
        elif self.temperature < 0.30:
            # SOĞUK! Doğa yumuşak!
            anger_reduction = (0.30 - self.temperature) * 0.3
            health_boost = (0.30 - self.temperature) * 0.2
            
            nature_state.anger = max(0.0, nature_state.anger - anger_reduction)
            nature_state.health = min(1.0, nature_state.health + health_boost)
        
        return {
            'difficulty_multiplier': difficulty,
            'temperature_effect': 'Aggressive' if self.temperature > 0.70 else 'Passive' if self.temperature < 0.30 else 'Neutral'
        }


# Global instance
nature_thermostat = NatureThermostat(target_entropy=0.80, α=0.1)

