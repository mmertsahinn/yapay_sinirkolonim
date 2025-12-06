"""
⚠️ POPÜLASYON ALARM SİSTEMİ
===========================

LoRA'lar soyun azaldığını anlayıp üreme odaklı olur!

ALARM SEVİYELERİ:
- YEŞİL (> 30): Normal, iş odaklı
- SARI (20-30): Dikkat, üreme teşvik edilir
- KIRMIZı (10-20): Tehlike, üreme öncelik!
- ACİL (< 10): Kriz! Maksimum üreme çabası!

Alarm seviyesine göre:
- Üreme şansı artar
- LoRA'ların hedefleri değişir
- Sosyal bağlar güçlenir
"""

from typing import Dict


class PopulationAlarm:
    """
    Popülasyon alarm sistemi
    """
    
    def __init__(self):
        self.current_level = "GREEN"
        self.history = []
    
    def check_alarm_level(self, population_size: int) -> Dict:
        """
        Popülasyon boyutuna göre alarm seviyesi
        
        Returns:
            {
                'level': 'GREEN' / 'YELLOW' / 'RED' / 'CRITICAL',
                'message': '...',
                'reproduction_multiplier': 1.0 - 10.0,
                'social_focus': 0.2 - 0.8  (ne kadar sosyal odaklı?)
            }
        """
        
        if population_size >= 30:
            level = "GREEN"
            message = "Popülasyon sağlıklı, normal yaşam"
            repro_mult = 1.0  # Normal üreme
            social_focus = 0.2  # %20 sosyal
        
        elif population_size >= 20:
            level = "YELLOW"
            message = "⚠️ Soy azalıyor! Üreme teşvik edilir"
            repro_mult = 2.0  # 2x üreme şansı
            social_focus = 0.4  # %40 sosyal
        
        elif population_size >= 10:
            level = "RED"
            message = "🚨 SOY TEHLİKEDE! Üreme öncelik!"
            repro_mult = 5.0  # 5x üreme şansı
            social_focus = 0.6  # %60 sosyal
        
        else:
            level = "CRITICAL"
            message = "💀 ACİL DURUM! Tür yok oluyor!"
            repro_mult = 10.0  # 10x üreme şansı
            social_focus = 0.8  # %80 sosyal
        
        # Seviye değiştiyse kaydet
        if level != self.current_level:
            self.history.append({
                'old_level': self.current_level,
                'new_level': level,
                'population': population_size
            })
            self.current_level = level
        
        return {
            'level': level,
            'message': message,
            'reproduction_multiplier': repro_mult,
            'social_focus': social_focus,
            'population': population_size
        }


# Global instance
population_alarm = PopulationAlarm()



