"""
🔍 DİNAMİK UZMANLIK KEŞFİ
==========================

Kodlanmış pattern YOK!
LoRA kendi pattern'lerini keşfeder!

NASIL ÇALIŞIR:
1. Her maçta feature kombinasyonlarını analiz eder
2. Hangi kombinasyonlarda başarılı olduğunu öğrenir
3. Kendi uzmanlık alanını kendisi tanımlar!

ÖRNEK:
  "home_form:yüksek + odds:düşük + hype:orta" → %85 başarı
  → Bu benim uzmanlığım!
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class DynamicSpecialization:
    """
    Dinamik uzmanlık keşif sistemi
    """
    
    def __init__(self):
        # Her LoRA'nın keşfettiği pattern'ler
        self.discovered_patterns = {}  # lora_id -> patterns
    
    def analyze_match_features(self, match_data) -> Dict:
        """
        Maç özelliklerini analiz et (Feature kombinasyonları!)
        
        Kodlanmış pattern YOK!
        Sistematik feature kombinasyonları çıkar!
        
        Returns:
            Feature kombinasyonları
        """
        if not isinstance(match_data, dict):
            # pandas Series ise dict'e çevir
            match_data = match_data.to_dict()
        
        combinations = {}
        
        # TEMEL FEATURES
        home_form = match_data.get('home_form', 0.5)
        away_form = match_data.get('away_form', 0.5)
        home_odds = match_data.get('home_odds', 2.0)
        away_odds = match_data.get('away_odds', 2.0)
        hype = match_data.get('total_tweets', 0)
        
        # KATEGORIZEL DÖNÜŞÜM (yüksek/orta/düşük)
        def categorize(value, thresholds):
            if value < thresholds[0]:
                return 'düşük'
            elif value < thresholds[1]:
                return 'orta'
            else:
                return 'yüksek'
        
        # Form kategorileri
        home_form_cat = categorize(home_form, [0.40, 0.65])
        away_form_cat = categorize(away_form, [0.40, 0.65])
        
        # Odds kategorileri
        home_odds_cat = categorize(home_odds, [1.8, 2.5])
        away_odds_cat = categorize(away_odds, [1.8, 2.5])
        
        # Hype kategorileri
        hype_cat = categorize(hype, [10000, 50000])
        
        # KOMBİNASYONLAR OLUŞTUR
        # 2'li kombinasyonlar
        combinations['home_form + hype'] = f"{home_form_cat}_{hype_cat}"
        combinations['away_form + hype'] = f"{away_form_cat}_{hype_cat}"
        combinations['home_odds + hype'] = f"{home_odds_cat}_{hype_cat}"
        combinations['form_dengesi'] = f"{home_form_cat}_vs_{away_form_cat}"
        combinations['odds_dengesi'] = f"{home_odds_cat}_vs_{away_odds_cat}"
        
        # 3'lü kombinasyonlar
        combinations['home_form + odds + hype'] = f"{home_form_cat}_{home_odds_cat}_{hype_cat}"
        combinations['away_form + odds + hype'] = f"{away_form_cat}_{away_odds_cat}_{hype_cat}"
        
        return combinations
    
    def update_lora_pattern_discovery(self, lora, match_features: Dict, correct: bool):
        """
        LoRA'nın pattern keşfini güncelle!
        
        LoRA hangi feature kombinasyonunda başarılı?
        """
        if lora.id not in self.discovered_patterns:
            self.discovered_patterns[lora.id] = {}
        
        patterns = self.discovered_patterns[lora.id]
        
        # Her kombinasyonu kaydet
        for combo_name, combo_value in match_features.items():
            if combo_value not in patterns:
                patterns[combo_value] = {
                    'total': 0,
                    'correct': 0,
                    'success_rate': 0.0,
                    'combo_type': combo_name
                }
            
            patterns[combo_value]['total'] += 1
            if correct:
                patterns[combo_value]['correct'] += 1
            
            # Başarı oranını güncelle
            total = patterns[combo_value]['total']
            correct_count = patterns[combo_value]['correct']
            patterns[combo_value]['success_rate'] = correct_count / total
    
    def detect_specialization(self, lora, min_samples: int = 20) -> Optional[str]:
        """
        LoRA'nın uzmanlığını tespit et (DİNAMİK!)
        
        En yüksek başarı oranına sahip pattern'i bul!
        
        Returns:
            Specialization metni (örn: "yüksek_orta_yüksek uzmanı")
        """
        if lora.id not in self.discovered_patterns:
            return None
        
        patterns = self.discovered_patterns[lora.id]
        
        # En başarılı pattern'i bul
        best_pattern = None
        best_success = 0.0
        
        for pattern_value, stats in patterns.items():
            if stats['total'] >= min_samples:  # Yeterli veri var mı?
                if stats['success_rate'] > best_success:
                    best_success = stats['success_rate']
                    best_pattern = pattern_value
        
        # %70+ başarı varsa uzman!
        if best_success >= 0.70 and best_pattern:
            combo_type = patterns[best_pattern]['combo_type']
            return f"{combo_type}: {best_pattern} ({best_success:.0%})"
        
        return None


# Global instance
dynamic_specialization = DynamicSpecialization()

