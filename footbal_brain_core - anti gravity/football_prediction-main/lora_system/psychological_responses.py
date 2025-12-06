"""
🧠 PSİKOLOJİK TEPKİLER SİSTEMİ (TAM DİNAMİK!)
===============================================

KODLANMIŞ TEPKİ YOK!
Sadece mizaç değerleri → formül → tepki

Her LoRA kendi mizacına göre farklı yorumlar.
"""

import random
from typing import Dict, Tuple
import numpy as np


class PsychologicalResponseSystem:
    """
    Psikolojik tepki sistemi - Tamamen dinamik!
    """
    
    @staticmethod
    def react_to_rank_drop(lora, old_rank: int, new_rank: int, current_match: int) -> Dict:
        """
        Scoreboard'da düşüşe MİZAÇ BAZLI DİNAMİK TEPKİ!
        
        Kodlanmış tepki YOK! Sadece formül!
        
        Returns:
            {
                'trauma_gain': float,
                'motivation_boost': float,
                'death_resistance': float,
                'fitness_modifier': float,
                'reaction_intensity': float (0-1),
                'emotion_type': str ('anger', 'sadness', 'determination', 'calm')
            }
        """
        temp = lora.temperament
        rank_drop = new_rank - old_rank  # Pozitif = düştü
        
        if rank_drop <= 0:
            return {'trauma_gain': 0, 'motivation_boost': 0, 'death_resistance': 0, 
                    'fitness_modifier': 0, 'reaction_intensity': 0, 'emotion_type': 'neutral'}
        
        # ============================================
        # FORMÜL BAZLI HESAPLAMA
        # ============================================
        
        # TEMEL FAKTÖRLER
        ambition = temp.get('ambition', 0.5)
        anger = temp.get('anger_tendency', 0.5)
        resilience = temp.get('resilience', 0.5)
        stress_tolerance = temp.get('stress_tolerance', 0.5)
        emotional_depth = temp.get('emotional_depth', 0.5)
        competitiveness = temp.get('competitiveness', 0.5)
        will_to_live = temp.get('will_to_live', 0.5)
        
        # DÜŞÜŞ BÜYÜKLÜĞÜ ETKİSİ
        drop_factor = min(rank_drop / 20.0, 1.0)  # 0-1 arası normalize
        
        # TRAVMA HESAPLA
        # Duyarlı + Düşük dayanıklılık = Yüksek travma
        trauma_base = emotional_depth * (1.0 - resilience) * 0.7
        trauma_from_drop = drop_factor * 0.5
        trauma_total = (trauma_base + trauma_from_drop) * (1.0 + (1.0 - stress_tolerance) * 0.5)
        
        # MOTİVASYON HESAPLA
        # Hırslı + Rekabetçi = Yüksek motivasyon
        motivation_base = (ambition + competitiveness) / 2.0
        motivation_from_anger = anger * 0.5  # Sinir → enerji
        motivation_total = (motivation_base + motivation_from_anger) * drop_factor * 2.0
        
        # ÖLÜM DİRENCİ HESAPLA
        # Hırslı + Yaşam isteği + Sinirli = Ölmek istemiyor!
        death_resistance = (ambition * 0.4 + will_to_live * 0.4 + anger * 0.2) * drop_factor * 0.4
        
        # FITNESS MODİFİER
        # Motivasyon yüksekse pozitif, travma yüksekse negatif
        fitness_modifier = (motivation_total * 0.03) - (trauma_total * 0.02)
        
        # TEPKİ YOĞ UNLUĞU
        reaction_intensity = (ambition + anger + emotional_depth) / 3.0
        
        # DUYGU TİPİ BELİRLE (en baskın özellik)
        if anger > 0.7 and ambition > 0.6:
            emotion_type = 'fury'  # Öfke + Hırs
        elif ambition > 0.7:
            emotion_type = 'determination'  # Kararlılık
        elif emotional_depth > 0.7 and resilience < 0.4:
            emotion_type = 'despair'  # Umutsuzluk
        elif stress_tolerance > 0.7:
            emotion_type = 'calm'  # Sakin
        else:
            emotion_type = 'mixed'  # Karışık
        
        return {
            'trauma_gain': min(trauma_total, 2.0),
            'motivation_boost': min(motivation_total, 3.0),
            'death_resistance': death_resistance,
            'fitness_modifier': fitness_modifier,
            'reaction_intensity': reaction_intensity,
            'emotion_type': emotion_type,
            'rank_drop': rank_drop
        }
    
    @staticmethod
    def react_to_loss(lora, lost_lora_id: str, bond_strength: float, loss_type: str = "death") -> Dict:
        """
        Birini kaybetmeye MİZAÇ BAZLI DİNAMİK TEPKİ!
        
        Kodlanmış tepki YOK! Sadece formül!
        """
        temp = lora.temperament
        
        # TEMEL FAKTÖRLER
        emotional_depth = temp.get('emotional_depth', 0.5)
        empathy = temp.get('empathy', 0.5)
        ambition = temp.get('ambition', 0.5)
        resilience = temp.get('resilience', 0.5)
        social_intelligence = temp.get('social_intelligence', 0.5)
        stress_tolerance = temp.get('stress_tolerance', 0.5)
        
        # BAĞ GÜCÜ ETKİSİ
        bond_factor = bond_strength  # 0-1 arası
        
        # KAYIP TİPİ AĞIRLIĞI
        loss_weight = {
            'death': 1.0,      # En ağır
            'hibernation': 0.3,  # Hafif
            'distance': 0.5    # Orta
        }.get(loss_type, 0.5)
        
        # ============================================
        # FORMÜL BAZLI HESAPLAMA
        # ============================================
        
        # TRAVMA HESAPLA
        # Duygusal derinlik + Empati + Bağ = Travma
        trauma_sensitivity = (emotional_depth * 0.5 + empathy * 0.5)
        trauma_from_bond = bond_factor * loss_weight * 1.5
        trauma_reduction = resilience * 0.5  # Dayanıklılık azaltır
        trauma_total = (trauma_sensitivity * trauma_from_bond) - trauma_reduction
        trauma_total = max(0.0, trauma_total)  # Negatif olamaz
        
        # MOTİVASYON DEĞİŞİMİ
        # Hırslı → Tetiklenme (pozitif)
        # Duygusal → Çöküş (negatif)
        if ambition > 0.65:
            # Hırslı: Kaybı motivasyona çevirir!
            motivation_change = bond_factor * ambition * 1.5
        else:
            # Hırssız: Motivasyon düşer
            motivation_change = -bond_factor * emotional_depth * 0.5
        
        # FITNESS MODİFİER
        # Travma negatif, motivasyon pozitif
        fitness_modifier = (motivation_change * 0.04) - (trauma_total * 0.03)
        
        # SOSYAL ADAPTASYON (Sosyal zeki çabuk toparlanır)
        adaptation_speed = social_intelligence * 0.3
        
        # TEPKİ YOĞ UNLUĞU
        reaction_intensity = (emotional_depth + empathy + bond_factor) / 3.0
        
        # DUYGU TİPİ
        if ambition > 0.7 and bond_factor > 0.5:
            emotion_type = 'triggered_motivation'  # Tetiklenme
        elif emotional_depth > 0.7 and empathy > 0.7:
            emotion_type = 'deep_grief'  # Derin keder
        elif resilience > 0.7:
            emotion_type = 'acceptance'  # Kabul
        elif social_intelligence > 0.7:
            emotion_type = 'adaptive_sadness'  # Adapte oluyor
        else:
            emotion_type = 'neutral_loss'  # Nötr
        
        return {
            'trauma_gain': min(trauma_total, 2.5),
            'motivation_change': motivation_change,
            'fitness_modifier': fitness_modifier,
            'adaptation_speed': adaptation_speed,
            'reaction_intensity': reaction_intensity,
            'emotion_type': emotion_type,
            'bond_strength': bond_strength
        }
    
    @staticmethod
    def calculate_death_threshold_modifier(lora) -> float:
        """
        Mizaç bazlı ölüm eşiği modifikasyonu
        
        FORMÜL:
        Hırslı + Dayanıklı + Yaşam isteği + Sinirli = ZOR ÖLÜR!
        
        Returns:
            Modifier (-0.04 to +0.04)
        """
        temp = lora.temperament
        
        ambition = temp.get('ambition', 0.5)
        resilience = temp.get('resilience', 0.5)
        will_to_live = temp.get('will_to_live', 0.5)
        anger_tendency = temp.get('anger_tendency', 0.5)
        stress_tolerance = temp.get('stress_tolerance', 0.5)
        
        # HAYATTA KALMA SKORU
        # Hırslı + Dayanıklı + Yaşam isteği = Ölmez!
        survival_score = (
            ambition * 0.30 +
            resilience * 0.30 +
            will_to_live * 0.25 +
            stress_tolerance * 0.10 +
            anger_tendency * 0.05  # Sinir biraz yardımcı
        )
        
        # 0.5 = nötr
        # >0.5 = güçlü (threshold düşer, ölmesi zor!)
        # <0.5 = zayıf (threshold artar, ölmesi kolay!)
        modifier = (0.5 - survival_score) * 0.08  # -0.04 to +0.04
        
        return modifier
    
    @staticmethod
    def generate_reaction_text(lora, response_data: Dict, event_type: str) -> str:
        """
        Tepki metni oluştur (MİZAÇ BAZLI!)
        
        Args:
            lora: LoRA instance
            response_data: Tepki dictionary
            event_type: 'rank_drop', 'loss', vs.
        
        Returns:
            Tepki metni
        """
        emotion = response_data.get('emotion_type', 'neutral')
        intensity = response_data.get('reaction_intensity', 0.5)
        
        # RANK DROP TEPKİLERİ
        if event_type == 'rank_drop':
            rank_drop = response_data.get('rank_drop', 0)
            
            if emotion == 'fury':
                if intensity > 0.8:
                    return f"🔥🔥🔥 ÇILDIRDIM! {rank_drop} BASAMAK DÜŞTÜM! BU KABUL EDİLEMEZ! GÜCÜMÜ KANITLAYACAĞIM!"
                else:
                    return f"🔥 Sinirliyim! {rank_drop} basamak... Geri döneceğim!"
            
            elif emotion == 'determination':
                return f"💪 Kararlıyım. {rank_drop} basamak düşüş ama vazgeçmiyorum. Geri dönüş zamanı!"
            
            elif emotion == 'despair':
                return f"😢 {rank_drop} basamak... Moralim bozuk. Yapabilir miyim acaba?"
            
            elif emotion == 'calm':
                return f"🧘 {rank_drop} basamak düşüş. Olabilir, sakin kalıyorum."
            
            else:
                return f"⚖️ {rank_drop} basamak düştüm. Üzgünüm ama devam edeceğim."
        
        # KAYIP TEPKİLERİ
        elif event_type == 'loss':
            if emotion == 'triggered_motivation':
                return f"⚡ Onu kaybettim ama onun adına başarılı olacağım! Bu beni güçlendirdi!"
            
            elif emotion == 'deep_grief':
                return f"💔 Çok yakın birini kaybettim... İçim acıyor. Onun yokluğunu hep hissedeceğim."
            
            elif emotion == 'acceptance':
                return f"🛡️ Üzücü ama hayat devam ediyor. Güçlü kalacağım."
            
            elif emotion == 'adaptive_sadness':
                return f"😔 Kaybettim. Üzgünüm ama yeni bağlar kuracağım."
            
            else:
                return f"😐 Birini kaybettim. Üzücü."
        
        return "..."


# Global instance
psychological_responses = PsychologicalResponseSystem()
