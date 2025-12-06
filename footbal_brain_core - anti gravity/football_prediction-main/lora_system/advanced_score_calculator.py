"""
🎓 GELİŞMİŞ SCOREBOARD FORMÜLÜ (İleri Düzey Matematik)
========================================================

Adil sistem:
- Yeni uzmanlar eskileri geçebilir!
- Trend önemli (yükseliyor mu?)
- İstikrar ödüllendirilir
- Yaşa göre normalize (genç yetenekler avantajlı!)
"""

import numpy as np
from typing import List


class AdvancedScoreCalculator:
    """
    İleri düzey scoreboard hesaplaması
    """
    
    @staticmethod
    def calculate_advanced_score(lora, match_count: int = None) -> float:
        """
        Gelişmiş skor hesapla
        
        FORMÜL:
        ADVANCED_SCORE = 
          (Weighted_Recent × 0.30) +      # Son performans (ağırlıklı)
          (Age_Normalized × 0.25) +       # Yaşa göre normalize başarı
          (Peak_Performance × 0.20) +     # En iyi dönem
          (Momentum × 0.15) +             # Trend (yükseliyor mu?)
          (Consistency × 0.10)            # İstikrar
        
        Returns:
            0-1 arası advanced score
        """
        # 🆕 ÇÖMEZLİK CEZASI! (Minimum 20 maç!)
        if len(lora.fitness_history) < 20:
            # Çok genç, henüz kanıtlanmadı!
            # Yaşa göre ceza: 5 maç = 0.25x, 10 maç = 0.50x, 20 maç = 1.0x
            rookie_penalty = len(lora.fitness_history) / 20.0
            return lora.get_recent_fitness() * rookie_penalty * 0.5  # Ağır ceza!
        
        # 1) WEIGHTED RECENT (Son performans - ağırlıklı!) - 30%
        weighted_recent = AdvancedScoreCalculator._calculate_weighted_recent(lora)
        weighted_score = weighted_recent * 0.30
        
        # 2) AGE-NORMALIZED SUCCESS (Yaşa göre normalize) - 25%
        age_normalized = AdvancedScoreCalculator._calculate_age_normalized(lora, match_count)
        age_score = age_normalized * 0.25
        
        # 3) PEAK PERFORMANCE (En iyi dönem) - 20%
        peak = AdvancedScoreCalculator._calculate_peak_performance(lora)
        peak_score = peak * 0.20
        
        # 4) MOMENTUM (Trend - yükseliyor mu?) - 15%
        momentum = AdvancedScoreCalculator._calculate_momentum(lora)
        momentum_score = momentum * 0.15
        
        # 5) CONSISTENCY (İstikrar - variance düşük mü?) - 10%
        consistency = AdvancedScoreCalculator._calculate_consistency(lora)
        consistency_score = consistency * 0.10
        
        # 6) 🌟 MIRACLE PROTECTION (Mucize Koruması)
        # Yüksek Lazarus Lambda değerine sahip olanlar (Potansiyelli!)
        # ek puan alır. Bu sayede "Uyuyan Devler" silinmez!
        miracle_bonus = 0.0
        if hasattr(lora, '_lazarus_lambda'):
            l_lambda = getattr(lora, '_lazarus_lambda', 0.5)
            if l_lambda > 0.70:
                # Yüksek potansiyel!
                miracle_bonus = (l_lambda - 0.70) * 0.5  # Max 0.15 bonus
        
        # TOPLAM
        total = weighted_score + age_score + peak_score + momentum_score + consistency_score + miracle_bonus
        
        return total
    
    @staticmethod
    def _calculate_weighted_recent(lora) -> float:
        """
        Son performans (exponential weighted average)
        
        Son maçlar daha önemli!
        Maç 1: ağırlık 0.5
        Maç 10: ağırlık 1.0
        Maç 50: ağırlık 2.0
        """
        history = lora.fitness_history[-50:]  # Son 50 maç
        
        if len(history) == 0:
            return 0.0
        
        # Exponential ağırlıklar
        weights = []
        for i in range(len(history)):
            # Son maç: en yüksek ağırlık
            # İlk maç: en düşük ağırlık
            weight = np.exp(i / len(history))  # Exponential artış
            weights.append(weight)
        
        # Normalize
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        weighted_avg = np.dot(history, weights)
        
        return weighted_avg
    
    @staticmethod
    def _calculate_age_normalized(lora, match_count: int = None) -> float:
        """
        Yaşa göre normalize başarı
        
        🆕 DEĞİŞİKLİK: Deneyim ödüllendirilir!
        Genç LoRA: %70 başarı = 0.70 skor (normal, bonus yok!)
        Yaşlı LoRA: %70 başarı = 0.90 skor (deneyim bonusu!)
        
        Beklenti: Yaşlılar deneyimlerinden bonus almalı!
        """
        history = lora.fitness_history
        
        if len(history) == 0:
            return 0.0
        
        # Başarı oranı
        success_rate = sum(1 for f in history if f > 0.5) / len(history)
        
        # Yaş faktörü
        if match_count:
            age = match_count - lora.birth_match
        else:
            age = len(history)
        
        # 🆕 DENEYİM BONUSU (yaşlılara avantaj!)
        # 0-50 maç: 1.0x (bonus yok)
        # 50-100 maç: 1.1x
        # 100-200 maç: 1.2x
        # 200+ maç: 1.3x (max bonus)
        experience_bonus = 1.0 + min(age / 666.0, 0.3)  # Max +30% bonus!
        
        # Normalize: Başarı × Deneyim Bonusu
        normalized = success_rate * experience_bonus
        
        # 0-1 arası sınırla
        return min(1.0, normalized)
    
    @staticmethod
    def _calculate_peak_performance(lora) -> float:
        """
        En iyi 20 maçlık dönem performansı
        
        Potansiyeli gösterir!
        """
        history = lora.fitness_history
        
        if len(history) < 20:
            # 🆕 20 maç yok? CEZA! (Çömezler peak alamaz!)
            # Ceza: Mevcut average × (maç_sayısı / 20)
            current_avg = sum(history) / len(history) if len(history) > 0 else 0.0
            penalty = len(history) / 20.0
            return current_avg * penalty  # Ağır ceza!
        
        # 20 maçlık sliding window
        best_avg = 0.0
        for i in range(len(history) - 19):
            window = history[i:i+20]
            window_avg = sum(window) / 20
            best_avg = max(best_avg, window_avg)
        
        return best_avg
    
    @staticmethod
    def _calculate_momentum(lora) -> float:
        """
        Momentum (Trend - yükseliyor mu düşüyor mu?)
        
        Son 20 maç vs Önceki 20 maç
        Yükseliyorsa: 1.0
        Düşüyorsa: 0.0
        Aynı: 0.5
        """
        history = lora.fitness_history
        
        if len(history) < 40:
            # Yeterli veri yok
            return 0.5  # Nötr
        
        # Son 20 maç
        recent_20 = history[-20:]
        recent_avg = sum(recent_20) / 20
        
        # Önceki 20 maç
        previous_20 = history[-40:-20]
        previous_avg = sum(previous_20) / 20
        
        # Momentum hesapla
        if previous_avg > 0:
            momentum_ratio = recent_avg / previous_avg
        else:
            momentum_ratio = 1.0
        
        # 0.5-1.5 arası → 0-1 arası normalize et
        # momentum_ratio:
        #   0.5 → düşüş (0.0)
        #   1.0 → sabit (0.5)
        #   1.5 → artış (1.0)
        normalized_momentum = (momentum_ratio - 0.5) / 1.0  # -0.5 to 0.5 → 0 to 1
        normalized_momentum = max(0.0, min(1.0, (normalized_momentum + 0.5)))
        
        return normalized_momentum
    
    @staticmethod
    def _calculate_consistency(lora) -> float:
        """
        İstikrar (Variance ne kadar düşük?)
        
        Düşük variance = istikrarlı = yüksek skor
        Yüksek variance = kararsız = düşük skor
        """
        history = lora.fitness_history[-50:]  # Son 50 maç
        
        if len(history) < 10:
            return 0.5  # Yeterli veri yok
        
        # Variance hesapla
        mean = sum(history) / len(history)
        variance = sum((f - mean) ** 2 for f in history) / len(history)
        std = variance ** 0.5
        
        # Düşük std = yüksek consistency
        # std: 0.0 → 1.0, 0.3+ → 0.0
        consistency = max(0.0, 1.0 - (std / 0.3))
        
        return consistency
    
    @staticmethod
    def get_detailed_breakdown(lora, match_count: int = None) -> dict:
        """
        Detaylı skor analizi (gelişmiş formül)
        """
        if len(lora.fitness_history) < 5:
            return {
                'total_score': lora.get_recent_fitness() * 0.5,
                'weighted_recent': 0.0,
                'age_normalized': 0.0,
                'peak_performance': 0.0,
                'momentum': 0.5,
                'consistency': 0.5,
                'note': 'Çok genç (< 5 maç)'
            }
        
        # Her bileşeni hesapla
        weighted_recent = AdvancedScoreCalculator._calculate_weighted_recent(lora)
        age_normalized = AdvancedScoreCalculator._calculate_age_normalized(lora, match_count)
        peak = AdvancedScoreCalculator._calculate_peak_performance(lora)
        momentum = AdvancedScoreCalculator._calculate_momentum(lora)
        consistency = AdvancedScoreCalculator._calculate_consistency(lora)
        
        # Toplam
        total = (
            weighted_recent * 0.30 +
            age_normalized * 0.25 +
            peak * 0.20 +
            momentum * 0.15 +
            consistency * 0.10
        )
        
        # Temel istatistikler
        success_rate = sum(1 for f in lora.fitness_history if f > 0.5) / len(lora.fitness_history)
        
        # Streak
        max_streak = 0
        current_streak = 0
        for f in lora.fitness_history:
            if f > 0.5:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return {
            'total_score': total,
            'weighted_recent': weighted_recent,
            'age_normalized': age_normalized,
            'peak_performance': peak,
            'momentum': momentum,
            'consistency': consistency,
            'success_rate': success_rate,
            'max_streak': max_streak,
            'total_matches': len(lora.fitness_history),
            'breakdown': {
                'weighted_recent_contribution': weighted_recent * 0.30,
                'age_normalized_contribution': age_normalized * 0.25,
                'peak_contribution': peak * 0.20,
                'momentum_contribution': momentum * 0.15,
                'consistency_contribution': consistency * 0.10
            }
        }


# Global instance
advanced_score_calculator = AdvancedScoreCalculator()



