"""
🏆 TOP 50 SKOR HESAPLAYICI
===========================

Sadece fitness değil, çok kriterli skor!

FORMÜL:
TOP_SCORE = (Fitness × 0.40) + (Başarı % × 0.30) + (Streak × 0.20) + (Yaş × 0.10)

KRİTERLER:
1. Fitness (0-1) - %40 ağırlık
2. Başarı Yüzdesi (0-1) - %30 ağırlık  
3. Streak Score (0-1) - %20 ağırlık
4. Yaş Score (0-1) - %10 ağırlık
"""


class TopScoreCalculator:
    """
    Top 50 için karma skor hesapla
    """
    
    @staticmethod
    def calculate_top_score(lora, match_count: int = None) -> float:
        """
        Karma skor hesapla
        
        Returns:
            float: 0-1 arası top score
        """
        # 1) FITNESS (40%)
        fitness = lora.get_recent_fitness()
        fitness_score = fitness * 0.40
        
        # 2) BAŞARI YÜZDESİ (30%)
        if len(lora.fitness_history) > 0:
            correct_count = sum(1 for f in lora.fitness_history if f > 0.5)
            total_count = len(lora.fitness_history)
            success_rate = correct_count / total_count
        else:
            success_rate = 0.0
        
        success_score = success_rate * 0.30
        
        # 3) STREAK SCORE (20%)
        streak_score = TopScoreCalculator._calculate_streak_score(lora) * 0.20
        
        # 4) YAŞ SCORE (10%)
        if match_count:
            age = match_count - lora.birth_match
            # Normalize: 200 maç = 1.0
            age_normalized = min(1.0, age / 200.0)
        else:
            age_normalized = 0.0
        
        age_score = age_normalized * 0.10
        
        # TOPLAM
        total_score = fitness_score + success_score + streak_score + age_score
        
        return total_score
    
    @staticmethod
    def _calculate_streak_score(lora) -> float:
        """
        Streak skoru hesapla
        
        Returns:
            0-1 arası normalize streak skoru
        """
        if len(lora.fitness_history) < 5:
            return 0.0
        
        # En uzun doğru streak'i bul
        current_streak = 0
        max_streak = 0
        
        for fit in lora.fitness_history:
            if fit > 0.5:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        # Normalize: 50 maç streak = 1.0
        normalized = min(1.0, max_streak / 50.0)
        
        return normalized
    
    @staticmethod
    def get_detailed_breakdown(lora, match_count: int = None) -> dict:
        """
        Detaylı skor analizi
        
        Returns:
            {
                'total_score': 0.675,
                'fitness': 0.85,
                'success_rate': 0.75,
                'max_streak': 15,
                'age': 100,
                'breakdown': {
                    'fitness_contribution': 0.34,
                    'success_contribution': 0.225,
                    'streak_contribution': 0.06,
                    'age_contribution': 0.05
                }
            }
        """
        fitness = lora.get_recent_fitness()
        
        # Başarı yüzdesi
        if len(lora.fitness_history) > 0:
            correct_count = sum(1 for f in lora.fitness_history if f > 0.5)
            total_count = len(lora.fitness_history)
            success_rate = correct_count / total_count
        else:
            correct_count = 0
            total_count = 0
            success_rate = 0.0
        
        # Streak
        max_streak = 0
        current_streak = 0
        for fit in lora.fitness_history:
            if fit > 0.5:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        streak_normalized = min(1.0, max_streak / 50.0)
        
        # Yaş
        if match_count:
            age = match_count - lora.birth_match
            age_normalized = min(1.0, age / 200.0)
        else:
            age = 0
            age_normalized = 0.0
        
        # Katkılar
        fitness_contrib = fitness * 0.40
        success_contrib = success_rate * 0.30
        streak_contrib = streak_normalized * 0.20
        age_contrib = age_normalized * 0.10
        
        total = fitness_contrib + success_contrib + streak_contrib + age_contrib
        
        return {
            'total_score': total,
            'fitness': fitness,
            'success_rate': success_rate,
            'correct_count': correct_count,
            'total_count': total_count,
            'max_streak': max_streak,
            'age': age,
            'breakdown': {
                'fitness_contribution': fitness_contrib,
                'success_contribution': success_contrib,
                'streak_contribution': streak_contrib,
                'age_contribution': age_contrib
            }
        }


# Global instance
top_score_calculator = TopScoreCalculator()



