"""
🏆 TOP 50 SKOR HESAPLAYICI (DENGELİ FORMÜL)
=============================================

KİMSE TORPİLLİ DEĞİL, ADALET VAR!

YENİ DENGELİ FORMÜL:
TOP_SCORE = Ham Performans (50%) + Tutarlılık (20%) + Deneyim (20%) + Potansiyel (10%)

KRİTERLER:
1. HAM PERFORMANS (50%) - Gerçek yetenek
   - Fitness: 30%
   - Başarı Oranı: 20%

2. TUTARLILIK (20%) - İstikrar
   - Streak Score: 20%

3. DENEYİM BONUSU (20%) - Yaşlılara adil avantaj
   - Kademeli bonus:
     * 0-50 maç: Bonus yok (genç)
     * 50-100 maç: +10% bonus
     * 100-150 maç: +15% bonus
     * 150+ maç: +20% bonus (max)

4. POTANSİYEL FAKTÖRÜ (10%) - Genç dahiler için
   - Yüksek fitness + genç yaş = ek puan

SONUÇ: Genç dahi → Girer, Deneyimli usta → Girer, İkisi de kötü → Giremez
"""


class TopScoreCalculator:
    """
    Dengeli Top 50 skor hesaplayıcı - Adil sistem!
    """
    
    @staticmethod
    def calculate_top_score(lora, match_count: int = None) -> float:
        """
        Dengeli karma skor hesapla
        
        Returns:
            float: 0-1 arası top score
        """
        # ═══════════════════════════════════════
        # 1) HAM PERFORMANS (50%)
        # ═══════════════════════════════════════
        
        # 1a) Fitness (30%)
        fitness = lora.get_recent_fitness()
        fitness_score = fitness * 0.30
        
        # 1b) Başarı Yüzdesi (20%)
        if len(lora.fitness_history) > 0:
            correct_count = sum(1 for f in lora.fitness_history if f > 0.5)
            total_count = len(lora.fitness_history)
            success_rate = correct_count / total_count
        else:
            success_rate = 0.0
        
        success_score = success_rate * 0.20
        
        # ═══════════════════════════════════════
        # 2) TUTARLILIK (20%)
        # ═══════════════════════════════════════
        streak_score = TopScoreCalculator._calculate_streak_score(lora) * 0.20
        
        # ═══════════════════════════════════════
        # 3) DENEYİM BONUSU (20%)
        # ═══════════════════════════════════════
        if match_count:
            age = match_count - lora.birth_match
        else:
            age = len(lora.fitness_history) if hasattr(lora, 'fitness_history') else 0
        
        # Kademeli deneyim bonusu (adil!)
        if age < 50:
            experience_bonus = 0.0  # Genç, bonus yok
        elif age < 100:
            experience_bonus = 0.10  # +10% bonus
        elif age < 150:
            experience_bonus = 0.15  # +15% bonus
        else:
            experience_bonus = 0.20  # +20% max bonus
        
        # Deneyim skoru: Base performans × (1 + bonus)
        base_performance = (fitness + success_rate) / 2.0
        experience_score = base_performance * experience_bonus * 0.20
        
        # ═══════════════════════════════════════
        # 4) POTANSİYEL FAKTÖRÜ (10%)
        # ═══════════════════════════════════════
        # Genç + yüksek performans = dahi adayı!
        potential_score = 0.0
        if age < 50 and fitness > 0.70:
            # Genç ama çok iyi → ek puan!
            potential_multiplier = (fitness - 0.70) / 0.30  # 0.70-1.0 arası normalize
            potential_score = potential_multiplier * 0.10
        
        # ═══════════════════════════════════════
        # TOPLAM SKOR
        # ═══════════════════════════════════════
        total_score = fitness_score + success_score + streak_score + experience_score + potential_score
        
        # 0-1 arası sınırla (güvenlik)
        return min(1.0, total_score)
    
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
        Detaylı skor analizi (Yeni Dengeli Formül)
        
        Returns:
            {
                'total_score': 0.675,
                'fitness': 0.85,
                'success_rate': 0.75,
                'max_streak': 15,
                'age': 100,
                'experience_bonus': 0.15,
                'breakdown': {
                    'fitness_contribution': 0.255,
                    'success_contribution': 0.15,
                    'streak_contribution': 0.15,
                    'experience_contribution': 0.10,
                    'potential_contribution': 0.02
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
        
        # Yaş ve deneyim bonusu
        if match_count:
            age = match_count - lora.birth_match
        else:
            age = len(lora.fitness_history) if hasattr(lora, 'fitness_history') else 0
        
        if age < 50:
            experience_bonus = 0.0
        elif age < 100:
            experience_bonus = 0.10
        elif age < 150:
            experience_bonus = 0.15
        else:
            experience_bonus = 0.20
        
        # Katkılar (Yeni formül)
        fitness_contrib = fitness * 0.30
        success_contrib = success_rate * 0.20
        streak_contrib = streak_normalized * 0.20
        
        base_performance = (fitness + success_rate) / 2.0
        experience_contrib = base_performance * experience_bonus * 0.20
        
        # Potansiyel
        potential_contrib = 0.0
        if age < 50 and fitness > 0.70:
            potential_multiplier = (fitness - 0.70) / 0.30
            potential_contrib = potential_multiplier * 0.10
        
        total = fitness_contrib + success_contrib + streak_contrib + experience_contrib + potential_contrib
        total = min(1.0, total)
        
        return {
            'total_score': total,
            'fitness': fitness,
            'success_rate': success_rate,
            'correct_count': correct_count,
            'total_count': total_count,
            'max_streak': max_streak,
            'age': age,
            'experience_bonus': experience_bonus,
            'breakdown': {
                'fitness_contribution': fitness_contrib,
                'success_contribution': success_contrib,
                'streak_contribution': streak_contrib,
                'experience_contribution': experience_contrib,
                'potential_contribution': potential_contrib
            }
        }


# Global instance
top_score_calculator = TopScoreCalculator()



