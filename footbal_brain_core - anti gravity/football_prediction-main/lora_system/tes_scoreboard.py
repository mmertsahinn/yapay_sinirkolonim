"""
🔬 TES SCOREBOARD (Termodinamik Evrimsel Skor!)
================================================

ESKİ SCOREBOARD:
  weighted_recent + age_normalized + peak + momentum + consistency

YENİ SCOREBOARD (TES!):
  Ω_i = Darwin + Einstein + Newton + Bonuslar

DARWIN:  Popülasyona katkı (Price)
EINSTEIN: Sürprizler (KL-Divergence)
NEWTON:  İstikrar (Flat Minima)
"""

import numpy as np
from typing import List, Dict


class TESScoreboard:
    """
    TES bazlı scoreboard sistemi
    """
    
    def __init__(self):
        # Ağırlıklar
        self.w_darwin = 0.35
        self.w_einstein = 0.35
        self.w_newton = 0.30
        
        print("🔬 TES Scoreboard başlatıldı")
    
    def calculate_tes_score(self, lora, population: List, collective_memory: Dict = None) -> Dict:
        """
        TES SKORU HESAPLA!
        
        Ω_i = Darwin + Einstein + Newton
        
        Returns:
            {
                'total_tes': float,
                'darwin': float,
                'einstein': float,
                'newton': float,
                'tier': str,
                'rank_potential': float
            }
        """
        # ============================================
        # 1. DARWIN TERİMİ (Popülasyona Katkı!)
        # ============================================
        darwin_score = self._calculate_darwin_simple(lora, population)
        
        # ============================================
        # 2. EINSTEIN TERİMİ (Sürprizler!)
        # ============================================
        einstein_score = self._calculate_einstein_from_memory(lora, collective_memory)
        
        # ============================================
        # 3. NEWTON TERİMİ (İstikrar!)
        # ============================================
        newton_score = self._calculate_newton_simple(lora)
        
        # ============================================
        # TOPLAM TES SKORU
        # ============================================
        total_tes = (
            self.w_darwin * darwin_score +
            self.w_einstein * einstein_score +
            self.w_newton * newton_score
        )
        
        # ============================================
        # TİP TESPİTİ! (Einstein/Newton/Darwin!)
        # ============================================
        
        lora_type = self._determine_type(darwin_score, einstein_score, newton_score)
        
        # TIER BELİRLE (TES bazlı!)
        if total_tes >= 0.80:
            tier = f"Efsane ({lora_type})"
        elif total_tes >= 0.65:
            tier = f"Usta ({lora_type})"
        elif total_tes >= 0.50:
            tier = f"Uzman ({lora_type})"
        elif total_tes >= 0.35:
            tier = f"İyi ({lora_type})"
        else:
            tier = "Gelişiyor"
        
        return {
            'total_tes': total_tes,
            'darwin': darwin_score,
            'einstein': einstein_score,
            'newton': newton_score,
            'lora_type': lora_type,  # ⭐ YENİ: Tip!
            'tier': tier,
            'rank_potential': total_tes,
            'breakdown': {
                'darwin_contribution': darwin_score * self.w_darwin,
                'einstein_contribution': einstein_score * self.w_einstein,
                'newton_contribution': newton_score * self.w_newton
            }
        }
    
    def _determine_type(self, darwin: float, einstein: float, newton: float) -> str:
        """
        LoRA tipini belirle! (Hangisi baskın?)
        
        🌊 AKIŞKAN KRİTERLER:
        - Sadece en yüksek olanı değil, göreceli üstünlüğü de kontrol et
        - Düşük eşikler (yeni popülasyonlar için)
        
        Returns:
            'EINSTEIN⭐', 'NEWTON🏛️', 'DARWIN🧬', 'HYBRID🌟', 'DENGELI⚖️'
        """
        # En yüksek skor hangisi?
        max_score = max(darwin, einstein, newton)
        
        # Göreceli fark (2. en yüksekten ne kadar fazla?)
        scores = sorted([darwin, einstein, newton], reverse=True)
        dominance = scores[0] - scores[1]  # İlk ile ikinci arasındaki fark
        
        # BASKINNLIK EŞİĞİ: 0.10'dan fazla fark varsa baskın
        dominance_threshold = 0.10
        
        # EINSTEIN TİPİ (Einstein en yüksek + baskın)
        if einstein == max_score and dominance >= dominance_threshold:
            return "EINSTEIN⭐"
        
        # NEWTON TİPİ (Newton en yüksek + baskın)
        elif newton == max_score and dominance >= dominance_threshold:
            return "NEWTON🏛️"
        
        # DARWIN TİPİ (Darwin en yüksek + baskın)
        elif darwin == max_score and dominance >= dominance_threshold:
            return "DARWIN🧬"
        
        # 🆕 HYBRID HİYERARŞİSİ (3 SEVİYE!)
        
        # 💎 PERFECT HYBRID (EN YÜKSEK! - Üçünde de mükemmel!)
        # Üçü de 0.75+ → PERFECT HYBRID!
        if einstein >= 0.75 and newton >= 0.75 and darwin >= 0.75:
            return "PERFECT HYBRID💎💎💎"
        
        # 🌟🌟 STRONG HYBRID (İKİNCİ SEVİYE! - Üçünde de güçlü!)
        # Üçü de 0.50+ → Strong Hybrid
        elif einstein >= 0.50 and newton >= 0.50 and darwin >= 0.50:
            return "STRONG HYBRID🌟🌟"
        
        # 🌟 HYBRID (ÜÇÜNCÜ SEVİYE! - Üçünde de iyi!)
        # Üçü de 0.30+ → Normal Hybrid
        elif einstein >= 0.30 and newton >= 0.30 and darwin >= 0.30:
            return "HYBRID🌟"
        
        # İKİLİ HYBRID'LER (Sadece ikisi güçlü)
        # Einstein + Newton
        elif einstein >= 0.25 and newton >= 0.25 and abs(einstein - newton) < 0.15:
            return "HYBRID(E-N)⚡"
        
        # Einstein + Darwin
        elif einstein >= 0.25 and darwin >= 0.25 and abs(einstein - darwin) < 0.15:
            return "HYBRID(E-D)⚡"
        
        # Newton + Darwin
        elif newton >= 0.25 and darwin >= 0.25 and abs(newton - darwin) < 0.15:
            return "HYBRID(N-D)⚡"
        
        # ZAYİF BASKINLIK (Fark var ama çok az)
        # Einstein biraz önde
        elif einstein == max_score and einstein > 0.20:
            return "EINSTEIN⭐"
        
        # Newton biraz önde
        elif newton == max_score and newton > 0.20:
            return "NEWTON🏛️"
        
        # Darwin biraz önde
        elif darwin == max_score and darwin > 0.20:
            return "DARWIN🧬"
        
        # DENGELI (Hiçbiri baskın değil veya hepsi çok düşük)
        else:
            return "DENGELI⚖️"
    
    def _calculate_darwin_simple(self, lora, population: List) -> float:
        """
        DARWIN (Basitleştirilmiş!):
        
        Popülasyona katkı = Fitness farkı × Mizaç uyumu
        """
        if len(population) < 3:
            return 0.5
        
        # Bu LoRA'nın fitness'ı
        lora_fitness = lora.get_recent_fitness()
        
        # Popülasyon ortalaması
        pop_avg_fitness = np.mean([l.get_recent_fitness() for l in population])
        
        # Katkı = Fitness - Ortalama
        contribution = lora_fitness - pop_avg_fitness
        
        # Normalize (0-1)
        darwin = 0.5 + contribution  # 0.5 = nötr
        darwin = max(0.0, min(1.0, darwin))
        
        return darwin
    
    def _calculate_einstein_from_memory(self, lora, collective_memory: Dict = None) -> float:
        """
        EINSTEIN (Hafızadan!):
        
        Kolektif hafızada ne kadar "sürpriz" başarısı var?
        """
        if not collective_memory or len(collective_memory) == 0:
            return 0.5  # Henüz hafıza yok
        
        # Hafızadan bu LoRA'nın sürpriz başarılarını say
        surprise_successes = 0
        total_predictions = 0
        
        for match_key, match_data in collective_memory.items():
            lora_insights = match_data.get('lora_insights', {})
            
            if lora.id in lora_insights:
                insight = lora_insights[lora.id]
                
                # Konsensüsten farklı mıydı?
                consensus = match_data.get('consensus', {}).get('majority', '')
                lora_prediction = insight.get('prediction', '')
                lora_correct = insight.get('correct', False)
                
                total_predictions += 1
                
                # Sürpriz başarısı: Konsensüsten farklı + Doğru!
                if lora_prediction != consensus and lora_correct:
                    surprise_successes += 1
        
        # Sürpriz oranı
        if total_predictions > 0:
            surprise_ratio = surprise_successes / total_predictions
        else:
            surprise_ratio = 0.0
        
        # Einstein skoru (0-1)
        # %10 sürpriz = 0.50
        # %30 sürpriz = 0.90
        einstein = 0.50 + (surprise_ratio * 2.0)
        einstein = max(0.0, min(1.0, einstein))
        
        return einstein
    
    def _calculate_newton_simple(self, lora) -> float:
        """
        NEWTON (Basitleştirilmiş!):
        
        İstikrar = Düşük variance + Yüksek consistency
        """
        if len(lora.fitness_history) < 10:
            return 0.5  # Yetersiz veri
        
        recent = lora.fitness_history[-50:]
        
        # Variance
        variance = np.var(recent)
        
        # Consistency score (Düşük variance = yüksek puan!)
        consistency = max(0.0, 1.0 - (variance / 0.3))
        
        # Newton = İstikrar
        newton = consistency
        
        return newton


# Global instance
tes_scoreboard = TESScoreboard()

