"""
🧠 META-ADAPTIF ÖĞRENME HIZI
==============================

Her LoRA kendi learning rate'ini bulur!

SABİT LEARNING RATE YOK!

NASIL ÇALIŞIR:
1. Başlangıç: Mizaç bazlı (Sabırlı: yavaş, Dürtüsel: hızlı)
2. Başarılıysa → Hızlan!
3. Başarısızsa → Yavaşla!
4. Overfit tespiti → Düşür!
5. Underfit tespiti → Artır!

Her LoRA kendi optimal hızını bulur!
"""

from typing import Dict, List, Tuple
import numpy as np


class MetaAdaptiveLearning:
    """
    Meta-adaptif öğrenme hız sistemi
    """
    
    def __init__(self):
        # Her LoRA'nın öğrenme hız geçmişi
        self.learning_rates = {}  # lora_id -> current_lr
        self.lr_history = {}      # lora_id -> [lr_history]
        self.performance_history = {}  # lora_id -> [(lr, performance)]
    
    def initialize_learning_rate(self, lora, base_lr: float = 0.0001) -> float:
        """
        Başlangıç learning rate'i belirle (MİZAÇ BAZLI!)
        
        Args:
            lora: LoRA instance
            base_lr: Base learning rate
        
        Returns:
            İlk learning rate
        """
        temp = lora.temperament
        
        # MİZAÇ FAKTÖRLERI
        patience = temp.get('patience', 0.5)
        impulsiveness = temp.get('impulsiveness', 0.5)
        risk_appetite = temp.get('risk_appetite', 0.5)
        
        # FORMÜL:
        # Sabırlı → Yavaş öğren (dikkatli!)
        # Dürtüsel → Hızlı öğren (agresif!)
        # Risk sever → Hızlı öğren
        
        temperament_multiplier = (
            (1.0 - patience) * 0.40 +      # Sabırsız → hızlı
            impulsiveness * 0.35 +          # Dürtüsel → hızlı
            risk_appetite * 0.25            # Risk sever → hızlı
        )
        
        # 0.5 - 2.0 arası
        temperament_multiplier = 0.5 + (temperament_multiplier * 1.5)
        
        initial_lr = base_lr * temperament_multiplier
        
        # Kaydet
        self.learning_rates[lora.id] = initial_lr
        self.lr_history[lora.id] = [initial_lr]
        self.performance_history[lora.id] = []
        
        return initial_lr
    
    def adapt_learning_rate(self, lora, recent_performance: List[float], 
                           current_lr: float = None) -> Tuple[float, str]:
        """
        Learning rate'i adapte et! (META-LEARNING!)
        
        Args:
            lora: LoRA instance
            recent_performance: Son 10 maçın fitness'ı
            current_lr: Mevcut learning rate
        
        Returns:
            (new_lr, reason)
        """
        if current_lr is None:
            current_lr = self.learning_rates.get(lora.id, 0.0001)
        
        if len(recent_performance) < 5:
            return current_lr, "Yetersiz veri"
        
        # ============================================
        # PERFORMANS ANALİZİ
        # ============================================
        
        # Trend (yükseliyor mu?)
        first_half = recent_performance[:len(recent_performance)//2]
        second_half = recent_performance[len(recent_performance)//2:]
        
        trend = np.mean(second_half) - np.mean(first_half)
        
        # Variance (stabil mi?)
        variance = np.var(recent_performance)
        
        # Son performans
        recent_avg = np.mean(recent_performance[-5:])
        
        # ============================================
        # KARAR (AKIŞKAN FORMÜL!)
        # ============================================
        
        adjustment = 1.0  # Çarpan (1.0 = değişmez)
        reason = ""
        
        # SENARYO 1: Yükseliyor + Düşük variance → HIZLAN!
        if trend > 0.05 and variance < 0.02:
            adjustment = 1.15  # %15 artır
            reason = "Performans yükseliyor, hızlanıyorum!"
        
        # SENARYO 2: Düşüyor → YAVAŞLA!
        elif trend < -0.05:
            adjustment = 0.85  # %15 düşür
            reason = "Performans düşüyor, yavaşlıyorum"
        
        # SENARYO 3: Yüksek variance → OVERFIT! Yavaşla!
        elif variance > 0.05:
            adjustment = 0.80  # %20 düşür
            reason = "Çok dalgalı (overfit?), yavaşlıyorum"
        
        # SENARYO 4: Düşük performans + Düşük variance → UNDERFIT! Hızlan!
        elif recent_avg < 0.50 and variance < 0.01:
            adjustment = 1.20  # %20 artır
            reason = "Underfit, daha agresif öğreniyorum!"
        
        # SENARYO 5: İyi performans → KORU!
        elif recent_avg > 0.70:
            adjustment = 1.0  # Değiştirme
            reason = "Performans iyi, değiştirmiyorum"
        
        # Değiştirme
        else:
            adjustment = 1.0
            reason = "Stabil"
        
        # YENİ LEARNING RATE
        new_lr = current_lr * adjustment
        
        # Sınırla (0.00001 - 0.001 arası)
        new_lr = max(0.00001, min(0.001, new_lr))
        
        # Kaydet
        self.learning_rates[lora.id] = new_lr
        self.lr_history[lora.id].append(new_lr)
        self.performance_history[lora.id].append((new_lr, recent_avg))
        
        return new_lr, reason
    
    def get_optimal_lr_for_lora(self, lora) -> float:
        """
        LoRA'nın mevcut optimal learning rate'i
        """
        return self.learning_rates.get(lora.id, 0.0001)


# Global instance
meta_adaptive_learning = MetaAdaptiveLearning()

