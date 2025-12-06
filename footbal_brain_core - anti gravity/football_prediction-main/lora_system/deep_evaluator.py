import math

class DeepEvaluator:
    """
    🧠 DEEP MATH EVALUATOR
    ======================
    
    Bu modül, LoRA'ların performansını "basit ortalama" yerine
    İstatistiksel ve Olasılıksal yöntemlerle (Bayesian Inference) değerlendirir.
    
    Amaç:
    - Az maç yapan "şanslı" LoRA'ları, çok maç yapan "istikrarlı" LoRA'ların önüne geçirmemek.
    - "Telef olma" riskini sıfıra indirmek (Gerçek yetenekleri ıskalamamak).
    """
    
    @staticmethod
    def calculate_bayesian_score(correct: int, total: int, total_confidence: float = 0.0) -> float:
        """
        Wilson Score Interval (Lower Bound) kullanarak güvenilir skor hesaplar.
        
        Mantık:
        "Bu LoRA'nın gerçek başarı oranı %90 ihtimalle EN AZ kaçtır?" sorusunun cevabıdır.
        
        Neden Wilson Score?
        - 2/2 yapan (%100) ile 100/100 yapan (%100) aynı değildir.
        - Wilson Score, az maç yapanı "cezalandırmaz" ama "şüpheyle yaklaşır".
        - Veri arttıkça skor, gerçek başarı oranına (p_hat) yakınsar.
        
        Args:
            correct: Doğru tahmin sayısı
            total: Toplam maç sayısı
            total_confidence: Toplam güven skoru (Opsiyonel, kalibrasyon için)
            
        Returns:
            0.0 - 1.0 arası "Güvenilir Skor"
        """
        if total == 0:
            return 0.0
            
        # 1. Wilson Score Interval (Lower Bound)
        # z = 1.28 (Approx 90% confidence)
        # Bu değer, "şans eseri" başarıyı elemek için idealdir.
        z = 1.28 
        
        p_hat = correct / total
        
        numerator = p_hat + (z*z)/(2*total) - z * math.sqrt((p_hat*(1-p_hat)/total) + (z*z)/(4*total*total))
        denominator = 1 + (z*z)/total
        
        wilson_score = numerator / denominator
        
        # 2. Confidence Calibration (Güven Kalibrasyonu)
        # LoRA ne kadar emin? (Emin olduğu maçları biliyor mu?)
        # Eğer total_confidence verildiyse, bunu küçük bir "bonus" veya "teyit" olarak kullanalım.
        # Ama ana belirleyici Wilson Score'dur.
        
        avg_confidence = total_confidence / total if total > 0 else 0
        
        # Kalibrasyon Bonusu:
        # Eğer LoRA çok eminse ve doğru biliyorsa, Wilson skorunu biraz yukarı itelim.
        # Amaç: "Cesur ve Doğru" olanı ödüllendirmek.
        
        final_score = wilson_score
        
        if avg_confidence > 0:
            # Güven ile doğruluk arasındaki uyum
            # (Basit bir ağırlıklandırma)
            final_score = (wilson_score * 0.85) + (avg_confidence * 0.15)
            
        return max(0.0, min(1.0, final_score))

    @staticmethod
    def calculate_trend_bonus(history: list) -> float:
        """
        Son maçlardaki performans artışını (Momentum) ölçer.
        Deep Learning mantığı: Sequence Analysis.
        
        Args:
            history: [True, False, True, True, ...] (Eskiden yeniye)
            
        Returns:
            0.0 - 0.1 arası bonus puan
        """
        if not history or len(history) < 3:
            return 0.0
            
        # Son 5 maça ağırlık ver
        recent = history[-5:]
        recent_acc = sum(recent) / len(recent)
        
        # Tüm geçmiş
        overall_acc = sum(history) / len(history)
        
        # Eğer son performans, genelden iyiyse "Öğreniyor" demektir.
        diff = recent_acc - overall_acc
        
        if diff > 0:
            return min(0.1, diff * 0.5) # Maksimum 0.1 bonus
        return 0.0
