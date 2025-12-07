"""
📊 TAKIM UZMANLIK SKORU HESAPLAYICI
====================================

Advanced Score Calculator (Einstein dışındaki uzmanlıklar için!)

FORMÜL:
  SKOR = Accuracy (0.30) + Age (0.20) + Consistency (0.15) + 
         Peak (0.15) + Momentum (0.10) + Match Count (0.10)

Eşik yok, sadece Top 5!
Minimum 20 maç (VS için 5 maç).
"""

import numpy as np
from typing import List, Tuple


def calculate_advanced_team_score(lora, team_name: str, spec_type: str, 
                                  predictions: List, match_count: int) -> float:
    """
    Takım uzmanlık skoru hesapla!
    
    🎯 ÖNEMLİ: SADECE O TAKIMIN MAÇLARI SAYILIR!
    - Momentum: Sadece Manchester maçlarında
    - Peak: Sadece Manchester maçlarında
    - Consistency: Sadece Manchester maçlarında
    
    Args:
        lora: LoRA instance
        team_name: Takım ismi
        spec_type: 'WIN', 'GOAL', 'HYPE', 'VS_opponent'
        predictions: Tahmin listesi (SADECE o takım/eşleşme için!)
        match_count: Toplam maç sayısı
    
    Returns:
        0-1 arası advanced score
    """
    
    if len(predictions) < 5:  # Çok az veri
        return 0.0
    
    # ============================================
    # 1) ACCURACY (Başarı) - %30
    # ============================================
    if spec_type == 'WIN' or spec_type == 'HYPE' or spec_type.startswith('VS_'):
        # Win/Hype/VS: [(correct, match_idx), ...]
        correct_count = sum(1 for (correct, _) in predictions if correct)
        accuracy = correct_count / len(predictions)
    
    elif spec_type == 'GOAL':
        # Goal: [(predicted, actual, match_idx), ...]
        mae = np.mean([abs(pred - actual) for (pred, actual, _) in predictions])
        # MAE'yi 0-1 skalaya çevir (0 MAE = 1.0, 3 MAE = 0.0)
        accuracy = max(0, 1 - (mae / 3.0))
    
    else:
        accuracy = 0.5
    
    accuracy_score = accuracy * 0.30
    
    # ============================================
    # 2) AGE NORMALIZED (Deneyim) - %20
    # ============================================
    age = match_count - lora.birth_match
    
    if age >= 100:
        age_normalized = 1.0  # Çok deneyimli
    elif age >= 50:
        age_normalized = 0.8
    elif age >= 20:
        age_normalized = 0.6
    elif age >= 10:
        age_normalized = 0.4
    elif age >= 5:
        age_normalized = 0.2
    else:
        age_normalized = 0.0  # Minimum 5 maç
    
    age_score = age_normalized * 0.20
    
    # ============================================
    # 3) CONSISTENCY (İstikrar) - %15
    # ============================================
    # 🎯 SADECE BU TAKIMIN MAÇLARINDA! (predictions zaten filtrelenmiş)
    recent = predictions[-20:] if len(predictions) > 20 else predictions
    
    if spec_type == 'WIN' or spec_type == 'HYPE' or spec_type.startswith('VS_'):
        # Doğru/yanlış varyansı (SADECE bu takımda!)
        recent_acc = [1.0 if correct else 0.0 for (correct, _) in recent]
    elif spec_type == 'GOAL':
        # MAE varyansı (SADECE bu takımda!)
        recent_mae = [abs(pred - actual) for (pred, actual, _) in recent]
        recent_acc = [max(0, 1 - (mae / 3.0)) for mae in recent_mae]
    else:
        recent_acc = [0.5]
    
    variance = np.var(recent_acc)
    consistency = max(0, 1 - variance)  # Düşük varyans = yüksek skor
    consistency_score = consistency * 0.15
    
    # 🎯 NOT: Consistency sadece bu takımın maçlarına bakıyor!
    # Örn: Manchester için → Sadece Manchester maçlarındaki varyans
    
    # ============================================
    # 4) PEAK PERFORMANCE (En iyi dönem) - %15
    # ============================================
    # 🎯 En iyi 10 maçlık dönem (SADECE BU TAKIMDA!)
    if len(predictions) >= 10:
        peak_accuracy = 0.0
        for i in range(len(predictions) - 9):
            window = predictions[i:i+10]
            
            if spec_type == 'WIN' or spec_type == 'HYPE' or spec_type.startswith('VS_'):
                window_acc = sum(1 for (correct, _) in window if correct) / 10.0
            elif spec_type == 'GOAL':
                window_mae = np.mean([abs(pred - actual) for (pred, actual, _) in window])
                window_acc = max(0, 1 - (window_mae / 3.0))
            
            peak_accuracy = max(peak_accuracy, window_acc)
    else:
        peak_accuracy = accuracy  # Yeterli veri yoksa genel accuracy
    
    peak_score = peak_accuracy * 0.15
    
    # 🎯 NOT: Peak sadece bu takımın maçlarındaki en iyi dönem!
    # Örn: Manchester için → Manchester maçlarındaki en iyi 10 maç
    
    # ============================================
    # 5) MOMENTUM (Trend) - %10
    # ============================================
    # 🎯 Trend (SADECE BU TAKIMIN MAÇLARINDA!)
    if len(predictions) >= 10:
        # İlk yarı vs İkinci yarı
        first_half = predictions[:len(predictions)//2]
        second_half = predictions[len(predictions)//2:]
        
        if spec_type == 'WIN' or spec_type == 'HYPE' or spec_type.startswith('VS_'):
            first_acc = sum(1 for (correct, _) in first_half if correct) / max(1, len(first_half))
            second_acc = sum(1 for (correct, _) in second_half if correct) / max(1, len(second_half))
        elif spec_type == 'GOAL':
            first_mae = np.mean([abs(pred - actual) for (pred, actual, _) in first_half])
            second_mae = np.mean([abs(pred - actual) for (pred, actual, _) in second_half])
            first_acc = max(0, 1 - (first_mae / 3.0))
            second_acc = max(0, 1 - (second_mae / 3.0))
        
        momentum = second_acc - first_acc  # Pozitif = yükseliş
        momentum_normalized = max(0, min(1, (momentum + 0.5)))  # -0.5 ile +0.5 arası normalize
    else:
        momentum_normalized = 0.5
    
    momentum_score = momentum_normalized * 0.10
    
    # 🎯 NOT: Momentum sadece bu takımın maçlarındaki trend!
    # Örn: Manchester için → Manchester maçlarında yükseliyor mu?
    #      İlk 25 Man maçı: %75, Son 25 Man maçı: %85 → +%10 momentum!
    
    # ============================================
    # 6) MATCH COUNT BONUS (Maç sayısı) - %10
    # ============================================
    match_count_team = len(predictions)
    
    if spec_type.startswith('VS_'):
        # VS için daha düşük eşikler (az eşleşme olur)
        if match_count_team >= 20:
            match_bonus = 1.0
        elif match_count_team >= 10:
            match_bonus = 0.8
        elif match_count_team >= 5:
            match_bonus = 0.5
        else:
            match_bonus = 0.0
    else:
        # Win/Goal/Hype için
        if match_count_team >= 50:
            match_bonus = 1.0
        elif match_count_team >= 20:
            match_bonus = 0.8
        elif match_count_team >= 10:
            match_bonus = 0.6
        elif match_count_team >= 5:
            match_bonus = 0.4
        elif match_count_team >= 3:
            match_bonus = 0.2
        else:
            match_bonus = 0.0  # Minimum 3!
    
    match_bonus_score = match_bonus * 0.10
    
    # ============================================
    # TOPLAM SKOR
    # ============================================
    total_score = (
        accuracy_score +
        age_score +
        consistency_score +
        peak_score +
        momentum_score +
        match_bonus_score
    )
    
    return min(1.0, total_score)  # Max 1.0

