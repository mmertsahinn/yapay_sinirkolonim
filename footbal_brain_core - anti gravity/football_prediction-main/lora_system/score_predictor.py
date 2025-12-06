"""
SKOR TAHMİN SİSTEMİ
===================

xG (Expected Goals) verilerinden maç skoru tahmini.

Kullanılan Yöntem:
- Poisson Distribution (futbol skorları için standart)
- xG değerleri beklenen gol sayısını verir
- Her takım için en olası gol sayısını hesapla
"""

import numpy as np
from scipy.stats import poisson
from typing import Tuple, Dict

class ScorePredictor:
    """
    Skor tahmin motoru
    """
    
    def __init__(self):
        self.history = []
    
    def predict_score_from_xg(self, home_xg: float, away_xg: float) -> Tuple[int, int]:
        """
        xG verilerinden en olası skoru tahmin et
        
        Args:
            home_xg: Ev sahibi expected goals
            away_xg: Deplasman expected goals
            
        Returns:
            (home_goals, away_goals): Tahmin edilen skor
        """
        # xG negatif veya çok düşükse minimum değer
        home_xg = max(0.1, home_xg) if not np.isnan(home_xg) else 1.0
        away_xg = max(0.1, away_xg) if not np.isnan(away_xg) else 1.0
        
        # Poisson dağılımından en olası değeri al
        home_goals = int(np.round(home_xg))
        away_goals = int(np.round(away_xg))
        
        # Maksimum 6 gol (aşırı değerleri sınırla)
        home_goals = min(6, max(0, home_goals))
        away_goals = min(6, max(0, away_goals))
        
        return home_goals, away_goals
    
    def predict_score_probabilistic(self, home_xg: float, away_xg: float, 
                                    max_goals: int = 5) -> Dict[Tuple[int, int], float]:
        """
        Tüm olası skorların olasılıklarını hesapla
        
        Returns:
            {(home_goals, away_goals): probability, ...}
        """
        home_xg = max(0.1, home_xg) if not np.isnan(home_xg) else 1.0
        away_xg = max(0.1, away_xg) if not np.isnan(away_xg) else 1.0
        
        score_probs = {}
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                # Poisson olasılıkları
                prob_h = poisson.pmf(h, home_xg)
                prob_a = poisson.pmf(a, away_xg)
                
                # Bağımsız olaylar (çarpım)
                score_probs[(h, a)] = prob_h * prob_a
        
        return score_probs
    
    def calculate_score_fitness(self, predicted: Tuple[int, int], 
                                actual: Tuple[int, int]) -> Dict[str, float]:
        """
        Tahmin edilen skor ile gerçek skoru karşılaştır
        
        Fitness Sistemi:
        - Tam skor: +5 puan (1-0 vs 1-0)
        - Gol farkı: +2 puan (1-0 vs 2-1, ikisi de +1 fark)
        - Yakın skor: +1 puan (her gol 1 hata içinde)
        - Kazanan doğru: +1 puan (HOME/DRAW/AWAY)
        
        Returns:
            {
                'exact_score': 0 veya 1,
                'goal_difference': 0 veya 1,
                'close_score': 0 veya 1,
                'correct_winner': 0 veya 1,
                'total_fitness': toplam puan
            }
        """
        # None kontrolü! xG yoksa skor tahmini yapılamaz
        if predicted is None or actual is None:
            return {
                'exact_score': 0.0,
                'goal_difference': 0.0,
                'close_score': 0.0,
                'correct_winner': 0.0,
                'total_fitness': 0.0  # Nötr
            }
        
        pred_h, pred_a = predicted
        actual_h, actual_a = actual
        
        fitness = {
            'exact_score': 0.0,
            'goal_difference': 0.0,
            'close_score': 0.0,
            'correct_winner': 0.0,
            'total_fitness': 0.0
        }
        
        # 1. TAM SKOR (+5 puan)
        if pred_h == actual_h and pred_a == actual_a:
            fitness['exact_score'] = 5.0
        
        # 2. GOL FARKI DOĞRU (+2 puan)
        pred_diff = pred_h - pred_a
        actual_diff = actual_h - actual_a
        
        if pred_diff == actual_diff:
            fitness['goal_difference'] = 2.0
        
        # 3. YAKIN SKOR (+1 puan) - Her gol max 1 hata
        home_error = abs(pred_h - actual_h)
        away_error = abs(pred_a - actual_a)
        
        if home_error <= 1 and away_error <= 1 and (home_error + away_error) <= 2:
            fitness['close_score'] = 1.0
        
        # 4. KAZANAN DOĞRU (+1 puan)
        pred_winner = 'HOME' if pred_h > pred_a else ('AWAY' if pred_a > pred_h else 'DRAW')
        actual_winner = 'HOME' if actual_h > actual_a else ('AWAY' if actual_a > actual_h else 'DRAW')
        
        if pred_winner == actual_winner:
            fitness['correct_winner'] = 1.0
        
        # Toplam
        fitness['total_fitness'] = (
            fitness['exact_score'] +
            fitness['goal_difference'] +
            fitness['close_score'] +
            fitness['correct_winner']
        )
        
        return fitness
    
    def get_winner_from_score(self, home_goals: int, away_goals: int) -> str:
        """Skordan kazananı belirle"""
        if home_goals > away_goals:
            return 'HOME'
        elif away_goals > home_goals:
            return 'AWAY'
        else:
            return 'DRAW'


# Global instance
score_predictor = ScorePredictor()



