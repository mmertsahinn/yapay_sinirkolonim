import numpy as np
from typing import Dict, List, Any
import math

class AdvancedCategorization:
    """
    Multi-Dimensional Categorization System for LoRA models.
    Replaces the single-label expertise system with a weighted, multi-dimensional profile.
    
    Ref: 3_ANA_SISTEM_GELISTIRME_PLANI.md
    """
    
    EXPERTISE_TYPES = [
        "DERBY_MASTER",    # Big games
        "HYPE_SURFER",     # High hype games
        "ODDS_HUNTER",     # Value bets / surprise odds
        "CONSISTENCY_GOD", # High win streak
        "HOME_DEFENDER",   # Home team wins
        "AWAY_WARRIOR"     # Away team wins
    ]
    
    # Parameters (Initial)
    ALPHA = 0.30  # Success Rate
    BETA  = 0.25  # Confidence
    GAMMA = 0.20  # Recency
    DELTA = 0.15  # Consistency
    EPSILON = 0.10 # Entropy Penalty
    
    @staticmethod
    def calculate_expertise_weights(lora: Any, match_history: List[Dict], current_match: Dict = None) -> Dict[str, float]:
        """
        Calculates the composite score for each expertise type and returns normalized weights.
        """
        expertise_scores = {}
        
        for expertise_type in AdvancedCategorization.EXPERTISE_TYPES:
            # 1. Success Rate
            success_rate = AdvancedCategorization._get_success_rate(lora, expertise_type, match_history)
            
            # 2. Confidence Weight
            confidence_weight = AdvancedCategorization._get_confidence_weight(lora, expertise_type, match_history)
            
            # 3. Recency Bonus (Last 10 matches)
            recent_history = match_history[-10:] if match_history else []
            recency_bonus = AdvancedCategorization._calculate_recency_bonus(lora, expertise_type, recent_history)
            
            # 4. Consistency Score
            consistency = AdvancedCategorization._calculate_consistency(lora, expertise_type, match_history)
            
            # 5. Entropy Penalty (Generic penalty for lacking focus, strictly calculated per lora usually, 
            # but here we can define it as inverse of specialization count or variance of weights)
            # For this step, we'll calculate it based on how flat their current profile is (if available) or 0 initially.
            entropy_penalty = 0.1 # Placeholder, will refine.
            
            # Composite Score
            composite = (AdvancedCategorization.ALPHA * success_rate + 
                         AdvancedCategorization.BETA * confidence_weight + 
                         AdvancedCategorization.GAMMA * recency_bonus + 
                         AdvancedCategorization.DELTA * consistency - 
                         AdvancedCategorization.EPSILON * entropy_penalty)
            
            expertise_scores[expertise_type] = max(0.0, composite) # Ensure non-negative
        
        # Softmax Normalization
        return AdvancedCategorization._softmax(expertise_scores)

    @staticmethod
    def _softmax(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """Applies Softmax to normalize scores into probabilities (weights)."""
        if not scores:
            return {}
            
        try:
            # Shift values for numerical stability (subtract max)
            max_score = max(scores.values())
            exp_scores = {k: math.exp((v - max_score) / temperature) for k, v in scores.items()}
            total_exp = sum(exp_scores.values())
            
            if total_exp == 0:
                return {k: 1.0 / len(scores) for k in scores} # Uniform distribution if all zero
                
            return {k: v / total_exp for k, v in exp_scores.items()}
        except Exception as e:
            print(f"Softmax error: {e}")
            return {k: 1.0 / len(scores) for k in scores}

    @staticmethod
    def _get_success_rate(lora, expertise_type: str, match_history: List[Dict]) -> float:
        """
        Calculates success rate specifically for matches relevant to the expertise type.
        Uses DeepEvaluator for Bayesian Smoothing if available.
        """
        relevant_matches = AdvancedCategorization._filter_relevant_matches(expertise_type, match_history)
        if not relevant_matches:
            return 0.5 # Neutral start
            
        wins = sum(1 for m in relevant_matches if m.get('is_correct', False))
        total = len(relevant_matches)
        
        # Uses DeepEvaluator for Bayesian Smoothing (Perfect Math)
        from lora_system.deep_evaluator import DeepEvaluator
        
        total_confidence = sum(m.get('confidence', 0.5) for m in relevant_matches)
        return DeepEvaluator.calculate_bayesian_score(wins, total, total_confidence)

    @staticmethod
    def _get_confidence_weight(lora, expertise_type: str, match_history: List[Dict]) -> float:
        """
        Average confidence in successful predictions for this expertise.
        """
        relevant_matches = AdvancedCategorization._filter_relevant_matches(expertise_type, match_history)
        if not relevant_matches:
            return 0.0
            
        # Only consider correct predictions for confidence 'weight' (quality of success)
        correct_matches = [m for m in relevant_matches if m.get('is_correct', False)]
        if not correct_matches:
            return 0.0
            
        total_conf = sum(m.get('confidence', 0.5) for m in correct_matches)
        return total_conf / len(correct_matches)

    @staticmethod
    def _calculate_recency_bonus(lora, expertise_type: str, recent_history: List[Dict]) -> float:
        """
        Success rate in the last N matches.
        """
        if not recent_history:
            return 0.0
        return AdvancedCategorization._get_success_rate(lora, expertise_type, recent_history)

    @staticmethod
    def _calculate_consistency(lora, expertise_type: str, match_history: List[Dict]) -> float:
        """
        Inverse variance of success or win streaks.
        Simple logic: Streak length / 10 (capped at 1.0)
        """
        relevant_matches = AdvancedCategorization._filter_relevant_matches(expertise_type, match_history)
        if not relevant_matches:
            return 0.0
            
        current_streak = 0
        max_streak = 0
        for m in relevant_matches:
            if m.get('is_correct'):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return min(1.0, max_streak / 5.0) # 5 wins in a row = 1.0 consistency score for this simple model

    @staticmethod
    def _filter_relevant_matches(expertise_type: str, match_history: List[Dict]) -> List[Dict]:
        """
        Filters match history based on expertise criteria.
        """
        filtered = []
        for m in match_history:
            match_info = m.get('match_info', {})
            is_relevant = False
            
            if expertise_type == "DERBY_MASTER":
                # Assuming 'is_derby' or similar flag, or high team ranks
                # Placeholder logic: both teams match "big team" list (needs external list ideally)
                is_relevant = match_info.get('is_derby', False) 
            elif expertise_type == "HYPE_SURFER":
                # Check hype score
                is_relevant = match_info.get('hype_score', 0) > 0.8
            elif expertise_type == "ODDS_HUNTER":
                # Check if high odds were predicted correctly? Or just match had high variance?
                # For now, let's say matches where underdog won? 
                # Simplification: odd > 2.5
                 is_relevant = match_info.get('odds_winner', 1.0) > 2.5
            elif expertise_type == "HOME_DEFENDER":
                is_relevant = match_info.get('home_goals', 0) > match_info.get('away_goals', 0)
            elif expertise_type == "AWAY_WARRIOR":
                is_relevant = match_info.get('away_goals', 0) > match_info.get('home_goals', 0)
            elif expertise_type == "CONSISTENCY_GOD":
                is_relevant = True # All matches count for consistency generally
            
            if is_relevant:
                filtered.append(m)
        return filtered

    @staticmethod
    def update_lora_expertise(lora, current_match: Dict, is_correct: bool):
        """
        Main entry point to update a LoRA's expertise weights.
        """
        # Construct match record
        match_record = {
            'match_info': current_match,
            'is_correct': is_correct,
            'confidence': 0.7 # Default confidence if not passed
        }
        
        # Get history from LoRA if available
        history = getattr(lora, 'match_history', [])
        if not isinstance(history, list):
            history = []
            
        # Combine
        effective_history = history + [match_record]
        
        weights = AdvancedCategorization.calculate_expertise_weights(lora, effective_history)
        
        # Store in LoRA object
        if not hasattr(lora, 'expertise_weights'):
            lora.expertise_weights = {}
            
        lora.expertise_weights = weights
        
        # Backward Compatibility: Set dominant as specialization
        dominant = max(weights, key=weights.get)
        lora.specialization = dominant
        
        return weights

    @staticmethod
    def get_dominant_expertise(lora) -> str:
        """Returns the expertise with the highest weight."""
        if not hasattr(lora, 'expertise_weights') or not lora.expertise_weights:
            return "ROOKIE"
        
        return max(lora.expertise_weights, key=lora.expertise_weights.get)
