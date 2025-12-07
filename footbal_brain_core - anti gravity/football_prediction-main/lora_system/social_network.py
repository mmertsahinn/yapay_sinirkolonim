import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math

class SocialNetwork:
    """
    Fluid Social LoRA System.
    Manages social bonds, interactions, and community formation among LoRA models.
    
    Ref: 3_ANA_SISTEM_GELISTIRME_PLANI.md
    """
    
    # Interaction Weights
    ALPHA = 0.30 # Prediction Similarity
    BETA  = 0.25 # Success Correlation
    GAMMA = 0.25 # Learning Transfer (Mentorship)
    DELTA = 0.20 # Complementary Score
    
    DECAY_RATE = 0.10 # Bond decay rate per interaction cycle
    
    def __init__(self):
        # Edge list: (id1, id2) -> bond_strength (0.0 to 1.0)
        # Store canonical keys: tuple(sorted((id1, id2)))
        self.bonds: Dict[Tuple[str, str], float] = {}
        self.mentorships: Dict[str, str] = {} # student_id -> mentor_id
        
    def update_social_bond(self, lora_i: Any, lora_j: Any, match_result: Dict) -> float:
        """
        Updates the bond strength between two LoRAs based on their recent interaction.
        Returns the new bond strength.
        """
        if lora_i.id == lora_j.id:
            return 0.0
            
        key = tuple(sorted((lora_i.id, lora_j.id)))
        current_bond = self.bonds.get(key, 0.1) # Start with weak bond
        
        # 1. Prediction Similarity
        pred_sim = self._calculate_prediction_similarity(lora_i, lora_j)
        
        # 2. Success Correlation
        success_corr = self._calculate_success_correlation(lora_i, lora_j)
        
        # 3. Learning Transfer (Mentorship)
        learning_transfer = self._calculate_mentorship_potential(lora_i, lora_j)
        
        # 4. Complementary Score
        comp_score = self._calculate_complementary_score(lora_i, lora_j)
        
        # Weighted Interaction Score
        interaction_score = (self.ALPHA * pred_sim + 
                             self.BETA * success_corr + 
                             self.GAMMA * learning_transfer + 
                             self.DELTA * comp_score)
        
        # Update Bond (Exponential Moving Average)
        new_bond = (1.0 - self.DECAY_RATE) * current_bond + (self.DECAY_RATE * interaction_score)
        new_bond = max(0.0, min(1.0, new_bond)) # Clamp
        
        self.bonds[key] = new_bond
        return new_bond

    def get_bond_strength(self, id1: str, id2: str) -> float:
        key = tuple(sorted((id1, id2)))
        return self.bonds.get(key, 0.0)

    def _calculate_prediction_similarity(self, lora_i, lora_j) -> float:
        """
        Cosine similarity or simple agreement check of last predictions.
        Assumption: lora objects have 'last_prediction' dict with 'home_score', 'away_score' or 'outcome'.
        """
        if not hasattr(lora_i, 'last_prediction') or not hasattr(lora_j, 'last_prediction'):
            return 0.5
            
        p1 = lora_i.last_prediction
        p2 = lora_j.last_prediction
        
        if not p1 or not p2:
            return 0.5

        # Check outcome agreement (Win/Draw/Loss)
        # Using simple equality for now. 1.0 if same outcome, 0.0 if not.
        return 1.0 if p1.get('outcome') == p2.get('outcome') else 0.0

    def _calculate_success_correlation(self, lora_i, lora_j) -> float:
        """
        1.0 if both correct OR both wrong (shared fate). 
        0.0 if one correct, one wrong (conflict).
        """
        correct_i = getattr(lora_i, 'was_correct', False)
        correct_j = getattr(lora_j, 'was_correct', False)
        
        return 1.0 if (correct_i == correct_j) else 0.0

    def _calculate_mentorship_potential(self, lora_i, lora_j) -> float:
        """
        Calculates if a mentorship relationship is beneficial.
        High fitness diff = high potential.
        """
        fitness_i = getattr(lora_i, 'fitness', 0.5)
        fitness_j = getattr(lora_j, 'fitness', 0.5)
        
        diff = abs(fitness_i - fitness_j)
        
        # If difference is significant (>0.2), potential is high
        if diff > 0.2:
            # Register mentorship (higher fitness becomes mentor)
            mentor = lora_i if fitness_i > fitness_j else lora_j
            student = lora_j if fitness_i > fitness_j else lora_i
            self.mentorships[student.id] = mentor.id
            return 1.0
            
        return 0.0

    def _calculate_complementary_score(self, lora_i, lora_j) -> float:
        """
        Do they have different expertise? (Diversity is good).
        """
        # Assuming we can access expertise from AdvancedCategorization or similar attribute
        # If logic not available, check specialized attribute
        spec_i = getattr(lora_i, 'specialization', 'None')
        spec_j = getattr(lora_j, 'specialization', 'None')
        
        if spec_i != 'None' and spec_j != 'None' and spec_i != spec_j:
            return 1.0 # Complementary
            
        return 0.5 # Neutral

    def get_social_cluster(self, lora_id: str, threshold: float = 0.5) -> List[str]:
        """Returns list of IDs strongly bonded to the given LoRA."""
        friends = []
        for key, strength in self.bonds.items():
            if strength > threshold:
                if key[0] == lora_id:
                    friends.append(key[1])
                elif key[1] == lora_id:
                    friends.append(key[0])
        return friends
    
    def apply_social_parameter_drift(self, population: List[Any], drift_strength: float = 0.05):
        """
        🌊 SOSYAL PARAMETRE SÜRÜKLENMESİ
        
        Güçlü sosyal bağları olan LoRA'lar parametre uzayında 
        birbirlerine yakınlaşırlar!
        
        Args:
            population: LoRA listesi
            drift_strength: Sürüklenme gücü (default: 0.05 = %5)
        """
        import torch
        
        drift_count = 0
        
        # Her güçlü bağ için
        for (id1, id2), bond_strength in self.bonds.items():
            if bond_strength < 0.7:  # Sadece güçlü bağlar
                continue
            
            # LoRA'ları bul
            lora_i = next((l for l in population if l.id == id1), None)
            lora_j = next((l for l in population if l.id == id2), None)
            
            if not lora_i or not lora_j:
                continue
            
            # Parametre drift
            params_i = lora_i.get_all_lora_params()
            params_j = lora_j.get_all_lora_params()
            
            # Her layer için drift uygula
            for layer in ['fc1', 'fc2', 'fc3']:
                for matrix in ['lora_A', 'lora_B']:
                    tensor_i = params_i[layer][matrix]
                    tensor_j = params_j[layer][matrix]
                    
                    # i → j'ye doğru drift
                    # Yeni_i = i + α × bond × (j - i)
                    delta = drift_strength * bond_strength * (tensor_j - tensor_i)
                    params_i[layer][matrix] = tensor_i + delta
                    
                    # j → i'ye doğru drift (karşılıklı!)
                    delta = drift_strength * bond_strength * (tensor_i - tensor_j)
                    params_j[layer][matrix] = tensor_j + delta
            
            # Parametreleri geri yaz
            lora_i.set_all_lora_params(params_i)
            lora_j.set_all_lora_params(params_j)
            
            drift_count += 1
        
        if drift_count > 0:
            print(f"   🌊 Sosyal drift: {drift_count} bağ için parametre yakınlaşması")
