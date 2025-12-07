import numpy as np
from typing import Dict, List, Any
import random

class AdaptiveNature:
    """
    Learning Nature System.
    Replaces static thresholds with an adaptive system that learns from colony health.
    
    Ref: 3_ANA_SISTEM_GELISTIRME_PLANI.md
    """
    
    def __init__(self):
        # Attributes for serialization/versioning
        self.nature_version = 1
        self.evolution_history = []
        self.lora_immunity = {}
        self.nature_memory = {}

        # Nature State
        self.state = {
            'anger': 0.1,    # Determines probability of disasters
            'chaos': 0.1,    # Determines magnitude of variations
            'health': 1.0    # Overall system health (inverse of anger usually)
        }
        
        # Learned weights for reactions (Action space)
        # Actions: 'mercy', 'minor_disaster', 'major_disaster', 'resource_boom'
        self.action_weights = {
            'mercy': 1.0,
            'minor_disaster': 0.5,
            'major_disaster': 0.1,
            'resource_boom': 0.5
        }
        
        self.learning_rate = 0.05
        self.history_window = [] # Store recent actions and their effects

    def assess_colony_health(self, population: List[Any], avg_success_rate: float) -> float:
        """
        Calculates the colony health based on population metrics.
        Returns health score (0.0 - 1.0).
        """
        if not population:
            return 0.0
            
        # 1. Diversity (std dev of fitness)
        fitnesses = [getattr(lora, 'fitness', 0.5) for lora in population]
        diversity = np.std(fitnesses) if fitnesses else 0.0
        
        # 2. Success (avg success rate)
        success = avg_success_rate
        
        # 3. Population size check (not too small, not too large)
        pop_size = len(population)
        optimal_size = 50 # Example target
        size_score = 1.0 - (abs(pop_size - optimal_size) / optimal_size)
        size_score = max(0.0, size_score)
        
        # Health Formula
        health = (0.4 * success) + (0.4 * size_score) + (0.2 * diversity)
        self.state['health'] = health
        return health

    def update_nature_state(self, health: float):
        """
        Updates Anger and Chaos based on Health.
        Low health -> High Anger (Nature tries to fix it via selection/pressure).
        Stagnation -> High Chaos (Nature tries to shake things up).
        """
        # Anger: Inverse of health
        target_anger = 1.0 - health
        self.state['anger'] += self.learning_rate * (target_anger - self.state['anger'])
        
        # Chaos: If health is stable/stagnant or high, reduce chaos? 
        # Actually in this system:
        # High Health -> Low Chaos (Stability)
        # Low Health -> High Chaos (Desperate measures)
        self.state['chaos'] = self.state['anger'] * 0.8 + (random.random() * 0.2)
        
    def decide_nature_action(self) -> str:
        """
        Decides on an action based on current state and learned weights.
        """
        # Probabilities influenced by Anger
        anger = self.state['anger']
        
        probs = {}
        probs['mercy'] = self.action_weights['mercy'] * (1.0 - anger)
        probs['minor_disaster'] = self.action_weights['minor_disaster'] * anger
        probs['major_disaster'] = self.action_weights['major_disaster'] * (anger ** 2) # Only at very high anger
        probs['resource_boom'] = self.action_weights['resource_boom'] * (1.0 - anger)
        
        # Normalize
        total = sum(probs.values())
        if total == 0:
            return 'mercy'
            
        keys = list(probs.keys())
        values = [p/total for p in probs.values()]
        
        chosen_action = np.random.choice(keys, p=values)
        return chosen_action

    def learn_from_result(self, action: str, old_health: float, new_health: float):
        """
        Reinforcement Learning step for nature.
        """
        # Reward: Did the action improve health?
        reward = 0.0
        if new_health > old_health:
            reward = 0.5
        
        if new_health < 0.3:
            reward -= 0.5
            
        current_weight = self.action_weights.get(action, 0.5)
        new_weight = current_weight + (self.learning_rate * reward * 10.0)
        new_weight = max(0.01, min(2.0, new_weight))
        
        self.action_weights[action] = new_weight
        self.history_window.append({'action': action, 'reward': reward, 'health': new_health})

    def calculate_adaptive_severity(self, population: List[Any], event_type: str, base_severity: float) -> float:
        """
        Calculates severity based on Nature's ANGER and CHAOS.
        """
        anger = self.state['anger']
        chaos = self.state['chaos']
        
        # Base modifier from Anger
        anger_modifier = 1.0 + (anger * 0.5)
        
        # Chaos adds randomness
        chaos_noise = (random.random() - 0.5) * chaos * 0.2
        
        adjusted_severity = base_severity * anger_modifier + chaos_noise
        
        return max(0.1, min(0.95, adjusted_severity))

    def evolve_nature(self, population: List[Any], match_idx: int) -> str:
        """
        Periodic evolution of Nature itself.
        """
        # Simple implementation: Report current state
        report = self.get_nature_report()
        return f"🌍 DOĞA EVRİMİ (Maç {match_idx}): {report}"

    def get_nature_report(self) -> str:
        return (f"Nature State: Health={self.state['health']:.2f}, "
                f"Anger={self.state['anger']:.2f}, Chaos={self.state['chaos']:.2f}")

    def lora_survived_event(self, lora: Any, event_type: str, survived_by: str = "luck"):
        """
        Record that a LoRA survived a nature event.
        Logic: If many survive by armor, Nature might evolve to pierce armor.
        """
        # Record survival
        self.history_window.append({
            'type': 'survival',
            'lora': lora.id if hasattr(lora, 'id') else 'unknown',
            'event': event_type,
            'method': survived_by
        })
        
        # If survived by armor, slightly increase anger?
        if survived_by == "armor":
            self.state['anger'] = min(1.0, self.state['anger'] + 0.001)

    def observe_lora_immunity(self, survivors: List[Any], event_type: str, death_rate: float) -> bool:
        """
        Observes if population is immune to a specific event.
        Returns True if immunity is detected (death rate low).
        """
        # If death rate is very low for a disaster < 20%
        if death_rate < 0.20:
             # Immunity detected!
             # Nature gets angry/creative
             self.state['anger'] = min(1.0, self.state['anger'] + 0.10)
             self.state['chaos'] = min(1.0, self.state['chaos'] + 0.05)
             return True
             
        return False

