import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lora_system.lazarus_potential import LazarusPotential

# Mock LoRA class
class MockLoRA:
    def __init__(self, name, birth_match=0):
        self.name = name
        self.birth_match = birth_match
        self._current_match = 100
        self.fitness_history = [0.8] * 100  # High fitness for low entropy

# Mock KFAC Fisher module
class MockKFAC:
    @staticmethod
    def compute_fisher_kfac(lora):
        # Simulate different scenarios
        if lora.name == "Legendary_LoRA":
            return {'fisher_logdet': 2640.0, 'fisher_det': 0.0} # 2640 / 48 = 55.0 (Legend)
        elif lora.name == "Average_LoRA":
            return {'fisher_logdet': 2160.0, 'fisher_det': 0.0} # 2160 / 48 = 45.0 (Medium)
        elif lora.name == "Weak_LoRA":
            return {'fisher_logdet': 1440.0, 'fisher_det': 0.0} # 1440 / 48 = 30.0 (Low)
        return {}

# Inject mock into sys.modules
import types
mock_kfac_module = types.ModuleType("lora_system.kfac_fisher")
mock_kfac_module.kfac_fisher = MockKFAC()
sys.modules["lora_system.kfac_fisher"] = mock_kfac_module

def test_lazarus():
    lazarus = LazarusPotential(beta=0.5)
    
    scenarios = [
        MockLoRA("Legendary_LoRA"),
        MockLoRA("Average_LoRA"),
        MockLoRA("Weak_LoRA")
    ]
    
    print("\n🧪 TESTING LAZARUS LOGIC...\n")
    
    for lora in scenarios:
        print(f"--- Testing {lora.name} ---")
        result = lazarus.calculate_lazarus_lambda(lora)
        print(f"Result: {result}")
        print("-" * 30)

if __name__ == "__main__":
    test_lazarus()
