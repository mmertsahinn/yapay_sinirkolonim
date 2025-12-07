import unittest
from advanced_categorization import AdvancedCategorization

class MockLoRA:
    def __init__(self, id):
        self.id = id
        self.expertise_weights = {}

class TestAdvancedCategorization(unittest.TestCase):
    
    def setUp(self):
        self.lora = MockLoRA("test_lora_1")
        # Create a mock match history where the LoRA is good at Derbies but bad at others
        self.match_history = []
        
        # 10 Derby matches (8 wins)
        for i in range(10):
            self.match_history.append({
                'match_info': {'is_derby': True, 'hype_score': 0.5, 'odds_winner': 1.5},
                'is_correct': i < 8, # 80% success
                'confidence': 0.8
            })
            
        # 10 Random matches (2 wins)
        for i in range(10):
            self.match_history.append({
                'match_info': {'is_derby': False, 'hype_score': 0.2, 'odds_winner': 1.2},
                'is_correct': i < 2, # 20% success
                'confidence': 0.4
            })

    def test_calculate_weights(self):
        weights = AdvancedCategorization.calculate_expertise_weights(self.lora, self.match_history)
        
        print("\nCalculated Weights:")
        for k, v in weights.items():
            print(f"{k}: {v:.4f}")
            
        # Derby Master should be the highest
        dominant = max(weights, key=weights.get)
        self.assertEqual(dominant, "DERBY_MASTER")
        
        # Verify sum is approx 1.0 (softmax property)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

    def test_update_lora(self):
        AdvancedCategorization.update_lora_expertise(self.lora, self.match_history)
        self.assertTrue(hasattr(self.lora, 'expertise_weights'))
        self.assertEqual(AdvancedCategorization.get_dominant_expertise(self.lora), "DERBY_MASTER")

if __name__ == '__main__':
    unittest.main()
