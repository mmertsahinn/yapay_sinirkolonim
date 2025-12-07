import unittest
from adaptive_nature import AdaptiveNature

class MockLoRA:
    def __init__(self, fitness):
        self.fitness = fitness

class TestAdaptiveNature(unittest.TestCase):
    
    def setUp(self):
        self.nature = AdaptiveNature()
        # Create a mock population
        self.population = [MockLoRA(0.5 + (i*0.05)) for i in range(10)] # Fitnesses 0.5 to 0.95

    def test_assess_health(self):
        health = self.nature.assess_colony_health(self.population, avg_success_rate=0.7)
        print(f"\nInitial Health: {health:.4f}")
        self.assertGreater(health, 0.0)
        self.assertLessEqual(health, 1.0)
        
        # Verify state update
        self.nature.update_nature_state(health)
        print(self.nature.get_nature_report())
        self.assertAlmostEqual(self.nature.state['health'], health)

    def test_decision_making(self):
        # High anger scenario
        self.nature.state['anger'] = 0.9
        
        actions = []
        for _ in range(100):
            actions.append(self.nature.decide_nature_action())
            
        print(f"\nActions in High Anger: {actions[:10]}...")
        # Should see more disasters
        disaster_count = actions.count('minor_disaster') + actions.count('major_disaster')
        self.assertGreater(disaster_count, 10) # At least some disasters

    def test_learning(self):
        # Test if weights update
        action = 'mercy'
        old_weight = self.nature.action_weights[action]
        
        # Simulate improvement
        self.nature.learn_from_result(action, old_health=0.5, new_health=0.6)
        
        new_weight = self.nature.action_weights[action]
        print(f"\nWeight for {action}: {old_weight} -> {new_weight}")
        self.assertGreater(new_weight, old_weight) # Should increase due to positive reward

if __name__ == '__main__':
    unittest.main()
