import unittest
from social_network import SocialNetwork

class MockLoRA:
    def __init__(self, id, fitness=0.5, specialization="None"):
        self.id = id
        self.fitness = fitness
        self.specialization = specialization
        self.last_prediction = {'outcome': 'WIN'}
        self.was_correct = True

class TestSocialNetwork(unittest.TestCase):
    
    def setUp(self):
        self.sn = SocialNetwork()
        self.lora1 = MockLoRA("lora_1", fitness=0.8, specialization="HYPE_MASTER")
        self.lora2 = MockLoRA("lora_2", fitness=0.4, specialization="ODDS_HUNTER")
        self.lora3 = MockLoRA("lora_3", fitness=0.4, specialization="HYPE_MASTER")

    def test_bond_update(self):
        # Initial bond should be low or default
        initial_bond = self.sn.get_bond_strength("lora_1", "lora_2")
        
        # Simulate interaction (Good interaction: Diff specialization, mentorship potential)
        # lora1 (0.8) should mentor lora2 (0.4) -> High interaction score
        new_bond = self.sn.update_social_bond(self.lora1, self.lora2, {})
        
        print(f"Bond updated from {initial_bond} to {new_bond}")
        self.assertGreater(new_bond, initial_bond)
        
        # Verify mentorship registration
        self.assertEqual(self.sn.mentorships.get("lora_2"), "lora_1")

    def test_social_cluster(self):
        # Artificially set bond
        key = tuple(sorted(("lora_1", "lora_3")))
        self.sn.bonds[key] = 0.9
        
        friends = self.sn.get_social_cluster("lora_1", threshold=0.8)
        self.assertIn("lora_3", friends)
        self.assertNotIn("lora_2", friends) # Bond with lora_2 was verified in previous test but instance is fresh here? 
        # Ah, setUp runs fresh for each test. So lora_2 bond is default 0.0 here unless set.

if __name__ == '__main__':
    unittest.main()
