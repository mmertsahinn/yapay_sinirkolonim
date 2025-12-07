"""
🔗 INTEGRATION WRAPPER - Backward Compatibility
================================================

Mevcut sistemlerle uyumluluk için wrapper'lar
"""

from typing import List, Optional, Any
from .lora_adapter import LoRAAdapter
from .evolvable_lora_adapter import EvolvableLoRAAdapter, create_evolvable_from_base
from .social_network import SocialNetwork
from .advanced_social_network import AdvancedSocialNetwork
from .collective_intelligence import CollectiveIntelligence
from .deep_knowledge_transfer import DeepKnowledgeTransfer


def upgrade_lora_to_evolvable(lora: LoRAAdapter, enable_neuroevolution: bool = True) -> EvolvableLoRAAdapter:
    """
    Mevcut LoRAAdapter'ı EvolvableLoRAAdapter'a yükselt
    
    Args:
        lora: Mevcut LoRA instance
        enable_neuroevolution: Nöroevrim etkin mi?
        
    Returns:
        EvolvableLoRAAdapter instance
    """
    return create_evolvable_from_base(lora, enable_neuroevolution)


def upgrade_social_network_to_advanced(social_network: SocialNetwork) -> AdvancedSocialNetwork:
    """
    Mevcut SocialNetwork'ü AdvancedSocialNetwork'e yükselt
    
    Args:
        social_network: Mevcut SocialNetwork instance
        
    Returns:
        AdvancedSocialNetwork instance (bonds transfer edilir)
    """
    advanced = AdvancedSocialNetwork()
    
    # Mevcut bonds'ları transfer et
    if hasattr(social_network, 'bonds'):
        advanced.bonds = social_network.bonds.copy()
    
    if hasattr(social_network, 'mentorships'):
        advanced.mentorships = social_network.mentorships.copy()
    
    print(f"✅ Upgraded SocialNetwork to AdvancedSocialNetwork")
    print(f"   Transferred {len(advanced.bonds)} bonds")
    
    return advanced


def gradual_migration_wrapper(population: List[Any],
                             old_social_network: SocialNetwork,
                             migration_ratio: float = 0.1) -> tuple:
    """
    Aşamalı geçiş wrapper'ı
    
    Popülasyonun bir kısmını evolvable'a çevirir
    
    Args:
        population: LoRA popülasyonu
        old_social_network: Eski sosyal ağ
        migration_ratio: Kaç oranı evolvable'a çevrilecek (0-1)
        
    Returns:
        (evolved_population, advanced_social_network)
    """
    num_to_evolve = int(len(population) * migration_ratio)
    
    evolved_population = []
    evolved_indices = set()
    
    # En iyi performans gösterenlerden başla
    if len(population) > 0 and hasattr(population[0], 'fitness_history'):
        # Fitness'e göre sırala
        population_with_fitness = [
            (lora, sum(lora.fitness_history[-20:]) / len(lora.fitness_history[-20:]) if len(lora.fitness_history) >= 20 else 0.5)
            for lora in population
        ]
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Top performers'ı evolve et
        for i, (lora, fitness) in enumerate(population_with_fitness[:num_to_evolve]):
            if isinstance(lora, EvolvableLoRAAdapter):
                evolved_population.append(lora)
            else:
                evolved = upgrade_lora_to_evolvable(lora)
                evolved_population.append(evolved)
                evolved_indices.add(i)
    else:
        # Random selection
        import random
        indices = random.sample(range(len(population)), num_to_evolve)
        for i, lora in enumerate(population):
            if i in indices:
                if isinstance(lora, EvolvableLoRAAdapter):
                    evolved_population.append(lora)
                else:
                    evolved = upgrade_lora_to_evolvable(lora)
                    evolved_population.append(evolved)
                    evolved_indices.add(i)
            else:
                evolved_population.append(lora)
    
    # Social network'i upgrade et
    advanced_social_network = upgrade_social_network_to_advanced(old_social_network)
    
    print(f"✅ Gradual migration completed")
    print(f"   Evolved {num_to_evolve}/{len(population)} LoRAs ({migration_ratio*100:.1f}%)")
    
    return evolved_population, advanced_social_network


