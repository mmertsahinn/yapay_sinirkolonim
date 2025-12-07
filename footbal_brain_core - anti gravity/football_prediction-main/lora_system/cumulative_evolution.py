"""
🧬 CUMULATIVE EVOLUTION - Generational Knowledge Accumulation
===============================================================

Kümülatif evrim: Her nesil önceki nesillerin bilgisini biriktirir.

Gen1: Newton → Discovery A
Gen2: Einstein (Newton'un çocuğu) → Discovery A + B  
Gen3: Darwin (Einstein'ın çocuğu) → Discovery A + B + C

Özellikler:
✅ Generational knowledge accumulation
✅ Neural architecture inheritance
✅ Progressive enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class AncestralKnowledge:
    """Ataların bilgisi"""
    ancestor_id: str
    ancestor_name: str
    generation: int
    knowledge_embedding: torch.Tensor
    discoveries: List[str]  # Yaptığı keşifler
    architecture: Dict  # Nöron mimarisi


class GenerationalMemory:
    """
    Nesiller arası hafıza
    
    Her neslin bilgisi saklanır ve aktarılır
    """
    
    def __init__(self, max_generations: int = 10):
        """
        Args:
            max_generations: Maksimum saklanacak nesil sayısı
        """
        self.max_generations = max_generations
        self.generational_knowledge: Dict[int, List[AncestralKnowledge]] = defaultdict(list)
        
        print(f"✅ GenerationalMemory initialized (max_generations={max_generations})")
    
    def store_generation(self, generation: int, lora, discoveries: List[str] = None):
        """
        Bir neslin bilgisini sakla
        
        Args:
            generation: Nesil numarası
            lora: LoRA instance
            discoveries: Yapılan keşifler
        """
        if generation >= self.max_generations:
            return  # Çok eski nesiller
        
        discoveries = discoveries or []
        
        # Architecture bilgisi
        architecture = {
            'hidden_dim': getattr(lora, 'hidden_dim', 128),
            'num_layers': 3,  # fc1, fc2, fc3
            'total_neurons': getattr(lora, 'hidden_dim', 128) + 64 + 3
        }
        
        # Knowledge embedding (basit: parametrelerden)
        try:
            params = lora.get_all_lora_params()
            param_vec = []
            for layer in ['fc1', 'fc2', 'fc3']:
                for matrix in ['lora_A', 'lora_B']:
                    param_vec.append(params[layer][matrix].flatten())
            knowledge_emb = torch.cat(param_vec)[:128]  # Truncate to 128
            if knowledge_emb.shape[0] < 128:
                padding = torch.zeros(128 - knowledge_emb.shape[0])
                knowledge_emb = torch.cat([knowledge_emb, padding])
        except:
            knowledge_emb = torch.zeros(128)
        
        ancestral_knowledge = AncestralKnowledge(
            ancestor_id=lora.id,
            ancestor_name=lora.name,
            generation=generation,
            knowledge_embedding=knowledge_emb,
            discoveries=discoveries,
            architecture=architecture
        )
        
        self.generational_knowledge[generation].append(ancestral_knowledge)
    
    def retrieve_ancestral_knowledge(self, 
                                     current_generation: int,
                                     lora_parents: List[str] = None) -> List[AncestralKnowledge]:
        """
        Ataların bilgisini getir
        
        Args:
            current_generation: Mevcut nesil
            lora_parents: Ebeveyn ID'leri (opsiyonel, spesifik atalar için)
            
        Returns:
            Ancestral knowledge list
        """
        ancestral = []
        
        # Geçmiş nesillerden bilgi al
        for gen in range(max(0, current_generation - self.max_generations), current_generation):
            if gen in self.generational_knowledge:
                # Ebeveyn ID'leri varsa filtrele
                if lora_parents:
                    filtered = [
                        ak for ak in self.generational_knowledge[gen]
                        if ak.ancestor_id in lora_parents
                    ]
                    ancestral.extend(filtered)
                else:
                    ancestral.extend(self.generational_knowledge[gen])
        
        return ancestral


class NeuralArchitectureInheritance:
    """
    Nöron mimarisi mirası
    
    Ebeveynlerin nöron yapısından başlayarak evrilme
    """
    
    def __init__(self):
        print("✅ NeuralArchitectureInheritance initialized")
    
    def inherit_architecture(self, parent, child, inheritance_strength: float = 0.7):
        """
        Mimariyi miras al
        
        Args:
            parent: Ebeveyn LoRA
            child: Çocuk LoRA
            inheritance_strength: Miras gücü (0-1)
        """
        # Hidden dimension inheritance
        if hasattr(parent, 'hidden_dim') and hasattr(child, 'hidden_dim'):
            parent_dim = parent.hidden_dim
            child_dim = child.hidden_dim
            
            # Blend: Child başlangıçta ebeveynin mimarisine benzer
            inherited_dim = int(parent_dim * inheritance_strength + child_dim * (1 - inheritance_strength))
            child.hidden_dim = inherited_dim
        
        # Architecture state inheritance
        if hasattr(parent, 'current_architecture') and hasattr(child, 'current_architecture'):
            parent_arch = parent.current_architecture
            child.current_architecture['layer_dims'][0] = int(
                parent_arch.get('layer_dims', [128])[0] * inheritance_strength +
                child.current_architecture['layer_dims'][0] * (1 - inheritance_strength)
            )
        
        # Neuroevolution state inheritance
        if hasattr(parent, 'neuroevolution_state') and hasattr(child, 'neuroevolution_state'):
            parent_neuro = parent.neuroevolution_state
            if parent_neuro and 'architecture' in parent_neuro:
                # Inherit architecture preferences
                pass  # Implementation can be added
    
    def progressive_enhancement(self, 
                               ancestral_knowledge: List[AncestralKnowledge],
                               current_lora) -> Dict:
        """
        Progressive enhancement: Ataların bilgisiyle güçlendir
        
        Args:
            ancestral_knowledge: Ataların bilgisi
            current_lora: Mevcut LoRA
            
        Returns:
            Enhancement report
        """
        if not ancestral_knowledge:
            return {'status': 'no_ancestors'}
        
        # Average ancestral architecture
        avg_ancestral_dim = np.mean([
            ak.architecture.get('hidden_dim', 128) for ak in ancestral_knowledge
        ])
        
        # Current dimension
        current_dim = getattr(current_lora, 'hidden_dim', 128)
        
        # Progressive enhancement: Mevcut nesil ataların ortalamasından daha büyük olabilir
        enhanced_dim = int(max(current_dim, avg_ancestral_dim * 1.1))
        
        # Discoveries accumulation
        all_discoveries = []
        for ak in ancestral_knowledge:
            all_discoveries.extend(ak.discoveries)
        
        enhancement = {
            'ancestral_dim': avg_ancestral_dim,
            'current_dim': current_dim,
            'enhanced_dim': enhanced_dim,
            'inherited_discoveries': list(set(all_discoveries)),
            'ancestors_count': len(ancestral_knowledge)
        }
        
        return enhancement


class CumulativeEvolutionSystem:
    """
    Kümülatif Evrim Sistemi
    
    Tüm nesiller arası bilgi birikimi ve aktarımı
    """
    
    def __init__(self, max_generations: int = 10):
        """
        Args:
            max_generations: Maksimum nesil sayısı
        """
        self.generational_memory = GenerationalMemory(max_generations)
        self.architecture_inheritance = NeuralArchitectureInheritance()
        
        # Tracking
        self.evolution_trees: Dict[str, List[str]] = {}  # LoRA ID → Ancestor IDs
        
        print("="*80)
        print("🧬 CUMULATIVE EVOLUTION SYSTEM INITIALIZED")
        print("="*80)
    
    def enable_cumulative_evolution(self, parent, child):
        """
        Kümülatif evrimi etkinleştir
        
        Çocuk ebeveynin tüm keşiflerini ve mimarisini miras alır
        
        Args:
            parent: Ebeveyn LoRA
            child: Çocuk LoRA
        """
        # 1. Mimari mirası
        self.architecture_inheritance.inherit_architecture(parent, child)
        
        # 2. Keşif mirası (mevcut collective_intelligence'dan)
        if hasattr(parent, 'adopted_discoveries'):
            if not hasattr(child, 'adopted_discoveries'):
                child.adopted_discoveries = []
            
            for disc in parent.adopted_discoveries:
                child.adopted_discoveries.append({
                    'discovery': disc['discovery'],
                    'from': disc.get('from', parent.name),
                    'match_idx': disc.get('match_idx', 0),
                    'adoption_rate': disc.get('adoption_rate', 0.5) * 0.7,  # Zayıflamış
                    'blend_alpha': disc.get('blend_alpha', 0.1) * 0.7,
                    'inherited_from_parent': parent.name,
                    'generation': getattr(parent, 'generation', 1)
                })
        
        # 3. Knowledge accumulation
        parent_discoveries = []
        if hasattr(parent, 'adopted_discoveries'):
            parent_discoveries = [d.get('discovery', '') for d in parent.adopted_discoveries]
        
        # 4. Generation tracking
        child.generation = max(getattr(parent, 'generation', 1), getattr(child, 'generation', 1)) + 1
        
        # Evolution tree
        if parent.id not in self.evolution_trees:
            self.evolution_trees[parent.id] = []
        
        # Get parent's ancestors
        parent_ancestors = self.evolution_trees.get(parent.id, [])
        child_ancestors = [parent.id] + parent_ancestors
        self.evolution_trees[child.id] = child_ancestors
        
        print(f"   🧬 {child.name} → Inherited from {parent.name}")
        print(f"      Generation: {child.generation}, Ancestors: {len(child_ancestors)}")
        
        # 5. Store parent in generational memory
        parent_gen = getattr(parent, 'generation', 1)
        self.generational_memory.store_generation(
            parent_gen,
            parent,
            parent_discoveries
        )
        
        # 6. Progressive enhancement
        ancestral_knowledge = self.generational_memory.retrieve_ancestral_knowledge(
            child.generation,
            child_ancestors
        )
        
        enhancement = self.architecture_inheritance.progressive_enhancement(
            ancestral_knowledge,
            child
        )
        
        # Apply enhancement if beneficial
        if enhancement.get('enhanced_dim', 0) > getattr(child, 'hidden_dim', 128):
            # Şimdilik sadece log
            print(f"      Progressive enhancement: {getattr(child, 'hidden_dim', 128)} → {enhancement['enhanced_dim']}")
    
    def get_ancestral_lineage(self, lora_id: str) -> List[str]:
        """
        Ataların soyunu getir
        
        Args:
            lora_id: LoRA ID
            
        Returns:
            Ancestor IDs list (direct parent → oldest ancestor)
        """
        return self.evolution_trees.get(lora_id, [])
    
    def get_generational_statistics(self) -> Dict:
        """Nesil istatistikleri"""
        stats = {
            'evolution_trees': len(self.evolution_trees),
            'generational_memory': {
                gen: len(loras) 
                for gen, loras in self.generational_memory.generational_knowledge.items()
            },
            'max_generation': max(self.evolution_trees.values(), key=len, default=[]) if self.evolution_trees else []
        }
        
        return stats


# Global instance
_global_cumulative_evolution = None


def get_cumulative_evolution_system(max_generations: int = 10) -> CumulativeEvolutionSystem:
    """Global cumulative evolution system instance"""
    global _global_cumulative_evolution
    if _global_cumulative_evolution is None:
        _global_cumulative_evolution = CumulativeEvolutionSystem(max_generations)
    return _global_cumulative_evolution


