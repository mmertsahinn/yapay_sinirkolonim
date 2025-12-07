"""
🧠 COLLECTIVE INTELLIGENCE BRAIN - Master Orchestrator
========================================================

Master koordinatör: Tüm sistemleri yöneten merkezi beyin.

Özellikler:
✅ Master orchestrator
✅ Knowledge graph
✅ Adaptive routing (bilgi hangi LoRA'ya nasıl gidecek?)
✅ Emergent intelligence (kolektif sistemden ortaya çıkan zeka)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import random

# Import all systems
from .deep_knowledge_transfer import DeepKnowledgeTransfer, Domain, get_deep_knowledge_transfer
from .advanced_social_network import AdvancedSocialNetwork, get_advanced_social_network
from .cumulative_evolution import CumulativeEvolutionSystem, get_cumulative_evolution_system
from .enhanced_temperament_processing import EnhancedTemperamentProcessing, get_enhanced_temperament_processing
from .neuroevolution_engine import NeuroevolutionEngine, get_neuroevolution_engine
from .thinking_patterns import ThinkingPattern


class KnowledgeGraph:
    """
    Knowledge Graph: Tüm keşifler, ilişkiler, domain'ler
    
    Graph structure:
    - Nodes: LoRAs, Discoveries, Domains
    - Edges: Relationships, Knowledge transfers
    """
    
    def __init__(self):
        # Graph structure
        self.nodes: Dict[str, Dict] = {}  # node_id → node_data
        self.edges: Dict[Tuple[str, str], Dict] = {}  # (source, target) → edge_data
        
        # Indexes
        self.lora_nodes: Dict[str, str] = {}  # lora_id → node_id
        self.discovery_nodes: Dict[str, str] = {}  # discovery_pattern → node_id
        self.domain_nodes: Dict[str, str] = {}  # domain → node_id
        
        print("✅ KnowledgeGraph initialized")
    
    def add_lora_node(self, lora_id: str, lora_data: Dict):
        """LoRA node ekle"""
        node_id = f"lora_{lora_id}"
        self.nodes[node_id] = {
            'type': 'lora',
            'id': lora_id,
            'data': lora_data
        }
        self.lora_nodes[lora_id] = node_id
    
    def add_discovery_node(self, discovery_pattern: str, discovery_data: Dict):
        """Discovery node ekle"""
        node_id = f"discovery_{discovery_pattern}"
        self.nodes[node_id] = {
            'type': 'discovery',
            'pattern': discovery_pattern,
            'data': discovery_data
        }
        self.discovery_nodes[discovery_pattern] = node_id
    
    def add_domain_node(self, domain: Domain):
        """Domain node ekle"""
        node_id = f"domain_{domain.value}"
        self.nodes[node_id] = {
            'type': 'domain',
            'domain': domain.value
        }
        self.domain_nodes[domain.value] = node_id
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str, edge_data: Dict = None):
        """Edge ekle"""
        edge_key = (source_id, target_id)
        self.edges[edge_key] = {
            'type': edge_type,
            'data': edge_data or {}
        }
    
    def get_connections(self, node_id: str, edge_type: str = None) -> List[str]:
        """Bir node'un bağlantılarını getir"""
        connections = []
        for (source, target), edge in self.edges.items():
            if source == node_id:
                if edge_type is None or edge['type'] == edge_type:
                    connections.append(target)
            elif target == node_id:
                if edge_type is None or edge['type'] == edge_type:
                    connections.append(source)
        return connections


class AdaptiveRouter(nn.Module):
    """
    Adaptive Router: Bilgi hangi LoRA'ya nasıl gidecek?
    
    Router öğrenir: Hangi bilgi hangi LoRA'ya gitsin?
    """
    
    def __init__(self, embed_dim: int = 128, num_domains: int = len(Domain)):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Router network: knowledge + lora_state → routing_probability
        self.router = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Domain-specific routing
        self.domain_router = nn.ModuleDict({
            domain.value: nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            for domain in list(Domain)[:num_domains]
        })
        
        print(f"✅ AdaptiveRouter initialized")
    
    def route_knowledge(self, 
                       knowledge_emb: torch.Tensor,
                       lora_emb: torch.Tensor,
                       lora_domain: Domain = None) -> float:
        """
        Bilgiyi LoRA'ya yönlendir
        
        Args:
            knowledge_emb: Knowledge embedding
            lora_emb: LoRA embedding
            lora_domain: LoRA'nın domain'i (opsiyonel)
            
        Returns:
            Routing probability (0-1)
        """
        # Combined embedding
        combined = torch.cat([knowledge_emb, lora_emb], dim=-1)
        
        # Base routing probability
        routing_prob = self.router(combined).item()
        
        # Domain-specific adjustment
        if lora_domain and lora_domain.value in self.domain_router:
            domain_adj = self.domain_router[lora_domain.value](knowledge_emb).item()
            routing_prob = (routing_prob + domain_adj) / 2
        
        return routing_prob


class EmergentIntelligence:
    """
    Emergent Intelligence: Kolektif sistemden ortaya çıkan zeka
    
    Bireysel LoRA'ların toplamından daha fazlası
    """
    
    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        
        # Collective memory
        self.collective_memory = []
        
        # Emergent patterns
        self.emergent_patterns = []
        
        print("✅ EmergentIntelligence initialized")
    
    def detect_emergent_patterns(self, population: List, social_network) -> List[Dict]:
        """
        Ortaya çıkan pattern'leri tespit et
        
        Args:
            population: LoRA popülasyonu
            social_network: Social network
            
        Returns:
            Emergent patterns
        """
        patterns = []
        
        # Pattern 1: Collective consensus (çoğunluk aynı şeyi düşünüyor mu?)
        if len(population) >= 10:
            predictions = []
            for lora in population[:10]:
                if hasattr(lora, 'last_prediction') and lora.last_prediction:
                    predictions.append(lora.last_prediction.get('outcome'))
            
            if predictions:
                consensus = max(set(predictions), key=predictions.count)
                consensus_rate = predictions.count(consensus) / len(predictions)
                
                if consensus_rate > 0.7:
                    patterns.append({
                        'type': 'collective_consensus',
                        'consensus': consensus,
                        'rate': consensus_rate
                    })
        
        # Pattern 2: Knowledge cascade (bilgi yayılımı)
        if hasattr(social_network, 'similarity_cache'):
            high_similarity_pairs = [
                (key, data) for key, data in social_network.similarity_cache.items()
                if data.get('enhanced_bond', 0) > 0.8
            ]
            
            if len(high_similarity_pairs) > len(population) * 0.3:
                patterns.append({
                    'type': 'knowledge_cascade',
                    'high_similarity_pairs': len(high_similarity_pairs)
                })
        
        self.emergent_patterns.extend(patterns)
        
        return patterns


class CollectiveIntelligenceBrain:
    """
    🧠 MASTER COLLECTIVE INTELLIGENCE BRAIN
    
    Tüm sistemleri yöneten merkezi koordinatör
    """
    
    def __init__(self, 
                 embed_dim: int = 128,
                 device='cpu',
                 enable_all_systems: bool = True):
        """
        Args:
            embed_dim: Embedding boyutu
            device: Device
            enable_all_systems: Tüm sistemleri etkinleştir mi?
        """
        self.embed_dim = embed_dim
        self.device = device
        self.enable_all_systems = enable_all_systems
        
        # Core systems
        if enable_all_systems:
            self.knowledge_transfer = get_deep_knowledge_transfer(embed_dim, device)
            self.social_network = get_advanced_social_network()
            self.cumulative_evolution = get_cumulative_evolution_system()
            self.temperament_processing = get_enhanced_temperament_processing(embed_dim, device)
            self.neuroevolution_engine = get_neuroevolution_engine()
        
        # Brain components
        self.knowledge_graph = KnowledgeGraph()
        self.adaptive_router = AdaptiveRouter(embed_dim).to(device)
        self.emergent_intelligence = EmergentIntelligence(embed_dim)
        
        # State tracking
        self.cycle_count = 0
        self.statistics = {}
        
        print("="*80)
        print("🧠 COLLECTIVE INTELLIGENCE BRAIN INITIALIZED")
        print("="*80)
        print(f"   Embedding dim: {embed_dim}")
        print(f"   Device: {device}")
        print(f"   All systems enabled: {enable_all_systems}")
        print("="*80)
    
    def process_cycle(self, 
                     population: List,
                     match_idx: int,
                     match_result: Dict = None) -> Dict:
        """
        Bir döngü işle (tahmin → öğrenme → evrim)
        
        Args:
            population: LoRA popülasyonu
            match_idx: Maç indexi
            match_result: Maç sonucu (opsiyonel)
            
        Returns:
            Cycle report
        """
        self.cycle_count += 1
        
        report = {
            'cycle': self.cycle_count,
            'match_idx': match_idx,
            'discoveries': [],
            'transfers': [],
            'evolutions': []
        }
        
        if not self.enable_all_systems:
            return report
        
        # 1. Update social network
        if match_result:
            for i, lora_i in enumerate(population):
                for j, lora_j in enumerate(population[i+1:], i+1):
                    bond = self.social_network.update_social_bond(lora_i, lora_j, match_result)
                    if bond > 0.7:
                        report['transfers'].append({
                            'from': lora_i.id,
                            'to': lora_j.id,
                            'bond': bond
                        })
        
        # 2. Detect discoveries
        for lora in population:
            discovery = self.knowledge_transfer.detect_discovery(
                lora, population, match_idx
            )
            
            if discovery:
                report['discoveries'].append({
                    'discoverer': lora.name,
                    'pattern': discovery.pattern,
                    'domain': discovery.source_domain.value
                })
                
                # Broadcast discovery
                broadcast_stats = self.knowledge_transfer.broadcast_discovery(
                    discovery, population, self.social_network
                )
                
                # Add to knowledge graph
                self.knowledge_graph.add_discovery_node(
                    discovery.pattern,
                    {'discoverer_id': discovery.discoverer_id}
                )
        
        # 3. Update thinking clusters
        self.social_network.update_thinking_clusters(population)
        
        # 4. Apply similarity-based attraction
        self.social_network.apply_similarity_based_attraction(population)
        
        # 5. Neuroevolution (her N cycle'da bir)
        if self.cycle_count % 10 == 0:
            for lora in population[:min(10, len(population))]:  # Top 10
                if hasattr(lora, 'fitness_history') and len(lora.fitness_history) > 0:
                    fitness = np.mean(lora.fitness_history[-20:]) if len(lora.fitness_history) >= 20 else lora.fitness_history[-1]
                    
                    if hasattr(lora, 'evolve'):
                        evolution = lora.evolve(fitness=fitness)
                        if evolution and evolution.get('architecture', {}).get('action') != 'maintain':
                            report['evolutions'].append({
                                'lora': lora.name,
                                'action': evolution.get('architecture', {}).get('action')
                            })
        
        # 6. Detect emergent patterns
        emergent = self.emergent_intelligence.detect_emergent_patterns(
            population, self.social_network
        )
        
        report['emergent_patterns'] = emergent
        
        # 7. Update statistics
        self.statistics = self.get_statistics(population)
        
        return report
    
    def route_knowledge_adaptive(self,
                                knowledge_emb: torch.Tensor,
                                population: List,
                                source_lora) -> List[Tuple[Any, float]]:
        """
        Bilgiyi adaptif olarak yönlendir
        
        Args:
            knowledge_emb: Knowledge embedding
            population: LoRA popülasyonu
            source_lora: Kaynak LoRA
            
        Returns:
            List of (target_lora, routing_probability)
        """
        routing_results = []
        
        source_domain = Domain(source_lora.specialization) if hasattr(source_lora, 'specialization') and source_lora.specialization else Domain.GENERAL
        
        for lora in population:
            if lora.id == source_lora.id:
                continue
            
            # Encode LoRA state (simplified)
            try:
                lora_params = lora.get_all_lora_params()
                param_vec = []
                for layer in ['fc1', 'fc2', 'fc3']:
                    for matrix in ['lora_A', 'lora_B']:
                        param_vec.append(lora_params[layer][matrix].flatten())
                lora_emb = torch.cat(param_vec)[:self.embed_dim].to(self.device)
                if lora_emb.shape[0] < self.embed_dim:
                    padding = torch.zeros(self.embed_dim - lora_emb.shape[0], device=self.device)
                    lora_emb = torch.cat([lora_emb, padding])
            except:
                lora_emb = torch.zeros(self.embed_dim, device=self.device)
            
            # Route
            target_domain = Domain(lora.specialization) if hasattr(lora, 'specialization') and lora.specialization else Domain.GENERAL
            routing_prob = self.adaptive_router.route_knowledge(
                knowledge_emb.to(self.device),
                lora_emb,
                target_domain
            )
            
            routing_results.append((lora, routing_prob))
        
        # Sort by probability
        routing_results.sort(key=lambda x: x[1], reverse=True)
        
        return routing_results
    
    def get_statistics(self, population: List) -> Dict:
        """Sistem istatistikleri"""
        stats = {
            'cycle_count': self.cycle_count,
            'population_size': len(population),
            'knowledge_graph': {
                'nodes': len(self.knowledge_graph.nodes),
                'edges': len(self.knowledge_graph.edges),
                'discoveries': len(self.knowledge_graph.discovery_nodes)
            },
            'social_network': self.social_network.get_network_statistics() if self.enable_all_systems else {},
            'cumulative_evolution': self.cumulative_evolution.get_generational_statistics() if self.enable_all_systems else {},
            'emergent_patterns_count': len(self.emergent_intelligence.emergent_patterns)
        }
        
        return stats
    
    def get_brain_state(self) -> Dict:
        """Beyin durumunu döndür"""
        return {
            'statistics': self.statistics,
            'knowledge_graph_size': len(self.knowledge_graph.nodes),
            'cycle_count': self.cycle_count
        }


# Global instance
_global_collective_brain = None


def get_collective_intelligence_brain(embed_dim: int = 128,
                                     device='cpu',
                                     enable_all_systems: bool = True) -> CollectiveIntelligenceBrain:
    """Global collective intelligence brain instance"""
    global _global_collective_brain
    if _global_collective_brain is None:
        _global_collective_brain = CollectiveIntelligenceBrain(
            embed_dim=embed_dim,
            device=device,
            enable_all_systems=enable_all_systems
        )
    return _global_collective_brain


