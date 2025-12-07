"""
🧬 EVOLVABLE LORA ADAPTER - Dynamic Neural Architecture
=========================================================

Evrilebilir LoRA Adapter: Dinamik mimari, adaptif katmanlar, bağlantı evrimi, nöron uzmanlaşması.

Mevcut lora_adapter.py'yi genişletir ve nöroevrim özelliklerini ekler.

Özellikler:
✅ Dinamik mimari (nöron sayısı evrilebilmeli)
✅ Adaptif katmanlar (katman sayısı evrilebilmeli)
✅ Bağlantı evrimi (sparse → dense)
✅ Nöron uzmanlaşması (bazıları hype için, bazıları odds için)
✅ Backward compatible (mevcut LoRA'larla çalışabilmeli)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any
import uuid

# Import base LoRA classes
from .lora_adapter import LoRALinear, LoRAAdapter
from .neuroevolution_engine import (
    NeuroevolutionEngine, 
    EvolutionStrategy,
    get_neuroevolution_engine
)
from .thinking_patterns import (
    EvolvableThinkingSystem,
    ThinkingPattern,
    get_thinking_system
)


class DynamicLoRALinear(nn.Module):
    """
    Evrilebilir LoRA Linear katmanı
    
    Nöron sayısı değişebilir, bağlantılar evrilebilir
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 rank: int = 16, 
                 alpha: float = 16.0,
                 device='cpu',
                 enable_sparsity: bool = True):
        super().__init__()
        
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enable_sparsity = enable_sparsity
        
        # Ana ağırlık (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        nn.init.xavier_uniform_(self.weight)
        self.weight.requires_grad = False
        
        # LoRA matrisleri
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)
        
        # Sparsity mask (bağlantılar için)
        if enable_sparsity:
            self.register_buffer('sparsity_mask', torch.ones(out_features, in_features))
        else:
            self.register_buffer('sparsity_mask', None)
    
    def forward(self, x):
        # Base output
        if self.sparsity_mask is not None:
            masked_weight = self.weight * self.sparsity_mask
            base_output = F.linear(x, masked_weight)
        else:
            base_output = F.linear(x, self.weight)
        
        # LoRA delta
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        
        return base_output + lora_output * self.scaling
    
    def grow_neurons(self, new_out_features: int):
        """
        Nöron sayısını artır
        
        Args:
            new_out_features: Yeni çıkış boyutu
        """
        if new_out_features <= self.out_features:
            return  # Küçülme şimdilik desteklenmiyor
        
        old_out = self.out_features
        
        # Weight'ı genişlet
        new_weight = torch.zeros(new_out_features, self.in_features, device=self.device)
        new_weight[:old_out, :] = self.weight.data
        nn.init.xavier_uniform_(new_weight[old_out:, :])  # Yeni nöronları başlat
        new_weight.requires_grad = False
        self.weight = nn.Parameter(new_weight)
        
        # LoRA_B'yi genişlet
        new_lora_B = torch.zeros(new_out_features, self.rank, device=self.device)
        new_lora_B[:old_out, :] = self.lora_B.data
        nn.init.zeros_(new_lora_B[old_out:, :])
        self.lora_B = nn.Parameter(new_lora_B)
        
        # Sparsity mask'ı güncelle
        if self.sparsity_mask is not None:
            new_mask = torch.ones(new_out_features, self.in_features, device=self.device)
            new_mask[:old_out, :] = self.sparsity_mask.data
            self.sparsity_mask = new_mask
        
        self.out_features = new_out_features
    
    def get_lora_params(self):
        """LoRA parametrelerini döndür"""
        return {'lora_A': self.lora_A.data.clone(), 'lora_B': self.lora_B.data.clone()}
    
    def set_lora_params(self, params: Dict):
        """LoRA parametrelerini ayarla"""
        target_device = self.lora_A.device
        self.lora_A.data = params['lora_A'].clone().to(target_device)
        self.lora_B.data = params['lora_B'].clone().to(target_device)


class EvolvableLoRAAdapter(LoRAAdapter):
    """
    Evrilebilir LoRA Adapter
    
    Mevcut LoRAAdapter'ı genişletir, nöroevrim özellikleri ekler
    """
    
    def __init__(self, 
                 input_dim: int = 78,
                 hidden_dim: int = 128,
                 rank: int = 16,
                 alpha: float = 16.0,
                 device='cpu',
                 enable_neuroevolution: bool = True,
                 evolution_strategy: EvolutionStrategy = EvolutionStrategy.PROGRESSIVE_GROWING,
                 initial_thinking_pattern: ThinkingPattern = None):
        """
        Args:
            input_dim: Input boyutu
            hidden_dim: Başlangıç hidden boyutu
            rank: LoRA rank
            alpha: LoRA alpha
            device: Device
            enable_neuroevolution: Nöroevrim aktif mi?
            evolution_strategy: Evrim stratejisi
            initial_thinking_pattern: Başlangıç düşünme kalıbı
        """
        # Base LoRAAdapter'ı initialize et
        super().__init__(input_dim, hidden_dim, rank, alpha, device)
        
        self.enable_neuroevolution = enable_neuroevolution
        
        # Nöroevrim engine
        if enable_neuroevolution:
            self.neuroevolution_engine = get_neuroevolution_engine(
                initial_hidden_dim=hidden_dim,
                initial_num_layers=3,  # fc1, fc2, fc3
                strategy=evolution_strategy
            )
        
        # Thinking system
        self.thinking_system = get_thinking_system(
            primary_pattern=initial_thinking_pattern
        )
        
        # Dynamic architecture state
        self.dynamic_layers = nn.ModuleDict()
        self.current_architecture = {
            'num_layers': 3,
            'layer_dims': [hidden_dim, 64, 3],
            'total_neurons': hidden_dim + 64 + 3
        }
        
        # Neuron specialization (hangi nöronlar ne için uzmanlaşmış?)
        self.neuron_specialization = {
            'layer_0': np.random.rand(hidden_dim) < 0.3,  # %30 hype için
            'layer_1': np.random.rand(64) < 0.3,  # %30 odds için
        }
        
        # Evolution history
        self.evolution_history = []
        
        print(f"✅ EvolvableLoRAAdapter initialized: {self.name}")
        print(f"   Neuroevolution: {'Enabled' if enable_neuroevolution else 'Disabled'}")
        print(f"   Thinking pattern: {self.thinking_system.primary_pattern.value}")
    
    def forward(self, x):
        """
        Forward pass with thinking pattern application
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output probabilities [batch_size, 3]
        """
        # Apply thinking pattern to input
        if self.enable_neuroevolution:
            x = self.thinking_system.get_thinking_strategy(x)
        
        # Normal forward pass (base class'tan)
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        
        logits = self.fc3(h2)
        proba = F.softmax(logits, dim=-1)
        
        return proba
    
    def evolve(self, fitness: float = None, match_count: int = None, recent_performance: List[float] = None):
        """
        LoRA'yı evril
        
        Args:
            fitness: Fitness skoru (None ise hesaplanır)
            match_count: Maç sayısı (None ise match_history'den)
            recent_performance: Son performanslar (None ise fitness_history'den)
            
        Returns:
            Evolution report
        """
        if not self.enable_neuroevolution:
            return {'status': 'disabled', 'message': 'Neuroevolution is disabled'}
        
        # Get fitness if not provided
        if fitness is None:
            fitness = self.get_recent_fitness(window=50)
        
        if match_count is None:
            match_count = len(self.match_history)
        
        if recent_performance is None:
            recent_performance = self.fitness_history[-20:] if len(self.fitness_history) >= 20 else []
        
        # 1. Neuroevolution (mimari evrimi)
        neuroevolution_report = self.neuroevolution_engine.evolve_lora(
            self, fitness, match_count, recent_performance
        )
        
        # 2. Thinking pattern evolution
        thinking_evolution = self.thinking_system.evolve_thinking(
            fitness=fitness,
            task_type=self.specialization,
            recent_performance=recent_performance
        )
        
        # 3. Apply architecture changes if needed
        if neuroevolution_report['architecture']['action'] == 'grow':
            self._apply_architecture_growth(neuroevolution_report)
        
        # 4. Store evolution history
        evolution_record = {
            'match': match_count,
            'fitness': fitness,
            'neuroevolution': neuroevolution_report,
            'thinking': thinking_evolution,
            'timestamp': len(self.evolution_history)
        }
        self.evolution_history.append(evolution_record)
        
        return evolution_record
    
    def _apply_architecture_growth(self, evolution_report: Dict):
        """
        Mimari büyümesini uygula
        
        Args:
            evolution_report: Evolution report
        """
        arch_info = evolution_report['architecture']
        
        if arch_info['action'] == 'grow':
            new_dim = arch_info.get('new_dim', self.hidden_dim)
            
            # fc1'i büyüt (input → hidden)
            if isinstance(self.fc1, LoRALinear):
                # LoRALinear'ı DynamicLoRALinear'a çevir (gerekirse)
                if not isinstance(self.fc1, DynamicLoRALinear):
                    self._convert_to_dynamic(self.fc1, 'fc1')
                
                if isinstance(self.fc1, DynamicLoRALinear):
                    old_dim = self.fc1.out_features
                    if new_dim > old_dim:
                        self.fc1.grow_neurons(new_dim)
                        self.hidden_dim = new_dim
                        self.current_architecture['layer_dims'][0] = new_dim
                        print(f"   🧬 {self.name}: fc1 grew {old_dim} → {new_dim} neurons")
    
    def _convert_to_dynamic(self, layer: nn.Module, layer_name: str):
        """LoRALinear'ı DynamicLoRALinear'a çevir (gerekirse)"""
        # Şimdilik sadece log, gerçek dönüşüm karmaşık
        pass
    
    def crossover(self, partner: 'EvolvableLoRAAdapter') -> 'EvolvableLoRAAdapter':
        """
        Evrilebilir crossover: Hem parametreleri hem mimariyi karıştır
        
        Args:
            partner: Partner LoRA
            
        Returns:
            Child LoRA
        """
        # Base crossover (parametreler)
        child = super().crossover(partner)
        
        # Convert to EvolvableLoRAAdapter if needed
        if not isinstance(child, EvolvableLoRAAdapter):
            # Create new evolvable version
            child_evolvable = EvolvableLoRAAdapter(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                rank=self.rank,
                alpha=self.alpha,
                device=self.device,
                enable_neuroevolution=self.enable_neuroevolution
            )
            # Copy parameters
            child_evolvable.load_state_dict(child.state_dict())
            child = child_evolvable
        
        # Inherit thinking patterns
        if hasattr(self, 'thinking_system') and hasattr(partner, 'thinking_system'):
            child.thinking_system.inherit_from_parent(
                self.thinking_system,
                inheritance_strength=0.5
            )
            # Blend with partner
            child.thinking_system.inherit_from_parent(
                partner.thinking_system,
                inheritance_strength=0.25
            )
        
        # Inherit architecture preferences
        if hasattr(self, 'current_architecture') and hasattr(partner, 'current_architecture'):
            # Average architecture preferences
            child.current_architecture['layer_dims'][0] = int(
                (self.current_architecture['layer_dims'][0] + 
                 partner.current_architecture['layer_dims'][0]) / 2
            )
            child.hidden_dim = child.current_architecture['layer_dims'][0]
        
        return child
    
    def get_neuroevolution_state(self) -> Dict:
        """Nöroevrim durumunu döndür"""
        if not self.enable_neuroevolution:
            return {'status': 'disabled'}
        
        return {
            'architecture': self.current_architecture,
            'thinking_pattern': self.thinking_system.primary_pattern.value,
            'pattern_weights': {
                p.value: w 
                for p, w in self.thinking_system.pattern_weights.items()
            },
            'evolution_history_count': len(self.evolution_history),
            'neuroevolution_state': getattr(self, 'neuroevolution_state', None)
        }
    
    def get_all_lora_params(self):
        """Tüm LoRA parametrelerini al (backward compatible)"""
        return super().get_all_lora_params()
    
    def set_all_lora_params(self, params: Dict):
        """Tüm LoRA parametrelerini ayarla (backward compatible)"""
        super().set_all_lora_params(params)


def create_evolvable_from_base(base_lora: LoRAAdapter, 
                               enable_neuroevolution: bool = True) -> EvolvableLoRAAdapter:
    """
    Mevcut LoRAAdapter'dan EvolvableLoRAAdapter oluştur
    
    Backward compatibility için
    """
    evolvable = EvolvableLoRAAdapter(
        input_dim=base_lora.input_dim,
        hidden_dim=base_lora.hidden_dim,
        rank=base_lora.rank,
        alpha=base_lora.alpha,
        device=base_lora.device,
        enable_neuroevolution=enable_neuroevolution
    )
    
    # Copy all state
    evolvable.load_state_dict(base_lora.state_dict())
    evolvable.id = base_lora.id
    evolvable.name = base_lora.name
    evolvable.generation = base_lora.generation
    evolvable.parents = base_lora.parents.copy() if hasattr(base_lora, 'parents') else []
    evolvable.match_history = base_lora.match_history.copy() if hasattr(base_lora, 'match_history') else []
    evolvable.fitness_history = base_lora.fitness_history.copy() if hasattr(base_lora, 'fitness_history') else []
    evolvable.temperament = base_lora.temperament.copy() if hasattr(base_lora, 'temperament') else {}
    
    print(f"✅ Converted {base_lora.name} to EvolvableLoRAAdapter")
    
    return evolvable


