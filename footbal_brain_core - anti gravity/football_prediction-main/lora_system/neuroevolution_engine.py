"""
🧬 NEUROEVOLUTION ENGINE - ADAPTIVE NEURAL ARCHITECTURE
========================================================

Nöronlar evrilebilir! Her LoRA'nın nöron sayısı, katman yapısı, kapasitesi evrilebilmeli.

Bilimsel Temel:
- NEAT (NeuroEvolution of Augmenting Topologies)
- Progressive Neural Networks
- Neural Architecture Search (NAS)
- Dynamic Graph Neural Networks

Özellikler:
✅ Nöron sayısı evrilebilmeli
✅ Katman sayısı evrilebilmeli  
✅ Bağlantılar evrilebilmeli (sparse → dense)
✅ Kapasite evrimi (yorum derinliği)
✅ Neuron birth/death mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from enum import Enum


class EvolutionStrategy(Enum):
    """Evrim stratejileri"""
    PROGRESSIVE_GROWING = "progressive_growing"  # Aşamalı büyüme
    NEAT_LIKE = "neat_like"  # NEAT benzeri
    DIFFERENTIABLE_NAS = "differentiable_nas"  # DARTS benzeri
    HYBRID = "hybrid"  # Hibrit yaklaşım


class NeuronGrowthPolicy:
    """Nöron büyüme politikası"""
    
    @staticmethod
    def fitness_based(current_neurons: int, fitness: float, target_capacity: int) -> int:
        """
        Fitness'a göre nöron sayısı belirle
        
        Args:
            current_neurons: Mevcut nöron sayısı
            fitness: Fitness skoru (0-1)
            target_capacity: Hedef kapasite
            
        Returns:
            Yeni nöron sayısı
        """
        if fitness > 0.8:
            # Yüksek fitness → Büyüme izni
            growth_factor = min(1.0, fitness * 1.2)
            new_neurons = int(current_neurons * growth_factor)
            return min(new_neurons, target_capacity)
        elif fitness < 0.3:
            # Düşük fitness → Küçülme riski
            shrinkage_factor = max(0.7, fitness * 1.5)
            new_neurons = int(current_neurons * shrinkage_factor)
            return max(new_neurons, 32)  # Minimum 32 nöron
        else:
            return current_neurons
    
    @staticmethod
    def experience_based(match_count: int, base_neurons: int) -> int:
        """
        Tecrübe (match sayısı) ile nöron sayısı
        
        Args:
            match_count: Kaç maç tecrübe var
            base_neurons: Başlangıç nöron sayısı
            
        Returns:
            Tecrübe ile artmış nöron sayısı
        """
        # Her 50 maçta +10% nöron (max 3x)
        experience_multiplier = 1.0 + (match_count / 50) * 0.1
        experience_multiplier = min(experience_multiplier, 3.0)
        return int(base_neurons * experience_multiplier)


class AdaptiveNeuralArchitecture:
    """
    Evrilebilir nöral mimari
    
    Her LoRA'nın kendi nöron yapısı, katman sayısı, bağlantıları var
    Bunlar fitness ve tecrübeye göre evrilebilmeli
    """
    
    def __init__(self, 
                 input_dim: int = 78,
                 initial_hidden_dim: int = 128,
                 initial_num_layers: int = 3,
                 max_neurons: int = 1024,
                 min_neurons: int = 32,
                 max_layers: int = 6,
                 min_layers: int = 2,
                 strategy: EvolutionStrategy = EvolutionStrategy.PROGRESSIVE_GROWING):
        """
        Args:
            input_dim: Input boyutu
            initial_hidden_dim: Başlangıç hidden boyutu
            initial_num_layers: Başlangıç katman sayısı
            max_neurons: Maksimum nöron sayısı
            min_neurons: Minimum nöron sayısı
            max_layers: Maksimum katman sayısı
            min_layers: Minimum katman sayısı
            strategy: Evrim stratejisi
        """
        self.input_dim = input_dim
        self.initial_hidden_dim = initial_hidden_dim
        self.current_hidden_dim = initial_hidden_dim
        self.initial_num_layers = initial_num_layers
        self.current_num_layers = initial_num_layers
        self.max_neurons = max_neurons
        self.min_neurons = min_neurons
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.strategy = strategy
        
        # Mimari geçmişi (tracking için)
        self.architecture_history = []
        self.evolution_steps = 0
        
        print(f"✅ AdaptiveNeuralArchitecture initialized")
        print(f"   Initial: {initial_num_layers} layers, {initial_hidden_dim} neurons")
        print(f"   Range: {min_layers}-{max_layers} layers, {min_neurons}-{max_neurons} neurons")
    
    def get_architecture(self) -> Dict:
        """Mevcut mimariyi döndür"""
        return {
            'num_layers': self.current_num_layers,
            'hidden_dim': self.current_hidden_dim,
            'total_neurons': self._calculate_total_neurons(),
            'strategy': self.strategy.value
        }
    
    def _calculate_total_neurons(self) -> int:
        """Toplam nöron sayısını hesapla"""
        # Input layer + hidden layers + output layer
        total = self.input_dim  # Input
        for _ in range(self.current_num_layers - 1):
            total += self.current_hidden_dim  # Hidden layers
        total += 3  # Output (home/draw/away)
        return total
    
    def evolve_architecture(self, 
                           fitness: float,
                           match_count: int = 0,
                           recent_performance: List[float] = None) -> Dict:
        """
        Mimariyi evril
        
        Args:
            fitness: Mevcut fitness (0-1)
            match_count: Toplam maç sayısı
            recent_performance: Son N maçın performansı
            
        Returns:
            Evolution decision: {'action': 'grow'|'shrink'|'maintain', 'details': ...}
        """
        recent_performance = recent_performance or []
        
        if self.strategy == EvolutionStrategy.PROGRESSIVE_GROWING:
            return self._progressive_growing(fitness, match_count, recent_performance)
        elif self.strategy == EvolutionStrategy.NEAT_LIKE:
            return self._neat_like(fitness, match_count)
        elif self.strategy == EvolutionStrategy.HYBRID:
            return self._hybrid_evolution(fitness, match_count, recent_performance)
        else:
            return {'action': 'maintain', 'details': 'No evolution'}
    
    def _progressive_growing(self, fitness: float, match_count: int, recent_performance: List[float]) -> Dict:
        """
        Progressive Growing: Başarılı olursa büyür
        
        - Yüksek fitness → Daha fazla nöron
        - Düşük fitness → Küçülme riski
        - Tecrübe arttıkça kapasite artar
        """
        old_dim = self.current_hidden_dim
        old_layers = self.current_num_layers
        
        # 1. Fitness-based growth
        if fitness > 0.75 and len(recent_performance) >= 10:
            avg_recent = np.mean(recent_performance[-10:])
            if avg_recent > 0.7:
                # Büyüme zamanı!
                # Nöron sayısını artır
                growth_factor = 1.15  # %15 artış
                new_dim = int(self.current_hidden_dim * growth_factor)
                new_dim = min(new_dim, self.max_neurons)
                
                # Katman sayısını da artırabiliriz (daha nadir)
                if fitness > 0.85 and random.random() < 0.3:
                    new_layers = min(self.current_num_layers + 1, self.max_layers)
                    self.current_num_layers = new_layers
                
                self.current_hidden_dim = new_dim
                
                decision = {
                    'action': 'grow',
                    'old_dim': old_dim,
                    'new_dim': self.current_hidden_dim,
                    'old_layers': old_layers,
                    'new_layers': self.current_num_layers,
                    'reason': f'High fitness ({fitness:.2f}) and good recent performance'
                }
                
                self.architecture_history.append(decision)
                self.evolution_steps += 1
                return decision
        
        # 2. Experience-based growth
        experience_dim = NeuronGrowthPolicy.experience_based(match_count, self.initial_hidden_dim)
        if experience_dim > self.current_hidden_dim and fitness > 0.5:
            self.current_hidden_dim = min(experience_dim, self.max_neurons)
            
            decision = {
                'action': 'grow',
                'old_dim': old_dim,
                'new_dim': self.current_hidden_dim,
                'reason': f'Experience-based growth ({match_count} matches)'
            }
            
            self.architecture_history.append(decision)
            self.evolution_steps += 1
            return decision
        
        # 3. Shrinkage (düşük performans)
        if fitness < 0.25 and len(recent_performance) >= 10:
            avg_recent = np.mean(recent_performance[-10:])
            if avg_recent < 0.3:
                # Küçülme zamanı
                shrinkage_factor = 0.85  # %15 azalma
                new_dim = int(self.current_hidden_dim * shrinkage_factor)
                new_dim = max(new_dim, self.min_neurons)
                
                self.current_hidden_dim = new_dim
                
                decision = {
                    'action': 'shrink',
                    'old_dim': old_dim,
                    'new_dim': self.current_hidden_dim,
                    'reason': f'Low fitness ({fitness:.2f}) and poor recent performance'
                }
                
                self.architecture_history.append(decision)
                self.evolution_steps += 1
                return decision
        
        return {'action': 'maintain', 'details': 'No significant change needed'}
    
    def _neat_like(self, fitness: float, match_count: int) -> Dict:
        """
        NEAT-like evolution: Daha agresif mutasyonlar
        """
        old_dim = self.current_hidden_dim
        
        # NEAT: Yüksek fitness → Mutasyon şansı düşük ama etkili
        if fitness > 0.8 and random.random() < 0.2:
            # Add neuron
            new_dim = min(self.current_hidden_dim + random.randint(8, 32), self.max_neurons)
            self.current_hidden_dim = new_dim
            
            return {
                'action': 'grow',
                'old_dim': old_dim,
                'new_dim': self.current_hidden_dim,
                'reason': 'NEAT-style neuron addition'
            }
        
        return {'action': 'maintain'}
    
    def _hybrid_evolution(self, fitness: float, match_count: int, recent_performance: List[float]) -> Dict:
        """Hibrit yaklaşım: Progressive + NEAT"""
        # Progressive growth
        prog_result = self._progressive_growing(fitness, match_count, recent_performance)
        if prog_result['action'] != 'maintain':
            return prog_result
        
        # NEAT-style mutation (daha nadir)
        if random.random() < 0.1:
            return self._neat_like(fitness, match_count)
        
        return {'action': 'maintain'}
    
    def get_connection_sparsity(self) -> float:
        """
        Bağlantı seyreklik oranı
        
        Returns:
            0.0 (fully dense) - 1.0 (fully sparse)
        """
        # Büyük ağlar daha sparse olabilir
        total_neurons = self._calculate_total_neurons()
        if total_neurons > 500:
            return 0.3  # %30 sparse
        elif total_neurons > 200:
            return 0.2  # %20 sparse
        else:
            return 0.1  # %10 sparse


class CapacityEvolution:
    """
    Kapasite evrimi: Yorum derinliği, düşünme kapasitesi
    
    Sadece nöron sayısı değil, nöronların "düşünme derinliği" de evrilebilmeli
    """
    
    def __init__(self):
        self.capacity_level = 1.0  # Başlangıç kapasitesi
        self.thinking_depth = 1  # Düşünme derinliği (kaç iterasyon düşünür)
        self.attention_capacity = 1.0  # Attention kapasitesi
        self.memory_capacity = 1.0  # Hafıza kapasitesi
    
    def evolve_capacity(self, fitness: float, architecture: Dict) -> Dict:
        """
        Kapasiteyi evril
        
        Args:
            fitness: Fitness skoru
            architecture: Mimari bilgisi
            
        Returns:
            Capacity evolution decision
        """
        old_capacity = self.capacity_level
        
        # Mimari büyüdükçe kapasite de artar
        total_neurons = architecture.get('total_neurons', 100)
        neuron_ratio = total_neurons / 200.0  # 200 nöron = 1.0x kapasite
        
        # Fitness yüksekse daha derin düşünebilir
        if fitness > 0.8:
            self.thinking_depth = max(2, int(2 * fitness))
            self.attention_capacity = min(2.0, 1.0 + (fitness - 0.8) * 2.0)
        
        # Kapasite = nöron oranı × fitness bonus
        self.capacity_level = neuron_ratio * (1.0 + (fitness - 0.5) * 0.5)
        self.capacity_level = max(0.5, min(self.capacity_level, 3.0))  # 0.5x - 3.0x arası
        
        # Hafıza kapasitesi
        self.memory_capacity = min(2.0, 1.0 + (fitness - 0.5) * 1.0)
        
        return {
            'old_capacity': old_capacity,
            'new_capacity': self.capacity_level,
            'thinking_depth': self.thinking_depth,
            'attention_capacity': self.attention_capacity,
            'memory_capacity': self.memory_capacity
        }
    
    def get_capacity_multiplier(self) -> float:
        """Kapasite çarpanı"""
        return self.capacity_level


class NeuroevolutionEngine:
    """
    🧬 MASTER NEUROEVOLUTION ENGINE
    
    Tüm nöroevrim işlemlerini yöneten merkezi motor
    """
    
    def __init__(self, 
                 initial_hidden_dim: int = 128,
                 initial_num_layers: int = 3,
                 strategy: EvolutionStrategy = EvolutionStrategy.PROGRESSIVE_GROWING):
        """
        Args:
            initial_hidden_dim: Başlangıç hidden boyutu
            initial_num_layers: Başlangıç katman sayısı
            strategy: Evrim stratejisi
        """
        self.architecture = AdaptiveNeuralArchitecture(
            initial_hidden_dim=initial_hidden_dim,
            initial_num_layers=initial_num_layers,
            strategy=strategy
        )
        self.capacity = CapacityEvolution()
        
        print("="*80)
        print("🧬 NEUROEVOLUTION ENGINE INITIALIZED")
        print("="*80)
    
    def evolve_lora(self, lora, fitness: float, match_count: int = None, recent_performance: List[float] = None) -> Dict:
        """
        Bir LoRA'yı evril
        
        Args:
            lora: LoRA adapter instance
            fitness: Fitness skoru
            match_count: Maç sayısı (opsiyonel, lora.match_history'den alınabilir)
            recent_performance: Son performanslar (opsiyonel)
            
        Returns:
            Evolution report
        """
        if match_count is None:
            match_count = len(getattr(lora, 'match_history', []))
        
        if recent_performance is None:
            recent_performance = getattr(lora, 'fitness_history', [])[-20:] if hasattr(lora, 'fitness_history') else []
        
        # 1. Mimari evrimi
        arch_evolution = self.architecture.evolve_architecture(fitness, match_count, recent_performance)
        
        # 2. Kapasite evrimi
        current_arch = self.architecture.get_architecture()
        capacity_evolution = self.capacity.evolve_capacity(fitness, current_arch)
        
        # 3. Evolution report
        report = {
            'architecture': arch_evolution,
            'capacity': capacity_evolution,
            'new_architecture': self.architecture.get_architecture(),
            'new_capacity': {
                'level': self.capacity.capacity_level,
                'thinking_depth': self.capacity.thinking_depth,
                'attention': self.capacity.attention_capacity,
                'memory': self.capacity.memory_capacity
            }
        }
        
        # LoRA'ya yeni mimariyi kaydet
        lora.neuroevolution_state = {
            'architecture': self.architecture.get_architecture(),
            'capacity': {
                'level': self.capacity.capacity_level,
                'thinking_depth': self.capacity.thinking_depth
            },
            'evolution_history': self.architecture.architecture_history[-5:]  # Son 5 evrim
        }
        
        return report
    
    def get_evolved_architecture(self) -> Dict:
        """Evrilmiş mimariyi döndür"""
        return self.architecture.get_architecture()
    
    def get_capacity_info(self) -> Dict:
        """Kapasite bilgilerini döndür"""
        return {
            'capacity_level': self.capacity.capacity_level,
            'thinking_depth': self.capacity.thinking_depth,
            'attention_capacity': self.capacity.attention_capacity,
            'memory_capacity': self.capacity.memory_capacity
        }


# Global instance
_global_neuroevolution_engine = None


def get_neuroevolution_engine(initial_hidden_dim: int = 128, 
                              initial_num_layers: int = 3,
                              strategy: EvolutionStrategy = EvolutionStrategy.PROGRESSIVE_GROWING):
    """Global neuroevolution engine instance"""
    global _global_neuroevolution_engine
    if _global_neuroevolution_engine is None:
        _global_neuroevolution_engine = NeuroevolutionEngine(
            initial_hidden_dim=initial_hidden_dim,
            initial_num_layers=initial_num_layers,
            strategy=strategy
        )
    return _global_neuroevolution_engine


