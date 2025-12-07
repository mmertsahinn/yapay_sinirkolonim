"""
🧠 THINKING PATTERNS - EVOLVABLE THOUGHT PROCESSES
====================================================

Her LoRA'nın düşünme biçimi evrilebilmeli. Farklı düşünme kalıpları, pattern library,
inheritance ve adaptive reasoning.

Bilimsel Temel:
- Cognitive architectures
- Reasoning patterns
- Transfer learning in reasoning
- Pattern-based thinking

Özellikler:
✅ Thinking style evolution
✅ Pattern library (analitik, sezgisel, kombinatorik, vs.)
✅ Pattern inheritance (ebeveynlerden)
✅ Adaptive reasoning (göreve göre seçim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import random


class ThinkingPattern(Enum):
    """Düşünme kalıpları"""
    ANALYTICAL = "analytical"  # Analitik, mantıksal
    INTUITIVE = "intuitive"  # Sezgisel, hızlı
    COMBINATORIAL = "combinatorial"  # Kombinatoryal, çok seçenekli
    SEQUENTIAL = "sequential"  # Sıralı, adım adım
    PARALLEL = "parallel"  # Paralel, eşzamanlı
    HOLISTIC = "holistic"  # Bütünsel, genel bakış
    FOCUSED = "focused"  # Odaklanmış, derinlemesine
    DIVERGENT = "divergent"  # Iraksak, çok yönlü


class PatternStrength:
    """Pattern güç seviyeleri"""
    WEAK = 0.0
    MODERATE = 0.5
    STRONG = 1.0


class ThinkingPatternLibrary:
    """
    Düşünme kalıpları kütüphanesi
    
    Her pattern farklı bir bilgi işleme stratejisi
    """
    
    @staticmethod
    def get_pattern_characteristics() -> Dict[ThinkingPattern, Dict]:
        """
        Her pattern'in özellikleri
        
        Returns:
            Pattern → Characteristics mapping
        """
        return {
            ThinkingPattern.ANALYTICAL: {
                'depth': 0.9,  # Derin düşünür
                'speed': 0.3,  # Yavaş
                'precision': 0.9,  # Yüksek kesinlik
                'exploration': 0.4,  # Az keşif
                'confidence_threshold': 0.8,  # Yüksek güven eşiği
                'requires_data': True
            },
            ThinkingPattern.INTUITIVE: {
                'depth': 0.3,
                'speed': 0.9,
                'precision': 0.6,
                'exploration': 0.7,
                'confidence_threshold': 0.5,
                'requires_data': False
            },
            ThinkingPattern.COMBINATORIAL: {
                'depth': 0.7,
                'speed': 0.5,
                'precision': 0.7,
                'exploration': 0.9,  # Çok keşif
                'confidence_threshold': 0.6,
                'requires_data': True
            },
            ThinkingPattern.SEQUENTIAL: {
                'depth': 0.8,
                'speed': 0.4,
                'precision': 0.8,
                'exploration': 0.3,
                'confidence_threshold': 0.7,
                'requires_data': True
            },
            ThinkingPattern.PARALLEL: {
                'depth': 0.6,
                'speed': 0.8,
                'precision': 0.7,
                'exploration': 0.6,
                'confidence_threshold': 0.6,
                'requires_data': True
            },
            ThinkingPattern.HOLISTIC: {
                'depth': 0.7,
                'speed': 0.6,
                'precision': 0.6,
                'exploration': 0.8,
                'confidence_threshold': 0.5,
                'requires_data': True
            },
            ThinkingPattern.FOCUSED: {
                'depth': 0.95,  # Çok derin
                'speed': 0.3,
                'precision': 0.9,
                'exploration': 0.2,  # Az keşif
                'confidence_threshold': 0.85,
                'requires_data': True
            },
            ThinkingPattern.DIVERGENT: {
                'depth': 0.5,
                'speed': 0.7,
                'precision': 0.5,
                'exploration': 0.95,  # Çok keşif
                'confidence_threshold': 0.4,
                'requires_data': False
            }
        }
    
    @staticmethod
    def apply_pattern(input_data: torch.Tensor, 
                     pattern: ThinkingPattern,
                     pattern_strength: float = 1.0) -> torch.Tensor:
        """
        Düşünme kalıbını input'a uygula
        
        Args:
            input_data: [batch_size, features]
            pattern: Düşünme kalıbı
            pattern_strength: Pattern gücü (0-1)
            
        Returns:
            Processed data
        """
        chars = ThinkingPatternLibrary.get_pattern_characteristics()[pattern]
        
        # Depth: Daha fazla transform
        if pattern == ThinkingPattern.ANALYTICAL or pattern == ThinkingPattern.FOCUSED:
            # Derin analiz: Multiple passes
            depth = int(chars['depth'] * pattern_strength * 3) + 1
            processed = input_data
            for _ in range(depth):
                processed = F.layer_norm(processed, processed.shape[-1:])
                processed = processed * (1.0 + chars['depth'] * 0.1)
            return processed
        
        # Exploration: Noise ekle (keşif için)
        if pattern == ThinkingPattern.DIVERGENT or pattern == ThinkingPattern.COMBINATORIAL:
            noise = torch.randn_like(input_data) * chars['exploration'] * pattern_strength * 0.1
            return input_data + noise
        
        # Speed: Hızlı işleme (basit transform)
        if pattern == ThinkingPattern.INTUITIVE or pattern == ThinkingPattern.PARALLEL:
            return F.relu(input_data)  # Basit activation
        
        # Default: Linear transform
        return input_data


class EvolvableThinkingSystem:
    """
    Evrilebilir düşünme sistemi
    
    Her LoRA'nın düşünme biçimleri var ve bunlar evrilebilmeli
    """
    
    def __init__(self, 
                 primary_pattern: ThinkingPattern = None,
                 pattern_weights: Dict[ThinkingPattern, float] = None):
        """
        Args:
            primary_pattern: Birincil düşünme kalıbı
            pattern_weights: Her pattern'in ağırlığı
        """
        # Pattern ağırlıkları (toplam = 1.0)
        if pattern_weights is None:
            # Rastgele başlangıç
            patterns = list(ThinkingPattern)
            weights = np.random.dirichlet(np.ones(len(patterns)))
            self.pattern_weights = {
                pattern: float(weight) 
                for pattern, weight in zip(patterns, weights)
            }
        else:
            self.pattern_weights = pattern_weights
        
        # Primary pattern (en yüksek ağırlıklı)
        if primary_pattern is None:
            self.primary_pattern = max(self.pattern_weights.items(), key=lambda x: x[1])[0]
        else:
            self.primary_pattern = primary_pattern
        
        # Pattern history (evrim tracking)
        self.pattern_evolution_history = []
        self.performance_by_pattern = {pattern: [] for pattern in ThinkingPattern}
        
        print(f"✅ EvolvableThinkingSystem initialized")
        print(f"   Primary pattern: {self.primary_pattern.value}")
        print(f"   Pattern distribution: {[(p.value, f'{w:.2f}') for p, w in sorted(self.pattern_weights.items(), key=lambda x: -x[1])[:3]]}")
    
    def evolve_thinking(self, 
                       fitness: float,
                       task_type: str = None,
                       recent_performance: List[float] = None) -> Dict:
        """
        Düşünme biçimini evril
        
        Args:
            fitness: Fitness skoru
            task_type: Görev tipi (hype_expert, odds_expert, vs.)
            recent_performance: Son performanslar
            
        Returns:
            Evolution decision
        """
        # 1. Pattern performanslarını güncelle
        if recent_performance:
            avg_perf = np.mean(recent_performance)
            self.performance_by_pattern[self.primary_pattern].append(avg_perf)
        
        # 2. Task-based pattern adaptation
        if task_type:
            recommended_pattern = self._recommend_pattern_for_task(task_type)
            if recommended_pattern != self.primary_pattern:
                # Pattern shift
                shift_strength = min(0.2, fitness * 0.3)
                self._shift_pattern_weights(recommended_pattern, shift_strength)
        
        # 3. Performance-based evolution
        if fitness > 0.75:
            # Başarılı → Primary pattern'i güçlendir
            self._strengthen_pattern(self.primary_pattern, 0.1)
        elif fitness < 0.3:
            # Başarısız → Alternatif pattern'ler dene
            self._explore_alternative_patterns(0.15)
        
        # 4. Update primary pattern
        old_primary = self.primary_pattern
        self.primary_pattern = max(self.pattern_weights.items(), key=lambda x: x[1])[0]
        
        evolution = {
            'old_primary': old_primary.value,
            'new_primary': self.primary_pattern.value,
            'pattern_weights': {p.value: w for p, w in self.pattern_weights.items()},
            'fitness': fitness
        }
        
        self.pattern_evolution_history.append(evolution)
        
        return evolution
    
    def _recommend_pattern_for_task(self, task_type: str) -> ThinkingPattern:
        """
        Görev tipine göre önerilen pattern
        
        Args:
            task_type: 'hype_expert', 'odds_expert', 'general', vs.
            
        Returns:
            Önerilen pattern
        """
        task_patterns = {
            'hype_expert': ThinkingPattern.INTUITIVE,  # Hızlı sezgisel
            'odds_expert': ThinkingPattern.ANALYTICAL,  # Analitik
            'score_predictor': ThinkingPattern.COMBINATORIAL,  # Çok seçenekli
            'defensive': ThinkingPattern.SEQUENTIAL,  # Adım adım
            'offensive': ThinkingPattern.PARALLEL,  # Hızlı paralel
            'general': ThinkingPattern.HOLISTIC  # Bütünsel
        }
        
        return task_patterns.get(task_type, ThinkingPattern.HOLISTIC)
    
    def _strengthen_pattern(self, pattern: ThinkingPattern, strength: float):
        """Bir pattern'i güçlendir"""
        # Bu pattern'in ağırlığını artır, diğerlerini azalt
        total_weight = sum(self.pattern_weights.values())
        self.pattern_weights[pattern] = min(1.0, self.pattern_weights[pattern] + strength)
        
        # Normalize
        current_total = sum(self.pattern_weights.values())
        if current_total > 1.0:
            for p in self.pattern_weights:
                self.pattern_weights[p] /= current_total
    
    def _shift_pattern_weights(self, target_pattern: ThinkingPattern, shift_strength: float):
        """Pattern ağırlıklarını hedefe kaydır"""
        # Mevcut primary'den target'a kaydır
        current_primary_weight = self.pattern_weights.get(self.primary_pattern, 0.0)
        
        if current_primary_weight > shift_strength:
            self.pattern_weights[self.primary_pattern] -= shift_strength
            self.pattern_weights[target_pattern] = self.pattern_weights.get(target_pattern, 0.0) + shift_strength
    
    def _explore_alternative_patterns(self, exploration_rate: float):
        """Alternatif pattern'ler dene"""
        # En az kullanılan pattern'e biraz ağırlık ver
        min_pattern = min(self.pattern_weights.items(), key=lambda x: x[1])[0]
        max_pattern = max(self.pattern_weights.items(), key=lambda x: x[1])[0]
        
        if max_pattern != min_pattern:
            shift = min(exploration_rate, self.pattern_weights[max_pattern] * 0.2)
            self.pattern_weights[max_pattern] -= shift
            self.pattern_weights[min_pattern] += shift
    
    def get_thinking_strategy(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Input'a düşünme stratejisini uygula
        
        Args:
            input_data: [batch_size, features]
            
        Returns:
            Processed data (thinking applied)
        """
        # Weighted combination of patterns
        result = torch.zeros_like(input_data)
        total_weight = 0.0
        
        for pattern, weight in self.pattern_weights.items():
            if weight > 0.01:  # Sadece önemli pattern'ler
                pattern_result = ThinkingPatternLibrary.apply_pattern(
                    input_data, pattern, weight
                )
                result += pattern_result * weight
                total_weight += weight
        
        if total_weight > 0:
            result /= total_weight
        
        return result
    
    def inherit_from_parent(self, parent: 'EvolvableThinkingSystem', inheritance_strength: float = 0.5):
        """
        Ebeveynin düşünme biçimini miras al
        
        Args:
            parent: Ebeveyn thinking system
            inheritance_strength: Miras gücü (0-1)
        """
        # Pattern ağırlıklarını blend et
        for pattern in ThinkingPattern:
            parent_weight = parent.pattern_weights.get(pattern, 0.0)
            child_weight = self.pattern_weights.get(pattern, 0.0)
            
            # Blend: child = (1-α) × child + α × parent
            blended = (1 - inheritance_strength) * child_weight + inheritance_strength * parent_weight
            self.pattern_weights[pattern] = blended
        
        # Normalize
        total = sum(self.pattern_weights.values())
        if total > 0:
            for pattern in self.pattern_weights:
                self.pattern_weights[pattern] /= total
        
        # Update primary
        self.primary_pattern = max(self.pattern_weights.items(), key=lambda x: x[1])[0]
        
        print(f"   🧬 Inherited thinking patterns from parent (strength: {inheritance_strength:.2f})")
        print(f"      New primary: {self.primary_pattern.value}")


class AdaptiveReasoning:
    """
    Adaptive Reasoning: Göreve göre düşünme biçimi seçimi
    
    Her görev için en uygun pattern'i dinamik olarak seçer
    """
    
    def __init__(self, thinking_system: EvolvableThinkingSystem):
        """
        Args:
            thinking_system: EvolvableThinkingSystem instance
        """
        self.thinking_system = thinking_system
        self.task_pattern_map = {}  # Task → Best pattern history
    
    def select_optimal_pattern(self, 
                              task_type: str,
                              input_complexity: float,
                              time_constraint: float) -> ThinkingPattern:
        """
        Göreve göre optimal pattern seç
        
        Args:
            task_type: Görev tipi
            input_complexity: Input karmaşıklığı (0-1)
            time_constraint: Zaman kısıtı (0-1, 1=çok zaman var)
            
        Returns:
            Seçilen pattern
        """
        chars = ThinkingPatternLibrary.get_pattern_characteristics()
        
        # Scoring function
        scores = {}
        for pattern in ThinkingPattern:
            char = chars[pattern]
            score = 0.0
            
            # Complexity match
            if input_complexity > 0.7:
                score += char['depth'] * 0.4  # Derin düşünme gerekli
            else:
                score += char['speed'] * 0.4  # Hızlı düşünme yeterli
            
            # Time constraint
            if time_constraint < 0.3:
                score += char['speed'] * 0.4  # Hızlı olmalı
            else:
                score += char['depth'] * 0.4  # Derin olabilir
            
            # Task-specific
            recommended = self.thinking_system._recommend_pattern_for_task(task_type)
            if pattern == recommended:
                score += 0.2
            
            scores[pattern] = score
        
        # En yüksek skorlu pattern
        optimal = max(scores.items(), key=lambda x: x[1])[0]
        
        # Learning: Bu task için hangi pattern iyi çalıştı?
        if task_type not in self.task_pattern_map:
            self.task_pattern_map[task_type] = []
        self.task_pattern_map[task_type].append(optimal)
        
        return optimal


# Global instance
_global_thinking_system = None


def get_thinking_system(primary_pattern: ThinkingPattern = None,
                       pattern_weights: Dict[ThinkingPattern, float] = None) -> EvolvableThinkingSystem:
    """Global thinking system instance"""
    global _global_thinking_system
    if _global_thinking_system is None:
        _global_thinking_system = EvolvableThinkingSystem(
            primary_pattern=primary_pattern,
            pattern_weights=pattern_weights
        )
    return _global_thinking_system


