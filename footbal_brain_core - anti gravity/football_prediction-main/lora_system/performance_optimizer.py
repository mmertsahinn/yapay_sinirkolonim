"""
⚡ PERFORMANCE OPTIMIZER
=========================

Performans optimizasyonu: GPU desteği, batch processing, memory efficiency
"""

import torch
import torch.nn as nn
from typing import List, Optional, Any, Dict
import gc


class GPUOptimizer:
    """GPU optimizasyonu"""
    
    @staticmethod
    def move_to_device(model: nn.Module, device: str = 'cuda'):
        """
        Model'i device'a taşı
        
        Args:
            model: PyTorch model
            device: 'cuda' veya 'cpu'
        """
        if torch.cuda.is_available() and device == 'cuda':
            model = model.cuda()
            torch.cuda.empty_cache()
            return model
        else:
            return model.cpu()
    
    @staticmethod
    def optimize_for_inference(model: nn.Module):
        """
        Inference için optimize et
        
        Args:
            model: PyTorch model
        """
        model.eval()
        if hasattr(model, 'half'):  # FP16
            try:
                model = model.half()
            except:
                pass
        return model
    
    @staticmethod
    def clear_cache():
        """GPU cache'i temizle"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class BatchProcessor:
    """Batch processing için yardımcı sınıf"""
    
    @staticmethod
    def process_batch(lora_list: List[Any], 
                     batch_size: int = 32,
                     process_func = None) -> List[Any]:
        """
        LoRA listesini batch'ler halinde işle
        
        Args:
            lora_list: LoRA listesi
            batch_size: Batch boyutu
            process_func: İşleme fonksiyonu
            
        Returns:
            İşlenmiş liste
        """
        if process_func is None:
            return lora_list
        
        results = []
        
        for i in range(0, len(lora_list), batch_size):
            batch = lora_list[i:i+batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
            
            # Memory cleanup
            if (i // batch_size) % 10 == 0:
                gc.collect()
        
        return results


class MemoryOptimizer:
    """Memory optimizasyonu"""
    
    @staticmethod
    def clear_unused_loras(population: List[Any], keep_top_n: int = 100):
        """
        Kullanılmayan LoRA'ları temizle
        
        Args:
            population: LoRA popülasyonu
            keep_top_n: Kaç tanesi saklanacak
            
        Returns:
            Temizlenmiş popülasyon
        """
        # Fitness'e göre sırala
        if len(population) <= keep_top_n:
            return population
        
        population_with_fitness = []
        for lora in population:
            if hasattr(lora, 'fitness_history') and len(lora.fitness_history) > 0:
                fitness = sum(lora.fitness_history[-20:]) / len(lora.fitness_history[-20:]) if len(lora.fitness_history) >= 20 else lora.fitness_history[-1]
            else:
                fitness = 0.5
            
            population_with_fitness.append((lora, fitness))
        
        # En iyileri al
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        kept_population = [lora for lora, _ in population_with_fitness[:keep_top_n]]
        
        # Cleanup
        for lora, _ in population_with_fitness[keep_top_n:]:
            del lora
        gc.collect()
        
        return kept_population
    
    @staticmethod
    def optimize_model_memory(model: nn.Module):
        """
        Model memory'sini optimize et
        
        Args:
            model: PyTorch model
        """
        # Gradient checkpointing (eğitim için)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()


class PerformanceMonitor:
    """Performans izleme"""
    
    def __init__(self):
        self.stats = {
            'gpu_memory_used': [],
            'cpu_memory_used': [],
            'processing_times': []
        }
    
    def log_gpu_memory(self):
        """GPU memory kullanımını logla"""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024**2
            self.stats['gpu_memory_used'].append(memory_mb)
            return memory_mb
        return 0
    
    def get_statistics(self) -> Dict:
        """İstatistikleri döndür"""
        return {
            'avg_gpu_memory': sum(self.stats['gpu_memory_used']) / len(self.stats['gpu_memory_used']) if self.stats['gpu_memory_used'] else 0,
            'max_gpu_memory': max(self.stats['gpu_memory_used']) if self.stats['gpu_memory_used'] else 0,
            'avg_processing_time': sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        }


# Global instances
_gpu_optimizer = GPUOptimizer()
_batch_processor = BatchProcessor()
_memory_optimizer = MemoryOptimizer()
_performance_monitor = PerformanceMonitor()


def get_gpu_optimizer() -> GPUOptimizer:
    """Global GPU optimizer"""
    return _gpu_optimizer


def get_batch_processor() -> BatchProcessor:
    """Global batch processor"""
    return _batch_processor


def get_memory_optimizer() -> MemoryOptimizer:
    """Global memory optimizer"""
    return _memory_optimizer


def get_performance_monitor() -> PerformanceMonitor:
    """Global performance monitor"""
    return _performance_monitor


