"""
🧠 META-LoRA - Attention Mekanizması
====================================

Meta-LoRA: "Hangi LoRA'yı dinleyelim?" kararını verir.

Attention mekanizması:
- Her maç için her LoRA'ya dinamik ağırlık verir
- En uygun uzmanları devreye sokar
- Ensemble gibi ama LoRA'lar üzerinde
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

from .lora_adapter import LoRAAdapter


class MetaLoRA(nn.Module):
    """
    Meta-LoRA: LoRA popülasyonunu yöneten üst akıl
    """
    
    def __init__(self, input_dim: int = 63, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Query network: Maç özelliklerinden query vektörü üretir
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 16 boyutlu query
        )
        
        # Key network: Her LoRA'nın key vektörünü tutar (öğrenilecek)
        # Bu dinamik olduğu için forward'da hesaplanacak
        
        self.attention_dim = 16
    
    def get_lora_key(self, lora: LoRAAdapter, device='cpu') -> torch.Tensor:
        """
        Her LoRA'nın 'key' vektörünü üret
        Bu, LoRA'nın özelliklerini/uzmanlığını temsil eder
        """
        # LoRA'nın parametrelerinden basit bir özellik vektörü çıkar
        params = lora.get_all_lora_params()
        
        # Her katmandan ortalama değerleri al
        features = []
        for layer in ['fc1', 'fc2', 'fc3']:
            for matrix in ['lora_A', 'lora_B']:
                param = params[layer][matrix].to(device)
                # İstatistikler: mean, std, min, max
                features.extend([
                    param.mean().item(),
                    param.std().item()
                ])
        
        # 12 özellik var (3 layer * 2 matrix * 2 stat)
        # 16 boyuta pad et
        while len(features) < self.attention_dim:
            features.append(0.0)
        
        key = torch.tensor(features[:self.attention_dim], dtype=torch.float32, device=device)
        return key
    
    def forward(self, match_features: torch.Tensor, lora_population: List[LoRAAdapter], device='cpu'):
        """
        Attention mekanizmasıyla LoRA'ları ağırlıklandır
        
        Args:
            match_features: (batch_size, 61) maç özellikleri
            lora_population: LoRA listesi
        
        Returns:
            attention_weights: (batch_size, num_loras) ağırlıklar
        """
        # Query: Maç özelliklerinden
        query = self.query_net(match_features)  # (batch_size, 16)
        
        # Keys: Her LoRA'dan
        keys = []
        for lora in lora_population:
            key = self.get_lora_key(lora, device)
            keys.append(key)
        
        keys = torch.stack(keys)  # (num_loras, 16)
        
        # Attention scores: query @ keys^T
        # (batch_size, 16) @ (16, num_loras) = (batch_size, num_loras)
        scores = torch.matmul(query, keys.T)
        
        # Softmax → ağırlıklar
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights
    
    def aggregate_predictions(
        self,
        match_features: np.ndarray,
        base_proba: np.ndarray,
        lora_population: List[LoRAAdapter],
        device='cpu'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Tüm LoRA'lardan tahmin al ve attention ile birleştir
        
        Args:
            match_features: (78,) = 60 base + 15 historical + 3 base_proba (zaten birleşik!)
            base_proba: (3,) sadece bilgi için
        
        Returns:
            aggregated_proba: (3,) nihai tahmin
            info: Detaylı bilgi
        """
        if len(lora_population) == 0:
            # LoRA yoksa base proba'yı döndür
            return base_proba, {'attention_weights': [], 'individual_probas': []}
        
        # Match features tensor (78 boyut)
        x = torch.from_numpy(match_features).unsqueeze(0).float().to(device)
        
        # Attention weights hesapla
        with torch.no_grad():
            attention_weights = self.forward(x, lora_population, device)  # (1, num_loras)
            attention_weights = attention_weights.squeeze(0).cpu().numpy()  # (num_loras,)
        
        # Her LoRA'dan tahmin al
        # LoRA.predict içinde zaten concat yapıyor: lora_features (75) + base_proba (3) = 78
        individual_probas = []
        lora_features = match_features[:75]  # İlk 75 feature (60 base + 15 historical)
        
        for lora in lora_population:
            lora_proba = lora.predict(lora_features, base_proba, device)
            individual_probas.append(lora_proba)
        
        individual_probas = np.array(individual_probas)  # (num_loras, 3)
        
        # Weighted average
        aggregated_proba = np.sum(individual_probas * attention_weights[:, None], axis=0)
        
        # Normalize (güvenlik)
        aggregated_proba = aggregated_proba / aggregated_proba.sum()
        
        info = {
            'attention_weights': attention_weights,
            'individual_probas': individual_probas,
            'num_loras': len(lora_population)
        }
        
        return aggregated_proba, info
    
    def get_top_loras(
        self,
        match_features: np.ndarray,
        lora_population: List[LoRAAdapter],
        top_k: int = 5,
        device='cpu'
    ) -> List[Tuple[LoRAAdapter, float]]:
        """
        Bu maç için en yüksek attention alan top-K LoRA'ları döndür
        """
        if len(lora_population) == 0:
            return []
        
        x = torch.from_numpy(match_features).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            attention_weights = self.forward(x, lora_population, device)
            attention_weights = attention_weights.squeeze(0).cpu().numpy()
        
        # Top-K indeks
        top_indices = np.argsort(attention_weights)[::-1][:top_k]
        
        top_loras = [(lora_population[i], attention_weights[i]) for i in top_indices]
        
        return top_loras


class SimpleMetaLoRA:
    """
    Basitleştirilmiş Meta-LoRA (PyTorch olmadan)
    Fitness bazlı ağırlıklandırma
    """
    
    def __init__(self):
        self.name = "SimpleMetaLoRA"
    
    def aggregate_predictions(
        self,
        match_features: np.ndarray,
        base_proba: np.ndarray,
        lora_population: List[LoRAAdapter],
        device='cpu'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fitness bazlı ağırlıklandırma
        
        Args:
            match_features: (78,) = 60 base + 15 historical + 3 base_proba
            base_proba: (3,) sadece bilgi için
        """
        if len(lora_population) == 0:
            return base_proba, {'attention_weights': [], 'individual_probas': []}
        
        # Her LoRA'dan tahmin al
        # LoRA.predict içinde base_proba tekrar concat yapılacak, o yüzden sadece lora_features gönder
        individual_probas = []
        fitnesses = []
        lora_features = match_features[:75]  # İlk 75 feature (60 base + 15 historical)
        
        for lora in lora_population:
            lora_proba = lora.predict(lora_features, base_proba, device)
            individual_probas.append(lora_proba)
            fitnesses.append(lora.get_recent_fitness())
        
        individual_probas = np.array(individual_probas)  # (num_loras, 3)
        fitnesses = np.array(fitnesses)  # (num_loras,)
        
        # Fitness'i ağırlık olarak kullan (softmax)
        fitnesses = np.clip(fitnesses, 0.01, 1.0)  # Negatif olmasın
        weights = np.exp(fitnesses * 5)  # 5: scaling factor
        weights = weights / weights.sum()
        
        # Weighted average
        aggregated_proba = np.sum(individual_probas * weights[:, None], axis=0)
        
        # Normalize
        aggregated_proba = aggregated_proba / aggregated_proba.sum()
        
        info = {
            'attention_weights': weights,
            'individual_probas': individual_probas,
            'num_loras': len(lora_population)
        }
        
        return aggregated_proba, info
    
    def get_top_loras(
        self,
        match_features: np.ndarray,
        lora_population: List[LoRAAdapter],
        top_k: int = 5,
        device='cpu'
    ) -> List[Tuple[LoRAAdapter, float]]:
        """
        Fitness bazlı top-K
        """
        if len(lora_population) == 0:
            return []
        
        sorted_loras = sorted(lora_population, key=lambda x: x.get_recent_fitness(), reverse=True)
        
        return [(lora, lora.get_recent_fitness()) for lora in sorted_loras[:top_k]]

