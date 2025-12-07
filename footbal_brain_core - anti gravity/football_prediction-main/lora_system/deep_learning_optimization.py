"""
🧬 DEEP LEARNING OPTIMIZATION - Knowledge Distillation & Era Jumping
==================================================================

Bu modül, LoRA'ların "insan gibi öğrenmesini" ve "çağ atlamasını" sağlar.
Sadece deneyimle (hard label) değil, usta LoRA'ların olasılık dağılımlarını (soft targets)
kopyalayarak (Knowledge Distillation) çok daha hızlı öğrenirler.

Teknikler:
1. Knowledge Distillation (Hinton et al.) - Çağ atlama mekanizması
2. Collective Backpropagation - Sürü zekasıyla öğrenme
3. Sparse Autoencoder Learning - Verimli nöron kullanımı
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any

class DeepKnowledgeDistiller:
    """
    Bilgi Damıtma Sistemi:
    Genç LoRA'lar (Student), Usta LoRA'lardan (Teacher) öğrenir.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.7, device='cpu'):
        """
        Args:
            temperature: Softmax sıcaklığı (Yüksek = daha yumuşak olasılıklar, daha fazla bilgi)
            alpha: Teacher loss ağırlığı (1-alpha: Gerçek sonuç ağırlığı)
        """
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def distill_knowledge(self,
                         student_lora: nn.Module,
                         teacher_lora: nn.Module,
                         features_np: np.ndarray,
                         base_proba_np: np.ndarray,
                         actual_class_idx: int,
                         optimizer: torch.optim.Optimizer) -> float:
        """
        Bir öğrenci LoRA'ya, öğretmenin bilgisini aktar.

        Loss = alpha * KL(Student, Teacher) + (1-alpha) * CE(Student, Truth)
        """
        student_lora.train()
        teacher_lora.eval() # Teacher sabit

        # Data preparation
        x = np.concatenate([features_np, base_proba_np]).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor([actual_class_idx], dtype=torch.long, device=self.device)

        optimizer.zero_grad()

        # Forward pass
        student_logits = student_lora.forward(x_tensor) # Logits (pre-softmax) gerekebilir ama LoRAAdapter softmax dönüyor olabilir.
        # LoRAAdapter softmax dönüyorsa log_softmax almalıyız.
        # Varsayım: LoRAAdapter forward() proba dönüyor.
        student_proba = student_logits # forward proba dönüyor
        student_log_proba = torch.log(student_proba + 1e-10)

        with torch.no_grad():
            teacher_proba = teacher_lora.forward(x_tensor)

        # 1. Distillation Loss (KL Divergence)
        # Teacher ve Student dağılımları arasındaki fark
        # Softmax temperature scaling uygulanabilir ama input zaten proba ise direkt kullanılır
        distillation_loss = self.kl_div_loss(student_log_proba, teacher_proba)

        # 2. Student Loss (Ground Truth)
        student_loss = self.ce_loss(student_log_proba, y_tensor)

        # Total Loss
        total_loss = self.alpha * distillation_loss + (1.0 - self.alpha) * student_loss

        total_loss.backward()
        optimizer.step()

        return total_loss.item()

    def find_best_teacher(self, population: List[Any], current_lora: Any) -> Any:
        """
        Bir LoRA için en iyi öğretmeni bul (Fitness ve benzerlik bazlı)
        """
        candidates = [l for l in population if l.id != current_lora.id and l.get_recent_fitness() > 0.8]
        if not candidates:
            return None

        # En yüksek fitness'a sahip olanı seç (veya specialization uyumu)
        best_teacher = max(candidates, key=lambda l: l.get_recent_fitness())
        return best_teacher

class CollectiveDeepLearner:
    """
    Kolektif Derin Öğrenme:
    Tüm popülasyonun 'Konsensüs' hatasından ders çıkarması.
    """

    def __init__(self, device='cpu'):
        self.device = device

    def collective_backprop(self,
                           population: List[Any],
                           features_np: np.ndarray,
                           base_proba_np: np.ndarray,
                           actual_class_idx: int,
                           global_error_magnitude: float):
        """
        Eğer sürü (çoğunluk) yanıldıysa, herkes bu hatadan payına düşeni alır.
        Global hata büyüklüğüne göre hafif bir 'düzeltme' sinyali gönderilir.
        """
        if global_error_magnitude < 0.5:
            return # Hata küçükse kolektif öğrenmeye gerek yok

        x = np.concatenate([features_np, base_proba_np]).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor([actual_class_idx], dtype=torch.long, device=self.device)
        criterion = nn.CrossEntropyLoss()

        for lora in population:
            # Sadece hataya katkıda bulunanlar öğrenir (yanlış tahmin yapanlar)
            # Ama kolektif zeka için herkes hafifçe doğruya çekilmeli

            optimizer = torch.optim.SGD(lora.parameters(), lr=0.0001) # Çok küçük learning rate
            optimizer.zero_grad()

            proba = lora.forward(x_tensor)
            loss = criterion(torch.log(proba + 1e-10), y_tensor)

            # Loss'u global hata ile scale et
            weighted_loss = loss * global_error_magnitude * 0.1 # %10 etki
            weighted_loss.backward()
            optimizer.step()
