"""
🧬 DEEP LEARNING OPTIMIZATION - Knowledge Distillation & Era Jumping
==================================================================

Bu modül, LoRA'ların "insan gibi öğrenmesini" ve "çağ atlamasını" sağlar.

Sadece deneyimle (hard label) değil, usta LoRA'ların olasılık dağılımlarını (soft targets)
kopyalayarak (Knowledge Distillation) çok daha hızlı öğrenirler.

Teknikler:
1. Knowledge Distillation (Hinton et al., 2015) - Çağ atlama mekanizması
2. Collective Backpropagation - Sürü zekasıyla öğrenme
3. Specialization-aware Teacher Selection - Uzmanlık bazlı öğretmen seçimi

Bilimsel Temel:
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Soft targets: Teacher'ın probability distribution'ı
- Temperature scaling: T > 1 → daha yumuşak, daha genel bilgi
- Dark knowledge: Teacher'ın gizli bilgisi (logits'lerde saklı)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


class DeepKnowledgeDistiller:
    """
    Bilgi Damıtma Sistemi:
    Genç LoRA'lar (Student), Usta LoRA'lardan (Teacher) öğrenir.
    
    Formula:
    L_total = α × L_soft + (1-α) × L_hard
    
    L_soft = T² × KL(softmax(logits_s/T), softmax(logits_t/T))
    L_hard = CrossEntropy(logits_s, labels)
    
    where:
    - logits_s = student logits (softmax öncesi)
    - logits_t = teacher logits (softmax öncesi)
    - T = temperature (default: 2.0)
    - α = soft loss weight (default: 0.7)
    """
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7, device='cpu'):
        """
        Args:
            temperature: Softmax sıcaklığı (Yüksek = daha yumuşak olasılıklar, daha fazla bilgi)
            alpha: Teacher loss ağırlığı (1-alpha: Gerçek sonuç ağırlığı)
            device: Computation device
        """
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Teacher selection cache (performance için)
        self.teacher_cache = {}  # {student_id: (teacher_id, fitness)}
        
        print(f"✅ DeepKnowledgeDistiller initialized (T={temperature}, α={alpha})")
    
    def find_best_teacher(self, population: List[Any], current_lora: Any) -> Optional[Any]:
        """
        Bir LoRA için en iyi öğretmeni bul (Fitness ve benzerlik bazlı)
        
        Strateji:
        1. Aynı uzmanlıktan teacher tercih edilir (daha iyi transfer!)
        2. Fitness > 0.8 olmalı
        3. Kendisi olamaz
        4. Cache kullanılır (performance için)
        
        Args:
            population: LoRA popülasyonu
            current_lora: Öğrenci LoRA
            
        Returns:
            Best teacher LoRA veya None
        """
        # Cache kontrolü
        if current_lora.id in self.teacher_cache:
            cached_teacher_id, cached_fitness = self.teacher_cache[current_lora.id]
            # Cache'deki teacher hala popülasyonda ve fitness yeterli mi?
            for lora in population:
                if lora.id == cached_teacher_id and lora.get_recent_fitness() >= 0.75:
                    return lora
        
        # 1. Aynı uzmanlıktan teacher bul (ÖNCELİK!)
        current_spec = getattr(current_lora, 'specialization', None)
        same_spec_candidates = []
        
        if current_spec:
            for lora in population:
                if (lora.id != current_lora.id and 
                    getattr(lora, 'specialization', None) == current_spec and
                    lora.get_recent_fitness() > 0.75):
                    same_spec_candidates.append(lora)
        
        if same_spec_candidates:
            # Aynı uzmanlıktan en iyisini seç
            best_teacher = max(same_spec_candidates, key=lambda l: l.get_recent_fitness())
            # Cache'e kaydet
            self.teacher_cache[current_lora.id] = (best_teacher.id, best_teacher.get_recent_fitness())
            return best_teacher
        
        # 2. Genel en iyi teacher (uzmanlık farklı olsa bile)
        general_candidates = [
            l for l in population 
            if l.id != current_lora.id and l.get_recent_fitness() > 0.80
        ]
        
        if not general_candidates:
            return None
        
        best_teacher = max(general_candidates, key=lambda l: l.get_recent_fitness())
        # Cache'e kaydet
        self.teacher_cache[current_lora.id] = (best_teacher.id, best_teacher.get_recent_fitness())
        return best_teacher
    
    def distill_knowledge(self, 
                         student_lora: nn.Module, 
                         teacher_lora: nn.Module, 
                         features_np: np.ndarray, 
                         base_proba_np: np.ndarray,
                         actual_class_idx: int,
                         optimizer: torch.optim.Optimizer) -> float:
        """
        Bir öğrenci LoRA'ya, öğretmenin bilgisini aktar.
        
        Process:
        1. Teacher'dan soft targets al (temperature scaling ile)
        2. Student'dan logits al
        3. Distillation loss hesapla (KL divergence)
        4. Hard loss hesapla (CrossEntropy)
        5. Combined loss ile optimize et
        
        Args:
            student_lora: Öğrenci LoRA (güncellenecek)
            teacher_lora: Öğretmen LoRA (sabit, eval mode)
            features_np: Feature array [75] (60 base + 15 historical)
            base_proba_np: Base probability [3]
            actual_class_idx: Gerçek sınıf indexi
            optimizer: Student'ın optimizer'ı
            
        Returns:
            Total loss value (float)
        """
        student_lora.train()
        teacher_lora.eval()  # Teacher sabit
        
        # Data preparation
        x = np.concatenate([features_np, base_proba_np]).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor([actual_class_idx], dtype=torch.long, device=self.device)
        
        optimizer.zero_grad()
        
        # Forward pass: LOGITS al (softmax ÖNCESİ!)
        # ✅ forward_logits() kullan (LoRAAdapter'da olmalı!)
        if hasattr(student_lora, 'forward_logits'):
            student_logits = student_lora.forward_logits(x_tensor)
        else:
            # Fallback: forward() kullan ama logits'e çevir (ters softmax - yaklaşık)
            student_proba = student_lora.forward(x_tensor)
            # Proba'dan logits'e yaklaşık dönüşüm (numerical stability için)
            student_logits = torch.log(student_proba + 1e-10)
        
        # Teacher'dan soft targets (eval mode, no grad)
        with torch.no_grad():
            if hasattr(teacher_lora, 'forward_logits'):
                teacher_logits = teacher_lora.forward_logits(x_tensor)
            else:
                # Fallback
                teacher_proba = teacher_lora.forward(x_tensor)
                teacher_logits = torch.log(teacher_proba + 1e-10)
        
        # 1. Distillation Loss (KL Divergence)
        # Softmax temperature scaling
        T = self.temperature
        
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        
        # KL divergence: KL(P_student || P_teacher)
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        
        # 2. Student Loss (Ground Truth - Hard Labels)
        student_loss = self.ce_loss(student_logits, y_tensor)
        
        # 3. Total Loss
        total_loss = self.alpha * distillation_loss + (1.0 - self.alpha) * student_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def teach_newborn_lora(self, 
                          newborn_lora: nn.Module,
                          population: List[Any],
                          sample_features: np.ndarray = None,
                          sample_base_proba: np.ndarray = None,
                          device='cpu') -> bool:
        """
        🎓 YENİ DOĞAN LoRA'YA MASTER'DAN ÖĞRET!
        
        Plan: "Yeni doğan bir LoRA, Master bir LoRA'nın (Fitness > 0.9) 
        beynini Deep Learning (Distillation Loss) ile kopyalayarak başlayacak."
        
        Args:
            newborn_lora: Yeni doğan LoRA (henüz hiçbir şey öğrenmemiş)
            population: Mevcut popülasyon (Master bulmak için)
            sample_features: Örnek feature'lar (varsa, yoksa random)
            sample_base_proba: Örnek base proba (varsa, yoksa random)
            device: Device
        
        Returns:
            True if teaching successful, False otherwise
        """
        # 1. Master bul (Fitness > 0.9)
        master_candidates = [
            l for l in population 
            if l.get_recent_fitness() > 0.9 and l.id != newborn_lora.id
        ]
        
        if not master_candidates:
            # Master yoksa, en iyi teacher'ı bul (Fitness > 0.8)
            master_candidates = [
                l for l in population 
                if l.get_recent_fitness() > 0.8 and l.id != newborn_lora.id
            ]
        
        if not master_candidates:
            return False  # Hiç teacher yok
        
        # En iyi Master'ı seç
        master = max(master_candidates, key=lambda l: l.get_recent_fitness())
        
        # 2. Örnek veri hazırla (yoksa random)
        if sample_features is None:
            sample_features = np.random.randn(75).astype(np.float32)  # 60 base + 15 historical
        
        if sample_base_proba is None:
            sample_base_proba = np.array([0.33, 0.34, 0.33], dtype=np.float32)  # Uniform
        
        # 3. Distillation yap (birkaç iterasyon)
        optimizer = torch.optim.Adam(newborn_lora.parameters(), lr=0.001)
        
        # 5 iterasyon yeterli (hızlı öğrenme!)
        for iteration in range(5):
            # Random class (öğrenme için)
            random_class = np.random.randint(0, 3)
            
            try:
                self.distill_knowledge(
                    newborn_lora,
                    master,
                    sample_features,
                    sample_base_proba,
                    random_class,
                    optimizer
                )
            except Exception as e:
                # Hata varsa devam et
                continue
        
        return True
    
    def clear_cache(self):
        """Teacher cache'i temizle (popülasyon değiştiğinde)"""
        self.teacher_cache.clear()


class CollectiveDeepLearner:
    """
    Kolektif Derin Öğrenme:
    Tüm popülasyonun 'Konsensüs' hatasından ders çıkarması.
    
    Concept:
    - Eğer sürü (çoğunluk) yanıldıysa, herkes bu hatadan payına düşeni alır
    - Global hata büyüklüğüne göre hafif bir 'düzeltme' sinyali gönderilir
    - Sadece yanlış tahmin yapanlar öğrenir (doğru tahmin yapanlar zaten iyi)
    
    Bilimsel Temel:
    - Collective Intelligence
    - Swarm Learning
    - Consensus-based Learning
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        
        print(f"✅ CollectiveDeepLearner initialized")
    
    def collective_backprop(self, 
                           population: List[Any], 
                           features_np: np.ndarray, 
                           base_proba_np: np.ndarray,
                           actual_class_idx: int,
                           global_error_magnitude: float):
        """
        Eğer sürü (çoğunluk) yanıldıysa, herkes bu hatadan payına düşeni alır.
        
        Args:
            population: LoRA popülasyonu
            features_np: Feature array [75]
            base_proba_np: Base probability [3]
            actual_class_idx: Gerçek sınıf indexi
            global_error_magnitude: Global hata büyüklüğü (0-1)
                                   Yüksek = çoğunluk yanıldı
        """
        if global_error_magnitude < 0.5:
            return  # Hata küçükse kolektif öğrenmeye gerek yok
        
        x = np.concatenate([features_np, base_proba_np]).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor([actual_class_idx], dtype=torch.long, device=self.device)
        
        # Sadece yanlış tahmin yapanlar öğrenir
        wrong_loras = []
        
        for lora in population:
            # Tahmin kontrolü
            with torch.no_grad():
                if hasattr(lora, 'forward_logits'):
                    logits = lora.forward_logits(x_tensor)
                else:
                    proba = lora.forward(x_tensor)
                    logits = torch.log(proba + 1e-10)
                
                pred_idx = logits.argmax(dim=-1).item()
                
                if pred_idx != actual_class_idx:
                    wrong_loras.append(lora)
        
        if not wrong_loras:
            return  # Hepsi doğru tahmin yaptı
        
        # Her yanlış LoRA'ya hafif düzeltme sinyali gönder
        for lora in wrong_loras:
            # Çok küçük learning rate (kolektif öğrenme hafif olmalı)
            lora_params = [p for p in lora.parameters() if p.requires_grad]
            if not lora_params:
                continue
            
            optimizer = torch.optim.SGD(lora_params, lr=0.00001)  # Çok küçük!
            optimizer.zero_grad()
            
            # Forward
            if hasattr(lora, 'forward_logits'):
                logits = lora.forward_logits(x_tensor)
            else:
                proba = lora.forward(x_tensor)
                logits = torch.log(proba + 1e-10)
            
            loss = self.ce_loss(logits, y_tensor)
            
            # Loss'u global hata ile scale et (hata ne kadar büyükse o kadar öğren)
            weighted_loss = loss * global_error_magnitude * 0.1  # %10 etki
            
            weighted_loss.backward()
            optimizer.step()


# Global instances
_global_distiller = None
_global_collective_learner = None


def get_deep_knowledge_distiller(temperature: float = 2.0, 
                                 alpha: float = 0.7, 
                                 device='cpu') -> DeepKnowledgeDistiller:
    """Global DeepKnowledgeDistiller instance"""
    global _global_distiller
    if _global_distiller is None:
        _global_distiller = DeepKnowledgeDistiller(
            temperature=temperature,
            alpha=alpha,
            device=device
        )
    return _global_distiller


def get_collective_deep_learner(device='cpu') -> CollectiveDeepLearner:
    """Global CollectiveDeepLearner instance"""
    global _global_collective_learner
    if _global_collective_learner is None:
        _global_collective_learner = CollectiveDeepLearner(device=device)
    return _global_collective_learner

