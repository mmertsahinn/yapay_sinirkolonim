"""
📚 KNOWLEDGE DISTILLATION - KEŞİF YAYILIMI
===========================================

Teacher-Student paradigma ile keşifleri yayıyoruz!

Bilimsel temel: Knowledge Distillation (Hinton et al., 2015)
- Keşfeden LoRA = Teacher
- Öğrenen LoRAlar = Students
- Soft targets ile yumuşak bilgi transferi

Neden KD?
✅ Soft targets: Sadece "doğru/yanlış" değil, probability distribution
✅ Temperature: Ne kadar soft olacağını kontrol eder
✅ Dark knowledge: Teacher'ın gizli bilgisini transfer eder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscoveryDistillation(nn.Module):
    """
    Keşfi knowledge distillation ile yay
    
    Formula:
    L = α * L_hard + (1-α) * L_soft
    L_soft = KL(softmax(z_s/T), softmax(z_t/T))
    
    z_t = teacher logits
    z_s = student logits  
    T = temperature
    """
    
    def __init__(self, embed_dim=128, temperature=2.0):
        """
        Args:
            embed_dim: LoRA embedding boyutu
            temperature: Distillation temperature (default: 2.0)
                        Yüksek T = daha soft, daha genel bilgi
                        Düşük T = daha hard, daha spesifik bilgi
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # Teacher knowledge encoder
        self.teacher_encoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
        
        # Student adaptation layer
        self.student_adapter = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),  # Student emb + Teacher knowledge
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
        
        # Attention: Student decides how much to learn from teacher
        self.learning_attention = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        print(f"✅ DiscoveryDistillation initialized (T={temperature})")
    
    def distill_knowledge(self, teacher_emb, student_emb, 
                         temperature=None, return_attention=False):
        """
        Teacher'dan student'a bilgi transfer et
        
        Args:
            teacher_emb: Keşfeden LoRA embedding [embed_dim]
            student_emb: Öğrenen LoRA embedding [embed_dim]
            temperature: Override default temperature (optional)
            return_attention: Attention weight'i döndür mü?
            
        Returns:
            distilled: Student'ın yeni embedding'i [embed_dim]
            soft_targets: Teacher'ın soft targets'ı (optional)
            attention: Learning attention weight (optional)
        """
        T = temperature if temperature is not None else self.temperature
        
        # 1. Teacher'dan soft targets
        teacher_knowledge = self.teacher_encoder(teacher_emb)
        soft_targets = F.softmax(teacher_knowledge / T, dim=-1)
        
        # 2. Student ne kadar öğrenmek istiyor?
        combined = torch.cat([student_emb, soft_targets], dim=-1)
        learning_attn = self.learning_attention(combined)
        
        # 3. Student'ın adaptasyonu
        adapted = self.student_adapter(combined)
        
        # 4. Blend: Student'ın kendisi + Teacher'dan öğrendikleri
        distilled = (1 - learning_attn) * student_emb + learning_attn * adapted
        
        if return_attention:
            return distilled, soft_targets, learning_attn.item()
        else:
            return distilled, soft_targets
    
    def compute_distillation_loss(self, teacher_logits, student_logits, 
                                 labels=None, alpha=0.7):
        """
        Distillation loss hesapla
        
        Args:
            teacher_logits: Teacher'ın çıktıları
            student_logits: Student'ın çıktıları
            labels: Gerçek labels (optional, hard loss için)
            alpha: Soft loss weight (default: 0.7)
                  Loss = α×L_soft + (1-α)×L_hard
                  
        Returns:
            total_loss: Distillation loss
        """
        T = self.temperature
        
        # Soft loss (KL divergence)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        
        # Hard loss (if labels provided)
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        else:
            total_loss = soft_loss
        
        return total_loss


class MultiTeacherDistillation(nn.Module):
    """
    Birden fazla teacher'dan öğrenme
    
    Örnek: 3 farklı uzman LoRA'dan bilgi al
    Her uzmanın farklı ağırlığı var
    """
    
    def __init__(self, embed_dim=128, max_teachers=5):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_teachers = max_teachers
        
        # Teacher weight network: Hangi teacher'a ne kadar önem verilecek?
        self.teacher_weights = nn.Sequential(
            nn.Linear(embed_dim * (max_teachers + 1), 256),  # All teachers + student
            nn.ReLU(),
            nn.Linear(256, max_teachers),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
        
        # Aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
        print(f"✅ MultiTeacherDistillation initialized (max_teachers={max_teachers})")
    
    def forward(self, student_emb, teacher_embs):
        """
        Birden fazla teacher'dan öğren
        
        Args:
            student_emb: [embed_dim]
            teacher_embs: List of [embed_dim] tensors
            
        Returns:
            aggregated: Tüm teacher'lardan öğrenilmiş bilgi
            weights: Her teacher'ın ağırlığı
        """
        # Pad teachers if needed
        while len(teacher_embs) < self.max_teachers:
            teacher_embs.append(torch.zeros_like(student_emb))
        
        # Truncate if too many
        teacher_embs = teacher_embs[:self.max_teachers]
        
        # Concatenate all
        all_embs = torch.cat([student_emb] + teacher_embs, dim=-1)
        
        # Compute weights
        weights = self.teacher_weights(all_embs)
        
        # Weighted average of teachers
        teacher_stack = torch.stack(teacher_embs)  # [max_teachers x embed_dim]
        weighted_teachers = (weights.unsqueeze(-1) * teacher_stack).sum(dim=0)
        
        # Aggregate
        aggregated = self.aggregator(weighted_teachers)
        
        return aggregated, weights


# Global instance
_global_distillation = None
_global_multi_teacher = None


def get_distillation(embed_dim=128, temperature=2.0):
    """Global knowledge distillation instance"""
    global _global_distillation
    if _global_distillation is None:
        _global_distillation = DiscoveryDistillation(
            embed_dim=embed_dim,
            temperature=temperature
        )
    return _global_distillation


def get_multi_teacher_distillation(embed_dim=128):
    """Global multi-teacher distillation instance"""
    global _global_multi_teacher
    if _global_multi_teacher is None:
        _global_multi_teacher = MultiTeacherDistillation(embed_dim=embed_dim)
    return _global_multi_teacher
