"""
🧬 META-LEARNING - KÜMÜLATİF EVRİM
==================================

"Learning to learn" - Her nesil nasıl öğrenileceğini öğreniyor!

Bilimsel temel: MAML (Model-Agnostic Meta-Learning, Finn et al., 2017)
- Meta-parameters: "Nasıl öğren" bilgisi
- Fast adaptation: Yeni görevlere hızlı uyum
- Few-shot learning: Az örnekle öğrenme

Neden Meta-Learning?
✅ Kümülati

f: Her nesil önceki nesillerin deneyiminden yararlanır
✅ Fast adaptation: Yeni LoRAlar hızlı öğrenir
✅ Transfer learning: Bilgi jenerasyonlar arası aktarılır
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeMetaLearner(nn.Module):
    """
    Meta-learning for cumulative evolution
    
    Çocuk ebeveynin "nasıl öğrenilir" bilgisini miras alır
    
    MAML-inspired approach:
    θ'_child = θ_parent - α∇L(θ_parent)
    """
    
    def __init__(self, embed_dim=128, meta_lr=0.01):
        """
        Args:
            embed_dim: LoRA embedding boyutu
            meta_lr: Meta-learning rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.meta_lr = meta_lr
        
        # Meta-parameters: "Nasıl öğren" bilgisi
        # Bu parametre nesiller arası aktarılacak
        self.meta_params = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        
        # Adaptation network: Çocuk ebeveynden nasıl uyarlanacak
        self.adaptation_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),  # Parent + child
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # Learning strategy encoder
        self.strategy_encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )
        
        # Meta-knowledge gate: Ne kadar meta-bilgi transfer edilecek
        self.meta_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        print(f"✅ CumulativeMetaLearner initialized (meta_lr={meta_lr})")
    
    def inherit_from_parent(self, parent_emb, child_emb, generation_gap=1):
        """
        Ebeveynin meta-bilgisini çocuğa aktar
        
        Args:
            parent_emb: Ebeveynin embedding'i [embed_dim]
            child_emb: Çocuğun başlangıç embedding'i [embed_dim]
            generation_gap: Kaç nesil fark var (1=direkt çocuk, 2=torun...)
            
        Returns:
            enhanced_child: Meta-bilgi ile güçlendirilmiş çocuk [embed_dim]
            inheritance_strength: Ne kadar miras aldı (0-1)
        """
        # 1. Ebeveynin öğrenme stratejisini encode et
        parent_strategy = self.strategy_encoder(parent_emb)
        
        # 2. Meta-knowledge hesapla
        # θ'_meta = W_meta × parent_emb
        meta_knowledge = torch.matmul(parent_emb, self.meta_params)
        
        # 3. Ne kadar meta-bilgi transfer edilecek?
        # Yakın nesiller daha fazla alır
        base_gate = self.meta_gate(child_emb)
        generation_decay = 0.7 ** (generation_gap - 1)  # Her nesil %70
        inheritance_gate = base_gate * generation_decay
        
        # 4. Çocuğun adaptasyonu
        combined = torch.cat([child_emb, meta_knowledge], dim=-1)
        adapted = self.adaptation_net(combined)
        
        # 5. Final blend
        # Çocuk = kendi + (gate × ebeveynin meta-bilgisi)
        enhanced_child = child_emb + inheritance_gate * adapted
        
        return enhanced_child, inheritance_gate.item()
    
    def multi_generation_inheritance(self, ancestor_embs, child_emb):
        """
        Birden fazla atadan miras al
        
        Args:
            ancestor_embs: List of ancestor embeddings [parent, grandparent, ...]
            child_emb: Çocuğun embedding'i
            
        Returns:
            enhanced: Tüm atalardan bilgi almış çocuk
            contributions: Her atanın katkısı
        """
        current_emb = child_emb
        contributions = []
        
        for idx, ancestor_emb in enumerate(ancestor_embs):
            generation_gap = idx + 1  # 1=parent, 2=grandparent, ...
            
            enhanced, strength = self.inherit_from_parent(
                ancestor_emb,
                current_emb,
                generation_gap
            )
            
            current_emb = enhanced
            contributions.append({
                'generation': generation_gap,
                'strength': strength
            })
        
        return current_emb, contributions
    
    def update_meta_params(self, parent_embs, child_embs, performance_gains):
        """
        Meta-parametreleri güncelle
        
        Başarılı inheritance'lardan öğren
        
        Args:
            parent_embs: [N x embed_dim] Ebeveyn embeddings
            child_embs: [N x embed_dim] Çocuk embeddings  
            performance_gains: [N] Çocukların performans artışları
            
        Returns:
            meta_loss: Meta-learning loss
        """
        # Meta-bilgi ile enhance edilmiş çocuklar
        enhanced_children = []
        for p_emb, c_emb in zip(parent_embs, child_embs):
            enhanced, _ = self.inherit_from_parent(p_emb, c_emb)
            enhanced_children.append(enhanced)
        
        enhanced_children = torch.stack(enhanced_children)
        
        # Performans gains'i kullanarak loss hesapla
        # İyi performans gösteren inheritance'ları ödüllendir
        weights = F.softmax(performance_gains, dim=0)
        
        # Cosine similarity loss (enhanced children should be similar to parents)
        similarities = F.cosine_similarity(
            enhanced_children,
            parent_embs,
            dim=-1
        )
        
        # Weighted loss: Başarılı inheritance'lara daha fazla ağırlık
        meta_loss = -(weights * similarities).sum()
        
        return meta_loss


class GenerationalMemory(nn.Module):
    """
    Nesiller arası hafıza
    
    Her neslin collective experience'ı saklanır
    Yeni nesilstartpage Copilot GPT Gemini Perplexity e-mail Translate burayı kullanabilir
    """
    
    def __init__(self, embed_dim=128, max_generations=10):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_generations = max_generations
        
        # Generational memory bank
        self.memory_bank = nn.Parameter(
            torch.zeros(max_generations, embed_dim)
        )
        
        # Attention over generations
        self.generation_attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, max_generations),
            nn.Softmax(dim=-1)
        )
        
        print(f"✅ GenerationalMemory initialized ({max_generations} generations)")
    
    def store_generation(self, generation_idx, collective_emb):
        """Bir neslin collective embedding'ini sakla"""
        if generation_idx < self.max_generations:
            with torch.no_grad():
                self.memory_bank[generation_idx] = collective_emb
    
    def retrieve_ancestral_knowledge(self, child_emb, current_generation):
        """
        Ataların bilgisini getir
        
        Attention mechanism ile hangi nesillerin bilgisi daha önemli?
        """
        # Which generations to attend to?
        attn_weights = self.generation_attention(child_emb)
        
        # Only attend to ancestors (not future generations)
        mask = torch.zeros(self.max_generations)
        mask[:current_generation] = 1.0
        attn_weights = attn_weights * mask
        attn_weights = attn_weights / (attn_weights.sum() + 1e-8)
        
        # Weighted sum of ancestral knowledge
        ancestral_knowledge = (attn_weights.unsqueeze(-1) * self.memory_bank).sum(dim=0)
        
        return ancestral_knowledge, attn_weights


# Global instances
_global_meta_learner = None
_global_generational_memory = None


def get_meta_learner(embed_dim=128):
    """Global meta-learner instance"""
    global _global_meta_learner
    if _global_meta_learner is None:
        _global_meta_learner = CumulativeMetaLearner(embed_dim=embed_dim)
    return _global_meta_learner


def get_generational_memory(embed_dim=128):
    """Global generational memory instance"""
    global _global_generational_memory
    if _global_generational_memory is None:
        _global_generational_memory = GenerationalMemory(embed_dim=embed_dim)
    return _global_generational_memory
