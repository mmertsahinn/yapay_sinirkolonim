"""
🔥 TRIBE TRAINER - Kabile Bazlı Toplu Eğitim
============================================

Elek sisteminin (Sieve) bulduğu kabileleri topluca eğitir.
Her kabilenin bir "Lideri" (Chieftain) seçilir.
Kabile üyeleri, Lider'in bilgeliğini (Knowledge Distillation) kopyalar.

Bu sayede "aynı hatayı yapanlar", aralarındaki "doğruyu yapan" kişiden ders alır.
"""

import torch
import numpy as np
from typing import Dict, List, Any

class TribeTrainer:
    """
    Kabile Eğitmeni
    """
    
    def __init__(self, distiller, device='cpu'):
        self.distiller = distiller
        self.device = device
        
    def train_tribes(self, tribes: Dict[int, List[Any]], replay_buffer):
        """
        Her kabile için toplu eğitim uygula.
        
        1. Kabile Liderini Seç (En yüksek fitness)
        2. Buffer'dan örneklem al
        3. Tüm kabile üyelerini Lider'e benzet (Distillation)
        """
        if not tribes:
            return
            
        print(f"\n🔥 TRIBE TRAINING: {len(tribes)} kabile kampta...")
        
        # Buffer'dan eğitim verisi al (Son 32 maç veya önemli anlar)
        batch = replay_buffer.sample(batch_size=32)
        if not batch:
            print("   ⚠️ Buffer boş, eğitim yapılamadı.")
            return
            
        # Veriyi hazırla
        features_np = np.stack([b['features'] for b in batch])
        base_proba_np = np.stack([b['base_proba'] for b in batch])
        # Actual class idx (Hard target için gerekirse)
        # actual_indices = torch.tensor([b['actual_class_idx'] for b in batch], device=self.device)
        
        # Her kabile için döngü
        for cluster_id, members in tribes.items():
            if len(members) < 2:
                continue # Tek kişilik kabilede eğitim olmaz
                
            # 1. Lideri Seç (Chieftain)
            chieftain = max(members, key=lambda l: l.get_recent_fitness())
            
            # Eğer lider bile başarısızsa (fitness < 0.5), dışarıdan (global elit) bir mentor ata?
            # Şimdilik sadece kabile içi.
            
            print(f"   ⛺ Kabile #{cluster_id} (N={len(members)}): Lider {chieftain.name} ({chieftain.get_recent_fitness():.2f})")
            
            # Liderin çıktılarını al (Soft Targets)
            chieftain.eval() # Gradient yok
            with torch.no_grad():
                # Input hazırlığı (Toplu)
                # LoRA predict methodu tekil çalışıyor, burada batch işlem lazım.
                # Manuel forward yapalım.
                x_input = np.concatenate([features_np, base_proba_np], axis=1).astype(np.float32)
                x_tensor = torch.from_numpy(x_input).to(self.device)
                
                teacher_logits = chieftain.forward(x_tensor)
                teacher_probs = torch.softmax(teacher_logits, dim=-1) # Veya zaten prob dönüyorsa direkt
            
            # 2. Üyeleri Eğit (Intra-Tribe Distillation)
            train_count = 0
            for student in members:
                if student.id == chieftain.id:
                    continue # Lider kendini eğitmez
                    
                # Öğrenciyi eğit
                # Loss = KL(Student, Chieftain)
                # Sadece distillation loss (Hard target yok, çünkü amaç lideri taklit etmek)
                
                optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
                student.train()
                
                optimizer.zero_grad()
                student_logits = student.forward(x_tensor)
                # student_log_probs = torch.log_softmax(student_logits, dim=-1) # Eğer forward logit dönüyorsa
                
                # Varsayım: LoRA forward prob dönüyor (softmaxli)
                # O zaman log almalıyız
                student_log_probs = torch.log(student_logits + 1e-10)
                
                loss = torch.nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)
                
                loss.backward()
                optimizer.step()
                train_count += 1
                
            # print(f"      -> {train_count} üye eğitildi.")
            
        print("   ✅ Kabile eğitimi tamamlandı.")
