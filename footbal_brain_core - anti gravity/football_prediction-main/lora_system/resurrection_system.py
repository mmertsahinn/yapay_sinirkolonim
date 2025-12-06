"""
⚡ DİRİLTME SİSTEMİ (RESURRECTION)
==================================

Ölen LoRA'ları diriltir.

MANTIK:
- Top 50 LoRA (ölüler dahil) listesinden diriltme yapılır
- Parametreleri korunur (ağırlıklar aynı)
- Wallet sıfırlanır (yeni başlangıç gibi)
- "Dirildi" etiketi eklenir
- Dirilme sayısı tutulur

KULLANIM:
python run_evolutionary_learning.py --resurrect
"""

import os
import torch
from typing import List, Dict
from datetime import datetime


class ResurrectionSystem:
    """
    LoRA diriltme sistemi
    """
    
    def __init__(self):
        self.resurrection_count = {}  # {lora_id: dirilme_sayısı}
        print("⚡ Diriltme Sistemi başlatıldı")
    
    def resurrect_to_target(self, export_dir: str = "en_iyi_loralar",
                           miracle_dir: str = "mucizeler",
                           current_population: int = 0, 
                           target_total: int = 50,
                           device='cpu') -> List:
        """
        50'ye tamamla (3 aşamalı sistem):
        1. Top listeden dirilt
        2. Mucizelerden yükle
        3. Rastgele spawn
        
        Args:
            export_dir: Top LoRA klasörü
            miracle_dir: Mucize LoRA klasörü
            current_population: Şu anki yaşayan LoRA sayısı
            target_total: Toplam hedef LoRA sayısı (varsayılan: 50)
            device: PyTorch device
            
        Returns:
            List of resurrected/spawned LoRAAdapter instances
            
        Örnek:
            13 yaşayan + 20 dirilen + 5 mucize + 12 spawn = 50
        """
        from .lora_adapter import LoRAAdapter
        
        active_dir = os.path.join(export_dir, "⭐_AKTIF_EN_IYILER")
        
        if not os.path.exists(active_dir):
            print("   ⚠️ Export klasörü bulunamadı. Önce bir test çalıştırmalısın!")
            return []
        
        # Tüm .pt dosyalarını al
        files = [f for f in os.listdir(active_dir) if f.endswith('.pt')]
        files.sort()  # Sıralı (en iyiler önce)
        
        resurrected = []
        
        # Kaç LoRA diriltmeli?
        needed = target_total - current_population
        if needed <= 0:
            print(f"⚠️ Zaten yeterli LoRA var ({current_population}). Diriltme gerekmiyor!")
            return []
        
        print(f"\n{'⚡'*40}")
        print(f"DİRİLTME BAŞLIYOR!")
        print(f"{'⚡'*40}")
        print(f"Mevcut yaşayan: {current_population} LoRA")
        print(f"Diriltilecek: {needed} LoRA")
        print(f"Hedef toplam: {target_total} LoRA")
        print(f"Toplam export dosyası: {len(files)}")
        
        # ⚠️ KONTROL: Yeterli dosya var mı?
        if len(files) < needed:
            print(f"\n⚠️ UYARI: Export klasöründe sadece {len(files)} dosya var!")
            print(f"   {needed} LoRA gerekiyor ama yeterli değil.")
            print(f"   Mevcut {len(files)} dosyanın hepsini dirilteceğim.")
            actual_needed = len(files)
        else:
            actual_needed = needed
        
        # ÖLDÜLERI ÖNCE DİRİLT!
        dead_files = []
        alive_files = []
        
        for file in files:
            if "💀" in file:
                dead_files.append(file)
            else:
                alive_files.append(file)
        
        print(f"\n📊 Dosya analizi:")
        print(f"   💀 Ölü LoRA'lar: {len(dead_files)}")
        print(f"   ⭐ Yaşayan LoRA'lar: {len(alive_files)}")
        
        # Önce ölüleri dirilt, sonra yaşayanları ekle
        priority_files = dead_files + alive_files
        
        for i, file in enumerate(priority_files[:actual_needed], 1):  # GEREKTİĞİ KADAR!
            file_path = os.path.join(active_dir, file)
            
            try:
                checkpoint = torch.load(file_path)
                meta = checkpoint['metadata']
                
                # LoRA oluştur
                lora = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device=device).to(device)
                lora.set_all_lora_params(checkpoint['lora_params'])
                
                # Metadata'yı geri yükle
                lora.id = meta['id']
                original_name = meta['name']
                
                # Dirilme sayısını güncelle
                if lora.id not in self.resurrection_count:
                    self.resurrection_count[lora.id] = 0
                self.resurrection_count[lora.id] += 1
                
                resurrection_num = self.resurrection_count[lora.id]
                
                # Yeni isim: "Resurrected_LoRA_001_x2" (2. dirilme)
                if resurrection_num > 1:
                    lora.name = f"Resurrected_{original_name}_x{resurrection_num}"
                else:
                    lora.name = f"Resurrected_{original_name}"
                
                lora.generation = meta.get('generation', 0)
                lora.birth_match = 0  # YENİ BAŞLANGIÇ!
                lora.fitness_history = []  # SIFIR! (yeni şans)
                lora.match_history = []  # SIFIR!
                lora.specialization = meta.get('specialization')
                temp = meta.get('temperament', {})
                if not isinstance(temp, dict):
                    print(f"⚠️ UYARI: {lora.name} mizaç verisi bozuk (Tip: {type(temp)}) -> Sıfırlanıyor.")
                    temp = {}
                lora.temperament = temp
                lora.parents = meta.get('parents', [])
                
                # Dirilme metadata
                lora.resurrection_count = resurrection_num
                lora.original_fitness = meta.get('fitness', 0.5)
                lora.was_dead = not meta.get('alive', True)
                
                resurrected.append(lora)
                
                status = "💀 ÖLDÜ" if lora.was_dead else "⭐ YAŞIYORDU"
                print(f"   {i}. ⚡ {lora.name} [{status}] (Eski fitness: {lora.original_fitness:.3f})")
                
            except Exception as e:
                print(f"   ❌ {file} yüklenemedi: {e}")
        
        print(f"\n✅ {len(resurrected)} LoRA dirildi!")
        print(f"{'⚡'*40}\n")
        
        return resurrected


# Global instance
resurrection_system = ResurrectionSystem()

