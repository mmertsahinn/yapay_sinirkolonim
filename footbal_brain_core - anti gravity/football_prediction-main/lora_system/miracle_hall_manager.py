"""
🏆 MUCİZE HALL MANAGER
=======================

Mucizeler klasöründeki tüm .pt dosyalarını okuyup
senkronize txt dosyası oluşturur.

Klasör yapısı:
mucizeler/
├── 🌱_POTANSIYEL/
│   └── *.pt
├── 🏆_MUCIZE/
│   └── *.pt
├── 👑_YUCE_MUCIZE/
│   └── *.pt
├── mucize_kayitlari.json
└── mucizeler_hall.txt  ← Bu dosyayı oluşturur!
"""

import os
import torch
from datetime import datetime
from typing import List, Dict


class MiracleHallManager:
    """
    Mucizeler klasörünü yönetir ve txt listesi oluşturur
    """
    
    def __init__(self, miracle_dir: str = "mucizeler"):
        self.miracle_dir = miracle_dir
        
        # 3 katman klasörleri
        self.potansiyel_dir = os.path.join(miracle_dir, "🌱_POTANSIYEL")
        self.mucize_dir = os.path.join(miracle_dir, "🏆_MUCIZE")
        self.yuce_dir = os.path.join(miracle_dir, "👑_YUCE_MUCIZE")
        
        # Klasörleri oluştur
        for directory in [self.potansiyel_dir, self.mucize_dir, self.yuce_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def generate_miracle_hall_txt(self, match_count: int = 0):
        """
        Tüm .pt dosyalarından senkronize txt listesi oluştur!
        """
        print("\n" + "🏆"*80)
        print("🏆 MUCİZE HALL TXT OLUŞTURULUYOR...")
        print("🏆"*80)
        
        all_miracles = []
        
        # Her katmandan .pt dosyalarını topla
        tiers = [
            ('POTANSIYEL🌱', self.potansiyel_dir),
            ('MUCIZE🏆', self.mucize_dir),
            ('YUCE_MUCIZE👑', self.yuce_dir)
        ]
        
        for tier_name, tier_dir in tiers:
            if not os.path.exists(tier_dir):
                continue
            
            pt_files = [f for f in os.listdir(tier_dir) if f.endswith('.pt')]
            
            for pt_file in pt_files:
                filepath = os.path.join(tier_dir, pt_file)
                try:
                    data = torch.load(filepath, map_location='cpu')
                    metadata = data.get('metadata', {})
                    
                    all_miracles.append({
                        'tier': tier_name,
                        'filename': pt_file,
                        'filepath': os.path.join(os.path.basename(tier_dir), pt_file),
                        'metadata': metadata
                    })
                except Exception as e:
                    print(f"   ⚠️ {pt_file} okunamadı: {e}")
        
        if len(all_miracles) == 0:
            print("   ℹ️ Henüz mucize LoRA yok!")
            return
        
        # Katmana ve fitness'a göre sırala
        tier_order = {'YUCE_MUCIZE👑': 0, 'MUCIZE🏆': 1, 'POTANSIYEL🌱': 2}
        all_miracles.sort(key=lambda x: (
            tier_order.get(x['tier'], 99),
            -x['metadata'].get('final_fitness', 0)
        ))
        
        # TXT dosyası oluştur
        txt_file = os.path.join(self.miracle_dir, "mucizeler_hall.txt")
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("🏆 MUCİZE HALL OF FAME (3 KATMANLI!)\n")
            f.write("="*80 + "\n")
            f.write(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maç: {match_count}\n")
            f.write(f"Toplam Mucize: {len(all_miracles)}\n")
            f.write("="*80 + "\n\n")
            
            # Katman istatistikleri
            tier_counts = {}
            for m in all_miracles:
                tier = m['tier']
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            f.write("📊 KATMAN İSTATİSTİKLERİ:\n")
            for tier, count in sorted(tier_counts.items(), key=lambda x: tier_order.get(x[0], 99)):
                f.write(f"   {tier}: {count} LoRA\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Her katmanı ayrı ayrı listele
            current_tier = None
            for idx, miracle_info in enumerate(all_miracles, start=1):
                tier = miracle_info['tier']
                meta = miracle_info['metadata']
                
                # Katman başlığı
                if tier != current_tier:
                    f.write("\n" + "━"*80 + "\n")
                    f.write(f"{tier} KATMANI\n")
                    f.write("━"*80 + "\n\n")
                    current_tier = tier
                
                # LoRA bilgileri
                lora_id = meta.get('id', 'Unknown')
                lora_name = meta.get('name', 'Unknown')
                fitness = meta.get('final_fitness', 0.0)
                age = meta.get('age', 0)
                miracle_score = meta.get('miracle_score', 0)
                miracle_tier = meta.get('miracle_tier', 'Unknown')
                reasons = meta.get('miracle_reasons', [])
                specialization = meta.get('specialization', 'Genel')
                
                f.write(f"{'='*80}\n")
                f.write(f"#{idx:03d} | {lora_name} | Fitness:{fitness:.3f}\n")
                f.write(f"{'='*80}\n")
                
                f.write(f"📊 TEMEL BİLGİLER:\n")
                f.write(f"   ID: {lora_id}\n")
                f.write(f"   Katman: {miracle_tier}\n")
                f.write(f"   Yaş: {age} maç\n")
                f.write(f"   Final Fitness: {fitness:.3f}\n")
                f.write(f"   Uzmanlık: {specialization}\n")
                f.write(f"   Mucize Puanı: {miracle_score}/100\n")
                f.write("\n")
                
                f.write(f"🌟 MUCİZE SEBEPLERİ:\n")
                if reasons:
                    for reason in reasons:
                        f.write(f"   • {reason}\n")
                else:
                    f.write(f"   • Belirtilmemiş\n")
                f.write("\n")
                
                f.write(f"💾 DOSYA:\n")
                f.write(f"   {miracle_info['filepath']}\n")
                f.write("\n")
        
        print(f"   ✅ Mucize Hall txt oluşturuldu: {len(all_miracles)} mucize")
        print(f"   📝 Dosya: {txt_file}")
        print("🏆"*80 + "\n")


# Global instance
miracle_hall_manager = MiracleHallManager()


