"""
🔄 DİNAMİK YER DEĞİŞTİRME MOTORU
=================================

CANLI SİSTEM! LoRA'lar performansa göre klasör değiştirir!

ÖZELLİKLER:
- Gerçek zamanlı kümelenme
- Otomatik yer değiştirme (terfi/düşme)
- Her hareket loglanır
- Debug mode (her şey görünür!)

ÖRNEK:
  LoRA_X: Darwin Hall → (Performans arttı) → Perfect Hybrid Hall
  LoRA_Y: Einstein Hall → (Performans düştü) → Potansiyel Hall
  LoRA_Z: Genel Uzman → (Manchester'da iyi) → Manchester Win Expert

HER MAÇ SONRASI KONTROL!
"""

import os
import torch
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class DynamicRelocationEngine:
    """
    LoRA'ları dinamik olarak doğru klasörlere yerleştirir ve taşır
    """
    
    def __init__(self, base_dir: str = "en_iyi_loralar", log_dir: str = "evolution_logs"):
        self.base_dir = base_dir
        self.log_dir = log_dir
        
        # Log dosyası
        self.relocation_log = os.path.join(log_dir, "🔄_DYNAMIC_RELOCATION.log")
        
        # Her LoRA'nın güncel konumları {lora_id: [klasör1, klasör2, ...]}
        self.lora_locations = defaultdict(set)
        
        # Yerleşme geçmişi {lora_id: [(maç, from, to, reason), ...]}
        self.relocation_history = defaultdict(list)
        
        # İstatistikler
        self.stats = {
            'total_relocations': 0,
            'promotions': 0,  # Yükselme (ör: Hybrid → Perfect Hybrid)
            'demotions': 0,   # Düşme (ör: Einstein → Normal)
            'new_placements': 0,  # İlk yerleşme
            'removals': 0  # Klasörden çıkarma
        }
        
        self._write_header()
        
        print(f"🔄 Dynamic Relocation Engine başlatıldı")
    
    def _write_header(self):
        """Log başlığı"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(self.relocation_log, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("🔄 DİNAMİK YER DEĞİŞTİRME MOTORU - HAREKETnLER LOG\n")
            f.write("=" * 120 + "\n")
            f.write(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 120 + "\n\n")
            f.write("AMAÇ: LoRA'ları performansa göre en uygun klasörlere yerleştir!\n\n")
            f.write("HAREKET TİPLERİ:\n")
            f.write("  ⬆️ TERFİ (Promotion): Daha prestijli klasöre taşınma\n")
            f.write("  ⬇️ DÜŞME (Demotion): Daha düşük klasöre taşınma\n")
            f.write("  🆕 YENİ (New): İlk yerleşme\n")
            f.write("  ➡️ TRANSFER (Transfer): Farklı kategoriye geçiş\n")
            f.write("  ❌ ÇIKARMA (Removal): Klasörden çıkarılma\n")
            f.write("=" * 120 + "\n\n")
    
    def evaluate_and_relocate_all(self,
                                  population: List,
                                  match_idx: int,
                                  tes_triple_scoreboard,
                                  team_spec_manager,
                                  global_spec_manager) -> Dict:
        """
        TÜM LoRA'LARI DEĞERLENDIR VE YERLEŞTİR!
        
        🔥 CANLI SİSTEM! Her 10 maçta dosya işlemleri!
        """
        
        relocations = []
        
        # Her 10 maçta detaylı log
        if match_idx % 10 == 0:
            print(f"\n🔄 CANLI DİNAMİK YER DEĞİŞTİRME (Maç #{match_idx})...")
            print(f"   🔍 DEBUG [Dynamic Relocation]: BAŞLADI!")
            print(f"   🔍 DEBUG: {len(population)} LoRA kontrol edilecek...")
        
        for lora in population:
            # Mevcut konumları
            current_locations = self.lora_locations.get(lora.id, set())
            
            # Yeni konumları hesapla
            new_locations = self._calculate_ideal_locations(
                lora, match_idx, population,
                team_spec_manager, global_spec_manager
            )
            
            # Değişiklik var mı?
            added = new_locations - current_locations
            removed = current_locations - new_locations
            
            if added or removed:
                # 🔥 ROL DEĞİŞİKLİĞİ! CANLI GÖSTER!
                relocation = {
                    'lora_id': lora.id,
                    'lora_name': lora.name,
                    'match': match_idx,
                    'added': list(added),
                    'removed': list(removed),
                    'current': list(current_locations),
                    'new': list(new_locations),
                    'tes_type': getattr(lora, '_tes_scores', {}).get('lora_type', 'Unknown')
                }
                relocations.append(relocation)
                
                # 👁️ GÖRÜNÜR OLSUN! (Her 10 maçta)
                if match_idx % 10 == 0:
                    self._print_role_change(lora, removed, added, match_idx)
                
                # History'e ekle
                for loc in added:
                    self.relocation_history[lora.id].append((match_idx, None, loc, 'ADDED'))
                    self.stats['new_placements'] += 1
                    
                    # Terfi mi?
                    if '💎_PERFECT' in loc:
                        self.stats['promotions'] += 1
                    elif '🌟_STRONG' in loc:
                        self.stats['promotions'] += 1
                
                for loc in removed:
                    self.relocation_history[lora.id].append((match_idx, loc, None, 'REMOVED'))
                    self.stats['removals'] += 1
                
                # Konumları güncelle
                self.lora_locations[lora.id] = new_locations
        
        # 🔥 HER 10 MAÇTA DOSYA İŞLEMLERİ! (50 değil!)
        if match_idx % 10 == 0 and match_idx > 0:
            if relocations:
                try:
                    print(f"\n   📁 DOSYA İŞLEMLERİ YAPILIYOR...")
                    print(f"   🔍 DEBUG: {len(relocations)} LoRA'nın dosyaları taşınacak...")
                    self._execute_file_operations(relocations, match_idx)
                    print(f"   ✅ {len(relocations)} LoRA'nın dosyaları güncellendi!")
                except Exception as e:
                    print(f"   ❌ HATA: Dosya işlemleri başarısız!")
                    print(f"   ❌ Hata: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Log yaz
        if match_idx % 10 == 0:
            if relocations:
                print(f"   🔍 DEBUG [Dynamic Relocation]: {len(relocations)} rol değişikliği tespit edildi!")
                self._log_relocations(relocations, match_idx)
            else:
                print(f"   🔍 DEBUG [Dynamic Relocation]: Rol değişikliği YOK (herkes yerinde)")
        
        return {
            'relocations': relocations,
            'stats': self.stats
        }
    
    def _print_role_change(self, lora, removed: set, added: set, match_idx: int):
        """
        Rol değişikliğini GÖZLE GÖRÜNÜR şekilde print et!
        """
        if not removed and not added:
            return
        
        print(f"\n   🎭 ROL DEĞİŞİKLİĞİ: {lora.name[:25]}")
        
        # Kaldırılan roller
        if removed:
            for loc in removed:
                emoji = self._get_hall_emoji(loc)
                print(f"      ⬅️  {emoji} {loc}")
        
        # Eklenen roller
        if added:
            for loc in added:
                emoji = self._get_hall_emoji(loc)
                is_promotion = any(x in loc for x in ['💎_PERFECT', '🌟_STRONG', '🌟_EINSTEIN', '🏛️_NEWTON'])
                arrow = "⬆️" if is_promotion else "➡️"
                print(f"      {arrow}  {emoji} {loc}")
    
    def _get_hall_emoji(self, hall_name: str) -> str:
        """Hall adından emoji çıkar"""
        if '💎' in hall_name:
            return '💎'
        elif '🌟' in hall_name:
            return '🌟'
        elif '🌈' in hall_name:
            return '🌈'
        elif '🏛️' in hall_name:
            return '🏛️'
        elif '🧬' in hall_name:
            return '🧬'
        elif '🌱' in hall_name:
            return '🌱'
        else:
            return '📁'
    
    def _calculate_ideal_locations(self,
                                   lora,
                                   match_idx: int,
                                   population: List,
                                   team_spec_manager,
                                   global_spec_manager) -> Set[str]:
        """
        LoRA için ideal konumları hesapla (DEBUG MODE!)
        """
        locations = set()
        
        # TES TİPİ
        if hasattr(lora, '_tes_scores'):
            tes_type = lora._tes_scores.get('lora_type', '')
            darwin = lora._tes_scores.get('darwin', 0)
            einstein = lora._tes_scores.get('einstein', 0)
            newton = lora._tes_scores.get('newton', 0)
            
            # 🌟 DEBUG: TES skorlarını print et
            if match_idx % 10 == 0:
                print(f"      🔍 {lora.name}: TES={lora._tes_scores.get('total_tes', 0):.3f} "
                      f"(D:{darwin:.2f}, E:{einstein:.2f}, N:{newton:.2f}) → {tes_type}")
            
            # PERFECT HYBRID (0.75+)
            if 'PERFECT HYBRID💎💎💎' in tes_type:
                locations.add('💎_PERFECT_HYBRID_HALL')
                if match_idx % 10 == 0:
                    print(f"         → 💎 PERFECT HYBRID HALL!")
            
            # STRONG HYBRID (0.50+)
            elif 'STRONG HYBRID🌟🌟' in tes_type:
                locations.add('🌟_STRONG_HYBRID_HALL')
                if match_idx % 10 == 0:
                    print(f"         → 🌟 STRONG HYBRID HALL!")
            
            # HYBRID (0.30+)
            elif 'HYBRID' in tes_type:
                locations.add('🌈_HYBRID_HALL')
            
            # EINSTEIN
            if 'EINSTEIN' in tes_type or einstein >= 0.30:
                locations.add('🌟_EINSTEIN_HALL')
            
            # NEWTON
            if 'NEWTON' in tes_type or newton >= 0.30:
                locations.add('🏛️_NEWTON_HALL')
            
            # DARWIN
            if 'DARWIN' in tes_type or darwin >= 0.30:
                locations.add('🧬_DARWIN_HALL')
        
        # TOP 50
        locations.add('⭐_AKTIF_EN_IYILER')  # Her yaşayan burada olmalı
        
        return locations
    
    def _execute_file_operations(self, relocations: List[Dict], match_idx: int):
        """
        Gerçek dosya taşıma işlemlerini yap (Her 50 maçta!)
        """
        
        for relocation in relocations:
            lora_id = relocation['lora_id']
            lora_name = relocation['lora_name']
            
            # PT dosya adı
            pt_file = f"{lora_name}_{lora_id}.pt"
            
            # EKLEME
            for location in relocation['added']:
                target_dir = os.path.join(self.base_dir, location)
                os.makedirs(target_dir, exist_ok=True)
                
                # Kaynaktan kopyala (AKTIF_EN_IYILER'den)
                source_file = os.path.join(self.base_dir, '⭐_AKTIF_EN_IYILER', f"{lora_id}.pt")
                target_file = os.path.join(target_dir, pt_file)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                    print(f"      ➕ {pt_file} → {location}")
            
            # ÇIKARMA
            for location in relocation['removed']:
                target_dir = os.path.join(self.base_dir, location)
                target_file = os.path.join(target_dir, pt_file)
                
                if os.path.exists(target_file):
                    os.remove(target_file)
                    print(f"      ➖ {pt_file} ← {location}")
    
    def _log_relocations(self, relocations: List[Dict], match_idx: int):
        """
        Yer değiştirmeleri logla
        """
        
        with open(self.relocation_log, 'a', encoding='utf-8') as f:
            f.write("\n" + "━" * 120 + "\n")
            f.write(f"🔄 MAÇ #{match_idx} - YER DEĞİŞTİRMELER\n")
            f.write("━" * 120 + "\n")
            f.write(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📊 Toplam Hareket: {len(relocations)}\n\n")
            
            for i, rel in enumerate(relocations, 1):
                f.write(f"#{i}. {rel['lora_name']} (ID: {rel['lora_id'][:8]}...)\n")
                
                # Eklenenler
                if rel['added']:
                    f.write(f"   ➕ EKLENDİ:\n")
                    for loc in rel['added']:
                        f.write(f"      → {loc}\n")
                
                # Çıkarılanlar
                if rel['removed']:
                    f.write(f"   ➖ ÇIKARILDI:\n")
                    for loc in rel['removed']:
                        f.write(f"      ← {loc}\n")
                
                f.write(f"   📍 ŞU AN: {', '.join(rel['new']) if rel['new'] else 'YOK'}\n")
                f.write("   " + "─" * 80 + "\n")
            
            f.write("\n" + "━" * 120 + "\n")
    
    def print_current_distribution(self, match_idx: int):
        """
        Mevcut dağılımı print et (DEBUG!)
        """
        
        distribution = defaultdict(set)
        
        for lora_id, locations in self.lora_locations.items():
            for loc in locations:
                distribution[loc].add(lora_id)
        
        print(f"\n📊 MEVCUT DAĞILIM (Maç #{match_idx}):")
        print("─" * 100)
        
        for loc in sorted(distribution.keys()):
            count = len(distribution[loc])
            print(f"   {loc:40s}: {count:3d} LoRA")
        
        print("─" * 100)


# Global instance
dynamic_relocation_engine = DynamicRelocationEngine()

