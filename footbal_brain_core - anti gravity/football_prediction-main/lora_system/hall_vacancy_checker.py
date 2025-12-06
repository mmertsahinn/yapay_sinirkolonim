"""
🔍 HALL BOŞLUK KONTROLCÜSÜ
=========================

Sistem başlangıcında:
- Hangi hall'ler boş?
- Hangi roller doldurulamıyor?
- Hangi LoRA'lar kategorilendirilmemiş?

AMAÇ: Boş roller tespit et ve doldur!
"""

import os
from typing import Dict, List, Set
from collections import defaultdict


class HallVacancyChecker:
    """
    Hall boşluklarını kontrol eder
    """
    
    def __init__(self, base_dir: str = "en_iyi_loralar"):
        self.base_dir = base_dir
        
        # Beklenen hall'ler
        self.expected_halls = [
            '⭐_AKTIF_EN_IYILER',
            '💎_PERFECT_HYBRID_HALL',
            '🌟_STRONG_HYBRID_HALL',
            '🌈_HYBRID_HALL',
            '🌟_EINSTEIN_HALL',
            '🏛️_NEWTON_HALL',
            '🧬_DARWIN_HALL',
            '🌱_POTANSIYEL_HALL',
            'EINSTEIN⭐',  # 🆕 Yeni Top List
            'HYBRID🌈'     # 🆕 Yeni Top List
        ]
        
        print(f"🔍 Hall Vacancy Checker başlatıldı")
    
    def check_all_halls(self, population: List, match_num: int = 0) -> Dict:
        """
        TÜM HALL'LERİ KONTROL ET!
        
        Returns:
            {
                'empty_halls': List[str],
                'hall_counts': Dict[str, int],
                'uncategorized_loras': List,
                'total_loras': int
            }
        """
        
        print(f"\n🔍 HALL BOŞLUK KONTROLÜ BAŞLIYOR...")
        print(f"{'═'*80}")
        
        try:
            print(f"   🔍 DEBUG: {len(self.expected_halls)} hall kontrol edilecek...")
            print(f"   🔍 DEBUG: Base dir: {self.base_dir}")
        except:
            pass
        
        # Hall'lerdeki dosya sayıları
        hall_counts = {}
        empty_halls = []
        
        for hall_name in self.expected_halls:
            hall_path = os.path.join(self.base_dir, hall_name)
            
            if not os.path.exists(hall_path):
                # Hall klasörü yok!
                print(f"   ⚠️  {hall_name}: KLASÖR YOK!")
                empty_halls.append(hall_name)
                hall_counts[hall_name] = 0
                continue
            
            # PT dosyalarını say
            pt_files = [f for f in os.listdir(hall_path) if f.endswith('.pt')]
            count = len(pt_files)
            hall_counts[hall_name] = count
            
            if count == 0:
                empty_halls.append(hall_name)
                print(f"   ❌ {hall_name}: BOŞ! (0 LoRA)")
            elif count < 5:
                print(f"   ⚠️  {hall_name}: {count} LoRA (az!)")
            else:
                print(f"   ✅ {hall_name}: {count} LoRA")
        
        # Kategorilendirilmemiş LoRA'ları bul
        uncategorized = self._find_uncategorized_loras(population, hall_counts, match_num)
        
        print(f"{'═'*80}")
        print(f"\n📊 ÖZET:")
        print(f"   • Toplam Hall: {len(self.expected_halls)}")
        print(f"   • Boş Hall: {len(empty_halls)}")
        print(f"   • Kategorilendirilmemiş: {len(uncategorized)} LoRA")
        print(f"   • Toplam Yaşayan: {len(population)} LoRA")
        
        if empty_halls:
            print(f"\n   🚨 BOŞ HALL'LER:")
            for hall in empty_halls:
                print(f"      • {hall}")
        
        if uncategorized:
            print(f"\n   ⚠️  KATEGORİLENDİRİLMEMİŞ LoRA'LAR:")
            
            # Sebeplere göre grupla
            by_reason = {}
            for lora in uncategorized:
                reason = getattr(lora, '_uncategorized_reason', 'Bilinmeyen')
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(lora)
            
            # Her sebep için göster
            for reason, loras in by_reason.items():
                print(f"\n      📌 SEBEP: {reason} ({len(loras)} LoRA)")
                for lora in loras[:3]:  # İlk 3'ü
                    tes_type = getattr(lora, '_tes_scores', {}).get('lora_type', 'Unknown')
                    age = match_num - getattr(lora, 'birth_match', match_num) if hasattr(lora, 'birth_match') else 0
                    print(f"         • {lora.name[:25]:25s} | Yaş:{age:2d} | {tes_type[:20]}")
                if len(loras) > 3:
                    print(f"         ... ve {len(loras) - 3} tane daha")
        
        print(f"{'═'*80}\n")
        
        return {
            'empty_halls': empty_halls,
            'hall_counts': hall_counts,
            'uncategorized_loras': uncategorized,
            'total_loras': len(population)
        }
    
    def _find_uncategorized_loras(self, population: List, hall_counts: Dict, match_num: int = 0) -> List:
        """
        Hangi LoRA'lar hiçbir hall'de yok?
        """
        
        # Tüm hall'lerdeki LoRA ID'lerini topla
        categorized_ids = set()
        
        for hall_name in self.expected_halls:
            hall_path = os.path.join(self.base_dir, hall_name)
            
            if not os.path.exists(hall_path):
                continue
            
            pt_files = [f for f in os.listdir(hall_path) if f.endswith('.pt')]
            
            for pt_file in pt_files:
                # Dosya adından ID'yi çıkar (NAME_ID.pt formatı)
                try:
                    lora_id = pt_file.split('_')[-1].replace('.pt', '')
                    categorized_ids.add(lora_id)
                except:
                    pass
        
        # Kategorilendirilmemiş olanları bul
        uncategorized = []
        
        for lora in population:
            # LoRA'nın ID'sinin ilk 8 karakterini kontrol et
            lora_id_short = lora.id[:8]
            
            # Tam ID veya kısa ID kategorilendirildiyse pas geç
            if lora.id in categorized_ids or lora_id_short in categorized_ids:
                continue
            
            # Daha esnek kontrol: ID parçası herhangi bir dosyada geçiyor mu?
            is_categorized = False
            for hall_name in self.expected_halls:
                hall_path = os.path.join(self.base_dir, hall_name)
                if not os.path.exists(hall_path):
                    continue
                
                pt_files = [f for f in os.listdir(hall_path) if f.endswith('.pt')]
                for pt_file in pt_files:
                    if lora.id[:8] in pt_file or lora.id[:16] in pt_file:
                        is_categorized = True
                        break
                
                if is_categorized:
                    break
            
            if not is_categorized:
                # 🔍 SEBEP ANALİZİ: Neden kategorilendirilmemiş?
                age = match_num - getattr(lora, 'birth_match', match_num) if hasattr(lora, 'birth_match') else 0
                fitness = lora.get_recent_fitness() if hasattr(lora, 'get_recent_fitness') else 0.5
                
                # Sebep belirleme
                if age == 0:
                    reason = "YENİ DOĞMUŞ (0 maç)"
                elif age < 10:
                    reason = f"ÇÖMEZ ({age} maç - deneyimsiz)"
                elif fitness < 0.30:
                    reason = f"DÜŞÜK FİTNESS ({fitness:.2f} - zayıf)"
                elif not hasattr(lora, '_tes_scores'):
                    reason = "TES HESAPLANMAMIŞ (sistem hatası?)"
                else:
                    reason = "SİSTEM HATASI (sebep belirsiz!)"
                
                lora._uncategorized_reason = reason
                uncategorized.append(lora)
        
        return uncategorized


# Global instance
hall_vacancy_checker = HallVacancyChecker()

