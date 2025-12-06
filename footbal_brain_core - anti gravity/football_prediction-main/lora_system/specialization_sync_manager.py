"""
🔄 SPECIALIZATION SYNC MANAGER
===============================

PT dosyalarını çoklu uzmanlık klasörlerine kopyalar ve sync eder!

ÖZELLİKLER:
- Bir LoRA birden fazla uzmanlığa sahipse → Her klasöre kopyala
- LoRA güncellendiğinde → Tüm kopyalar sync olur
- Uzmanlık kaybedilirse → O klasörden PT silinir
- Dosya adı: İsim_ID.pt (wallet ile tutarlı)

KULLANIM:
    sync_manager.register_lora_specializations(lora, specializations)
    sync_manager.sync_all_lora_copies(lora)  # Her güncellemede!
"""

import os
import torch
import shutil
from typing import Dict, List, Set
from datetime import datetime
from collections import defaultdict


class SpecializationSyncManager:
    """
    LoRA'ların PT dosyalarını çoklu lokasyonlara kopyalar ve sync eder
    """
    
    def __init__(self):
        # Her LoRA'nın PT dosyasının bulunduğu tüm lokasyonlar
        # {lora_id: [path1, path2, path3, ...]}
        self.lora_locations = defaultdict(set)
        
        # Her LoRA'nın aktif uzmanlıkları
        # {lora_id: {'Manchester_Win', 'Global_Win', ...}}
        self.lora_specializations = defaultdict(set)
        
        print("🔄 Specialization Sync Manager başlatıldı!")
    
    def register_lora_specializations(self, 
                                     lora, 
                                     team_specializations: Dict[str, List[str]],
                                     global_specializations: List[str],
                                     base_dirs: Dict[str, str]):
        """
        Bir LoRA'nın tüm uzmanlıklarını kaydet ve PT dosyalarını kopyala!
        
        Args:
            lora: LoRA instance
            team_specializations: {'Manchester_United': ['Win', 'Goal'], ...}
            global_specializations: ['Win', 'Goal', ...]
            base_dirs: {
                'team': 'takım_uzmanlıkları',
                'global': 'en_iyi_loralar/🌍_GENEL_UZMANLAR'
            }
        """
        lora_id = lora.id
        pt_filename = f"{lora.name}_{lora.id}.pt"
        
        # Yeni lokasyonlar
        new_locations = set()
        new_specs = set()
        
        # 1) TAKIM UZMANLIKLARI
        for team_name, spec_types in team_specializations.items():
            for spec_type in spec_types:
                # Klasör yolu
                safe_team = self._safe_team_name(team_name)
                team_dir = os.path.join(base_dirs['team'], safe_team)
                
                if spec_type == 'Win':
                    subdir = '🎯_WIN_EXPERTS'
                elif spec_type == 'Goal':
                    subdir = '⚽_GOAL_EXPERTS'
                elif spec_type == 'Hype':
                    subdir = '🔥_HYPE_EXPERTS'
                else:
                    continue  # VS için ayrı mantık gerekebilir
                
                expert_dir = os.path.join(team_dir, subdir)
                os.makedirs(expert_dir, exist_ok=True)
                
                pt_path = os.path.join(expert_dir, pt_filename)
                new_locations.add(pt_path)
                new_specs.add(f"{team_name}_{spec_type}")
        
        # 2) GENEL UZMANLIKLAR
        for spec_type in global_specializations:
            if spec_type == 'Win':
                subdir = '🎯_WIN_EXPERTS'
            elif spec_type == 'Goal':
                subdir = '⚽_GOAL_EXPERTS'
            elif spec_type == 'Hype':
                subdir = '🔥_HYPE_EXPERTS'
            else:
                continue
            
            expert_dir = os.path.join(base_dirs['global'], subdir)
            os.makedirs(expert_dir, exist_ok=True)
            
            pt_path = os.path.join(expert_dir, pt_filename)
            new_locations.add(pt_path)
            new_specs.add(f"Global_{spec_type}")
        
        # 3) KAYDET
        self.lora_locations[lora_id] = new_locations
        self.lora_specializations[lora_id] = new_specs
        
        # 4) PT DOSYALARINI KOPYALA (ilk kez)
        if len(new_locations) > 0:
            self._save_lora_to_locations(lora, new_locations)
            print(f"   🔄 {lora.name}: {len(new_locations)} lokasyona kopyalandı")
    
    def sync_all_lora_copies(self, lora):
        """
        Bir LoRA'nın TÜM kopyalarını güncelle!
        
        Her maç sonrası çağrılmalı (parametreler, fizik değişti!)
        """
        lora_id = lora.id
        
        if lora_id not in self.lora_locations:
            return  # Bu LoRA henüz kayıtlı değil
        
        locations = self.lora_locations[lora_id]
        
        if len(locations) == 0:
            return
        
        # Tüm lokasyonlara güncelle
        self._save_lora_to_locations(lora, locations)
    
    def remove_specialization(self, lora, specialization_name: str):
        """
        Bir LoRA uzmanlığını kaybetti → O klasörden PT'yi sil!
        
        Args:
            lora: LoRA instance
            specialization_name: 'Manchester_Win', 'Global_Goal', etc.
        """
        lora_id = lora.id
        
        if lora_id not in self.lora_specializations:
            return
        
        # Uzmanlığı kaldır
        if specialization_name in self.lora_specializations[lora_id]:
            self.lora_specializations[lora_id].remove(specialization_name)
            
            # İlgili PT dosyasını bul ve sil
            pt_filename = f"{lora.name}_{lora.id}.pt"
            
            # O uzmanlığa ait lokasyonu bul ve sil
            locations_to_remove = set()
            for loc in self.lora_locations[lora_id]:
                if specialization_name in loc:
                    if os.path.exists(loc):
                        os.remove(loc)
                        print(f"   🗑️ {specialization_name} uzmanlığı kaybedildi → {loc} silindi")
                    locations_to_remove.add(loc)
            
            # Lokasyonları güncelle
            self.lora_locations[lora_id] -= locations_to_remove
    
    def get_lora_specialization_count(self, lora_id: str) -> int:
        """
        Bir LoRA'nın kaç uzmanlığı var?
        """
        return len(self.lora_specializations.get(lora_id, set()))
    
    def _save_lora_to_locations(self, lora, locations: Set[str]):
        """
        Bir LoRA'yı tüm lokasyonlara kaydet!
        """
        # PT data oluştur
        pt_data = {
            'lora_params': lora.get_all_lora_params(),
            'metadata': {
                'id': lora.id,
                'name': lora.name,
                'pt_filename': f"{lora.name}_{lora.id}.pt",
                'sync_timestamp': datetime.now().isoformat(),
                
                # ✅ TÜM FİZİK PARAMETRELERİ!
                'life_energy': getattr(lora, 'life_energy', 1.0),
                'lazarus_lambda': getattr(lora, '_lazarus_lambda', 0.5),
                'tes_scores': getattr(lora, '_tes_scores', {}),
                'lora_type': getattr(lora, 'lora_type', 'HYBRID'),
                
                # Fizik
                'langevin_temp': getattr(lora, '_langevin_temp', 0.01),
                'nose_hoover_xi': getattr(lora, '_nose_hoover_xi', 0.0),
                'kinetic_energy': getattr(lora, '_kinetic_energy', 0.0),
                'om_action': getattr(lora, '_om_action', 0.0),
                'ghost_potential': getattr(lora, '_ghost_potential', 0.0),
                
                # Arketipler
                'particle_archetype': getattr(lora, '_particle_archetype', 'Unknown'),
                'emotional_archetype': getattr(lora, 'emotional_archetype', 'Dengeli'),
                'physics_archetype': getattr(lora, 'physics_archetype', 'Standart'),
                
                # Mizaç
                'temperament': getattr(lora, 'temperament', {}),
                
                # Diğer
                'generation': lora.generation,
                'birth_match': lora.birth_match,
                'fitness': lora.get_recent_fitness(),
                'specializations': list(self.lora_specializations.get(lora.id, set()))
            }
        }
        
        # Tüm lokasyonlara kaydet
        for location in locations:
            try:
                torch.save(pt_data, location)
            except Exception as e:
                print(f"⚠️ {location} kaydedilemedi: {e}")
    
    def _safe_team_name(self, team_name: str) -> str:
        """
        Takım adını dosya sistemi için güvenli hale getir
        """
        return team_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    
    def cleanup_orphaned_files(self, base_dirs: Dict[str, str]):
        """
        Artık uzmanlığı olmayan PT dosyalarını temizle
        
        (Opsiyonel - her 100 maçta bir çağrılabilir)
        """
        # TODO: Implement if needed
        pass


# Global instance
specialization_sync_manager = SpecializationSyncManager()


