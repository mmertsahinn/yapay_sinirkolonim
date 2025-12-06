"""
🔄 LoRA SENKRONIZASYON KOORDİNATÖRÜ
====================================

TÜM KOPYALARI SENKRON VE GÜNCEL TUTAR!

BİR LoRA GÜNCELLENSE:
1. Ana dosya güncellenir (⭐_AKTIF_EN_IYILER)
2. TÜM kopyalar senkronize edilir
   - Einstein Hall
   - Hybrid Hall
   - Darwin Hall
   - Newton Hall
   - Takım uzmanlıkları
   - VS klasörleri
3. Population history kaydeder
4. Auditor kontrol eder

AMAÇ: Hiçbir kopya eski kalmasın!
"""

import os
import torch
from datetime import datetime
from typing import Dict, List, Set, Tuple
from pathlib import Path


class LoRASyncCoordinator:
    """
    LoRA kopyalarını senkronize eder ve denetler
    """
    
    def __init__(self, base_dir: str = "en_iyi_loralar"):
        self.base_dir = base_dir
        
        # Her LoRA'nın tüm kopyalarının konumları {lora_id: [paths]}
        self.lora_copy_map = {}
        
        # Son senkronizasyon zamanları {lora_id: timestamp}
        self.last_sync = {}
        
        # Senkronizasyon sayacı
        self.sync_count = 0
        
        print(f"🔄 LoRA Sync Coordinator başlatıldı")
    
    def register_lora_copy(self, lora_id: str, lora_name: str, file_path: str):
        """
        Bir LoRA kopyasını kaydet
        """
        
        if lora_id not in self.lora_copy_map:
            self.lora_copy_map[lora_id] = {
                'name': lora_name,
                'copies': set()
            }
        
        self.lora_copy_map[lora_id]['copies'].add(file_path)
    
    def sync_all_copies(self, lora, match_idx: int, population_history=None, reason: str = "UPDATE"):
        """
        Bir LoRA'nın TÜM kopyalarını senkronize et!
        
        Args:
            lora: LoRA instance (ana kaynak)
            match_idx: Maç numarası
            population_history: History kaydedici (opsiyonel)
            reason: Senkronizasyon sebebi
        
        Returns:
            {
                'synced_count': int,
                'failed_count': int,
                'locations': List[str]
            }
        """
        
        try:
            lora_id = lora.id
            lora_name = lora.name
            
            # Debug: Başlangıç
            if match_idx % 10 == 0:  # Sadece her 10 maçta print
                print(f"      🔍 DEBUG: Sync başlatılıyor → {lora_name[:25]}")
            
            # Bu LoRA'nın tüm kopyalarını bul
            all_copies = self._find_all_copies(lora_id, lora_name)
            
            if match_idx % 10 == 0:
                print(f"         • {len(all_copies)} kopya bulundu")
        
        except Exception as e:
            print(f"      ❌ HATA: Sync başlatılamadı!")
            print(f"      ❌ LoRA: {lora.name if hasattr(lora, 'name') else 'Unknown'}")
            print(f"      ❌ Hata: {str(e)}")
            return {'synced_count': 0, 'failed_count': 0, 'locations': []}
        
        if len(all_copies) == 0:
            return {
                'synced_count': 0,
                'failed_count': 0,
                'locations': []
            }
        
        # Ana veriyi hazırla
        main_data = {
            'lora_params': lora.get_all_lora_params(),
            'metadata': {
                'id': lora.id,
                'name': lora.name,
                'generation': lora.generation,
                'birth_match': lora.birth_match,
                'fitness_history': lora.fitness_history,
                'life_energy': getattr(lora, 'life_energy', 1.0),
                'temperament': getattr(lora, 'temperament', {}),
                '_tes_scores': getattr(lora, '_tes_scores', {}),
                '_lazarus_lambda': getattr(lora, '_lazarus_lambda', 0.5),
                '_langevin_temp': getattr(lora, '_langevin_temp', 0.01),
                '_nose_hoover_xi': getattr(lora, '_nose_hoover_xi', 0.0),
                '_kinetic_energy': getattr(lora, '_kinetic_energy', 0.0),
                '_om_action': getattr(lora, '_om_action', 0.0),
                '_ghost_potential': getattr(lora, '_ghost_potential', 0.0),
                'sync_info': {
                    'last_sync_match': match_idx,
                    'last_sync_time': datetime.now().isoformat(),
                    'sync_reason': reason
                }
            }
        }
        
        # Tüm kopyaları güncelle
        synced_count = 0
        failed_count = 0
        synced_locations = []
        
        try:
            for copy_path in all_copies:
                try:
                    # Mevcut dosyayı yükle (metadata'yı korumak için)
                    if os.path.exists(copy_path):
                        existing_data = torch.load(copy_path, map_location='cpu')
                        existing_metadata = existing_data.get('metadata', {})
                        
                        # Özel metadata'yı koru (team, specialization_key, score, vs.)
                        preserved_keys = ['team', 'specialization_key', 'score', 'match_count', 'exported_at']
                        for key in preserved_keys:
                            if key in existing_metadata:
                                main_data['metadata'][key] = existing_metadata[key]
                    
                    # Güncel veriyi kaydet
                    torch.save(main_data, copy_path)
                    synced_count += 1
                    synced_locations.append(copy_path)
                    
                except Exception as e:
                    failed_count += 1
                    print(f"      ⚠️  Senkronizasyon hatası: {copy_path}")
                    print(f"         Hata: {str(e)}")
        
        except Exception as e:
            print(f"      ❌ HATA: Kopyalar güncellenirken hata!")
            print(f"      ❌ Hata: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Son senkronizasyon zamanını güncelle
        self.last_sync[lora_id] = datetime.now()
        self.sync_count += 1
        
        # Population history'ye kaydet (eğer verilmişse)
        if population_history:
            try:
                population_history.record_lora_event(
                    lora.id,
                    lora.name,
                    match_idx,
                    'SYNC',
                    {
                        'synced_copies': synced_count,
                        'failed_copies': failed_count,
                        'total_copies': len(all_copies),
                        'reason': reason
                    }
                )
            except:
                pass
        
        return {
            'synced_count': synced_count,
            'failed_count': failed_count,
            'locations': synced_locations
        }
    
    def _find_all_copies(self, lora_id: str, lora_name: str) -> Set[str]:
        """
        Bir LoRA'nın tüm kopyalarını bul
        """
        
        all_copies = set()
        
        # Base directory'de ara
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.pt'):
                    # Dosya adında LoRA ID veya adı geçiyor mu?
                    if lora_id in file or lora_id[:16] in file or lora_id[:8] in file:
                        file_path = os.path.join(root, file)
                        all_copies.add(file_path)
                    elif lora_name in file:
                        file_path = os.path.join(root, file)
                        all_copies.add(file_path)
        
        # Kendi map'imizde varsa ekle
        if lora_id in self.lora_copy_map:
            all_copies.update(self.lora_copy_map[lora_id]['copies'])
        
        return all_copies
    
    def sync_entire_population(self, population: List, match_idx: int, population_history=None) -> Dict:
        """
        TÜM popülasyonu senkronize et!
        
        Her 10 maçta çağrılmalı
        """
        
        print(f"\n🔄 TOPLU SENKRONIZASYON BAŞLIYOR (Maç #{match_idx})...")
        
        try:
            print(f"   🔍 DEBUG: {len(population)} LoRA senkronize edilecek...")
            
            total_synced = 0
            total_failed = 0
            loras_with_copies = 0
            
            for lora in population:
                result = self.sync_all_copies(lora, match_idx, population_history, reason="PERIODIC_SYNC")
                
                if result['synced_count'] > 0:
                    loras_with_copies += 1
                    total_synced += result['synced_count']
                    total_failed += result['failed_count']
            
            print(f"   ✅ {loras_with_copies} LoRA senkronize edildi")
            print(f"   📁 Toplam {total_synced} dosya güncellendi")
            
            if total_failed > 0:
                print(f"   ⚠️  {total_failed} dosya başarısız")
            
            print(f"   🔍 DEBUG: Sync tamamlandı başarıyla!")
            
            return {
                'loras_synced': loras_with_copies,
                'files_synced': total_synced,
                'files_failed': total_failed
            }
            
        except Exception as e:
            print(f"   ❌ HATA: Toplu senkronizasyon başarısız!")
            print(f"   ❌ Hata: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'loras_synced': 0,
                'files_synced': 0,
                'files_failed': 999
            }
    
    def verify_sync_integrity(self, lora_id: str, lora_name: str) -> Dict:
        """
        Bir LoRA'nın tüm kopyalarının tutarlı olduğunu doğrula
        """
        
        all_copies = self._find_all_copies(lora_id, lora_name)
        
        if len(all_copies) == 0:
            return {
                'is_consistent': True,
                'total_copies': 0,
                'issues': []
            }
        
        # İlk dosyayı referans al
        reference_data = None
        reference_path = None
        
        for copy_path in all_copies:
            try:
                data = torch.load(copy_path, map_location='cpu')
                if reference_data is None:
                    reference_data = data
                    reference_path = copy_path
                    break
            except:
                continue
        
        if reference_data is None:
            return {
                'is_consistent': False,
                'total_copies': len(all_copies),
                'issues': ['NO_VALID_REFERENCE']
            }
        
        # Diğer kopyaları referansla karşılaştır
        issues = []
        
        for copy_path in all_copies:
            if copy_path == reference_path:
                continue
            
            try:
                data = torch.load(copy_path, map_location='cpu')
                
                # Parametreleri karşılaştır
                ref_params = reference_data.get('lora_params', {})
                copy_params = data.get('lora_params', {})
                
                # Parametre sayısı aynı mı?
                if len(ref_params) != len(copy_params):
                    issues.append({
                        'type': 'PARAM_COUNT_MISMATCH',
                        'file': copy_path,
                        'expected': len(ref_params),
                        'actual': len(copy_params)
                    })
                    continue
                
                # Her parametre aynı mı?
                for key in ref_params:
                    if key not in copy_params:
                        issues.append({
                            'type': 'MISSING_PARAM',
                            'file': copy_path,
                            'param': key
                        })
                    elif not torch.equal(ref_params[key], copy_params[key]):
                        issues.append({
                            'type': 'PARAM_MISMATCH',
                            'file': copy_path,
                            'param': key
                        })
                
            except Exception as e:
                issues.append({
                    'type': 'LOAD_ERROR',
                    'file': copy_path,
                    'error': str(e)
                })
        
        return {
            'is_consistent': len(issues) == 0,
            'total_copies': len(all_copies),
            'reference': reference_path,
            'issues': issues
        }
    
    def get_sync_stats(self) -> Dict:
        """
        Senkronizasyon istatistikleri
        """
        
        total_loras = len(self.lora_copy_map)
        total_copies = sum(len(data['copies']) for data in self.lora_copy_map.values())
        
        return {
            'total_loras_tracked': total_loras,
            'total_copies_tracked': total_copies,
            'total_syncs_performed': self.sync_count,
            'average_copies_per_lora': total_copies / total_loras if total_loras > 0 else 0
        }


# Global instance
lora_sync_coordinator = LoRASyncCoordinator()

