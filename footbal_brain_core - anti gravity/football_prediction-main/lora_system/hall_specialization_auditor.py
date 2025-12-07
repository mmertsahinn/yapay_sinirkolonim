"""
🔬 HALL & UZMANLIK AUDIT SİSTEMİ
==================================

Hall of Fame ve Uzmanlık sistemlerini detaylı kontrol eder:
- Kategorisiz kalan LoRA'ları bulur
- Yanlış kategorilendirmeleri tespit eder
- Superhybrid ve özel durumları yakalar
- Eksik PT dosyalarını raporlar
- Otomatik düzeltme önerileri sunar

SORUN: Superhybrid var ama dosyada görünmüyor!
ÇÖZÜM: Her LoRA'yı debug modda kontrol et!
"""

import os
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class HallSpecializationAuditor:
    """
    Hall ve Uzmanlık sistemlerinin denetçisi
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Audit log
        self.audit_log = os.path.join(log_dir, "🔬_HALL_SPEC_AUDIT.log")
        
        # Sorunlu LoRA'lar
        self.uncategorized_loras = []
        self.miscategorized_loras = []
        self.missing_files = []
        self.superhybrids = []
        
        self._write_header()
        
        print(f"🔬 Hall & Specialization Auditor başlatıldı")
    
    def _write_header(self):
        """Audit log başlığı"""
        with open(self.audit_log, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("🔬 HALL & UZMANLIK AUDIT RAPORU\n")
            f.write("=" * 120 + "\n")
            f.write(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 120 + "\n\n")
            f.write("AMAÇ:\n")
            f.write("  • Kategorisiz LoRA'ları bul\n")
            f.write("  • Yanlış kategorileri tespit et\n")
            f.write("  • Superhybrid ve özel durumları yakala\n")
            f.write("  • Eksik dosyaları raporla\n")
            f.write("=" * 120 + "\n\n")
    
    def full_audit(self,
                   match_idx: int,
                   population: List,
                   all_loras_ever: Dict,
                   miracle_system,
                   tes_triple_scoreboard,
                   team_spec_manager,
                   global_spec_manager) -> Dict:
        """
        KAPSAMLI AUDIT! Her LoRA'yı tek tek kontrol et!
        """
        
        print(f"\n🔬 HALL & UZMANLIK AUDIT BAŞLIYOR (Maç #{match_idx})...")
        
        # Temizle
        self.uncategorized_loras = []
        self.miscategorized_loras = []
        self.missing_files = []
        self.superhybrids = []
        
        # 1) TÜM LORA'LARI KONTROL ET
        all_lora_categories = self._categorize_all_loras(
            population, all_loras_ever, miracle_system, match_idx,
            team_spec_manager, global_spec_manager
        )
        
        # 2) TES HALLs KONTROL
        tes_hall_status = self._audit_tes_halls(
            population, tes_triple_scoreboard
        )
        
        # 3) UZMANLIK SİSTEMLERİ KONTROL
        specialization_status = self._audit_specializations(
            population, team_spec_manager, global_spec_manager
        )
        
        # 4) DOSYA TUTARLILIĞI KONTROL
        file_status = self._audit_files(all_lora_categories)
        
        # Rapor oluştur
        report = {
            'match': match_idx,
            'timestamp': datetime.now().isoformat(),
            'total_loras': len(population),
            'categorizations': all_lora_categories,
            'tes_halls': tes_hall_status,
            'specializations': specialization_status,
            'files': file_status,
            'uncategorized_count': len(self.uncategorized_loras),
            'miscategorized_count': len(self.miscategorized_loras),
            'missing_files_count': len(self.missing_files),
            'superhybrid_count': len(self.superhybrids)
        }
        
        # Log yaz
        self._write_audit_report(report)
        
        return report
    
    def _categorize_all_loras(self,
                              population: List,
                              all_loras_ever: Dict,
                              miracle_system,
                              match_idx: int,
                              team_spec_manager,
                              global_spec_manager) -> Dict:
        """
        Her LoRA'yı kategorilere ayır (DEBUG MODE!)
        """
        categories = defaultdict(list)
        
        for lora in population:
            lora_categories = []
            
            # TES TİPİ
            if hasattr(lora, '_tes_scores'):
                tes_type = lora._tes_scores.get('lora_type', 'Unknown')
                lora_categories.append(f"TES:{tes_type}")
                categories[f"TES_{tes_type}"].append(lora)
            else:
                lora_categories.append("TES:HESAPLANMADI")
            
            # MUCİZE
            miracle_check = miracle_system.check_miracle_criteria(lora, match_idx)
            if miracle_check['is_miracle']:
                tier = miracle_check['miracle_tier']
                lora_categories.append(f"MIRACLE:{tier}")
                categories[f"MIRACLE_{tier}"].append(lora)
            
            # UZMANLIKLAR (kaç tane?)
            spec_count = 0
            
            # Takım uzmanlıkları
            for team_name, team_stats in team_spec_manager.team_stats.items():
                for spec_type, predictions in team_stats.items():
                    if spec_type == 'vs_predictions':
                        for opponent, vs_preds in predictions.items():
                            lora_matches = [p for p in vs_preds if p[0] == lora.id]
                            if len(lora_matches) >= 5:
                                spec_count += 1
                                lora_categories.append(f"SPEC:TEAM_{team_name}_{spec_type}_{opponent}")
                    else:
                        lora_matches = [p for p in predictions if p[0] == lora.id]
                        if len(lora_matches) >= 20:
                            spec_count += 1
                            lora_categories.append(f"SPEC:TEAM_{team_name}_{spec_type}")
            
            # Genel uzmanlıklar
            for spec_type, predictions in global_spec_manager.all_match_stats.items():
                lora_matches = [p for p in predictions if p[0] == lora.id]
                if len(lora_matches) >= 20:
                    spec_count += 1
                    lora_categories.append(f"SPEC:GLOBAL_{spec_type}")
            
            # SUPERHYBRID kontrol (5+ uzmanlık!)
            if spec_count >= 5:
                self.superhybrids.append((lora, spec_count, lora_categories))
                lora_categories.append(f"SUPERHYBRID:{spec_count}")
                categories["SUPERHYBRID"].append(lora)
            
            # KATEGORİSİZ mi?
            if len(lora_categories) == 0 or (len(lora_categories) == 1 and "TES:HESAPLANMADI" in lora_categories):
                self.uncategorized_loras.append((lora, lora_categories))
                categories["UNCATEGORIZED"].append(lora)
            else:
                categories["CATEGORIZED"].append(lora)
            
            # LoRA'ya ata (debug için)
            lora._audit_categories = lora_categories
        
        return dict(categories)
    
    def _audit_tes_halls(self,
                        population: List,
                        tes_triple_scoreboard) -> Dict:
        """
        TES Hall'leri kontrol et
        """
        halls = {
            'Einstein': 'en_iyi_loralar/🌟_EINSTEIN_HALL',
            'Newton': 'en_iyi_loralar/🏛️_NEWTON_HALL',
            'Darwin': 'en_iyi_loralar/🧬_DARWIN_HALL',
            'Hybrid': 'en_iyi_loralar/🌈_HYBRID_HALL',
            'PerfectHybrid': 'en_iyi_loralar/💎_PERFECT_HYBRID_HALL'
        }
        
        status = {}
        
        for hall_name, hall_dir in halls.items():
            if not os.path.exists(hall_dir):
                status[hall_name] = {
                    'exists': False,
                    'pt_count': 0,
                    'txt_exists': False
                }
                continue
            
            pt_files = [f for f in os.listdir(hall_dir) if f.endswith('.pt')]
            txt_files = [f for f in os.listdir(hall_dir) if f.endswith('.txt')]
            
            # PT dosyaları popülasyonla uyumlu mu?
            hall_lora_ids = set()
            for pt_file in pt_files:
                try:
                    data = torch.load(os.path.join(hall_dir, pt_file), map_location='cpu')
                    lora_id = data['metadata']['id']
                    hall_lora_ids.add(lora_id)
                except:
                    self.missing_files.append((hall_name, pt_file, "Okunamadı"))
            
            # Popülasyonda olup Hall'de olmayan
            expected_in_hall = []
            for lora in population:
                if hasattr(lora, '_tes_scores'):
                    lora_type = lora._tes_scores.get('lora_type', '')
                    
                    # Einstein Hall'e girmeli mi?
                    if hall_name == 'Einstein' and 'EINSTEIN' in lora_type:
                        expected_in_hall.append(lora.id)
                    elif hall_name == 'Newton' and 'NEWTON' in lora_type:
                        expected_in_hall.append(lora.id)
                    elif hall_name == 'Darwin' and 'DARWIN' in lora_type:
                        expected_in_hall.append(lora.id)
                    elif hall_name == 'Hybrid' and 'HYBRID' in lora_type and 'PERFECT' not in lora_type:
                        expected_in_hall.append(lora.id)
                    elif hall_name == 'PerfectHybrid' and 'PERFECT_HYBRID' in lora_type:
                        expected_in_hall.append(lora.id)
            
            missing_in_hall = set(expected_in_hall) - hall_lora_ids
            extra_in_hall = hall_lora_ids - set([lora.id for lora in population])
            
            status[hall_name] = {
                'exists': True,
                'pt_count': len(pt_files),
                'txt_exists': len(txt_files) > 0,
                'hall_lora_ids': len(hall_lora_ids),
                'expected_count': len(expected_in_hall),
                'missing_count': len(missing_in_hall),
                'extra_count': len(extra_in_hall),
                'missing_loras': list(missing_in_hall)[:5],  # İlk 5
                'extra_loras': list(extra_in_hall)[:5]
            }
            
            # Eksikleri raporla
            for missing_id in missing_in_hall:
                lora = next((l for l in population if l.id == missing_id), None)
                if lora:
                    self.miscategorized_loras.append((lora, hall_name, "Hall'de olmalı ama yok"))
        
        return status
    
    def _audit_specializations(self,
                              population: List,
                              team_spec_manager,
                              global_spec_manager) -> Dict:
        """
        Uzmanlık sistemlerini kontrol et
        """
        status = {
            'team_specializations': {},
            'global_specializations': {}
        }
        
        # Takım uzmanlıkları
        for team_name, team_stats in team_spec_manager.team_stats.items():
            team_dir = os.path.join(team_spec_manager.base_dir, team_spec_manager._safe_team_name(team_name))
            
            exists = os.path.exists(team_dir)
            pt_count = 0
            subdirs = []
            
            if exists:
                subdirs = [d for d in os.listdir(team_dir) if os.path.isdir(os.path.join(team_dir, d))]
                for subdir in subdirs:
                    subdir_path = os.path.join(team_dir, subdir)
                    pt_files = [f for f in os.listdir(subdir_path) if f.endswith('.pt')]
                    pt_count += len(pt_files)
            
            status['team_specializations'][team_name] = {
                'exists': exists,
                'subdirs': len(subdirs),
                'pt_count': pt_count
            }
        
        # Genel uzmanlıklar
        global_dir = os.path.join("en_iyi_loralar", "🌍_GENEL_UZMANLAR")
        if os.path.exists(global_dir):
            subdirs = [d for d in os.listdir(global_dir) if os.path.isdir(os.path.join(global_dir, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(global_dir, subdir)
                pt_files = [f for f in os.listdir(subdir_path) if f.endswith('.pt')]
                status['global_specializations'][subdir] = {
                    'exists': True,
                    'pt_count': len(pt_files)
                }
        
        return status
    
    def _audit_files(self, all_lora_categories: Dict) -> Dict:
        """
        Dosya tutarlılığını kontrol et
        """
        status = {
            'superhybrids_with_files': 0,
            'superhybrids_without_files': 0,
            'categorized_with_files': 0,
            'categorized_without_files': 0
        }
        
        # Superhybrid'leri kontrol et
        for lora, spec_count, categories in self.superhybrids:
            has_file = False
            
            # Herhangi bir Hall'de dosyası var mı?
            halls_to_check = [
                'en_iyi_loralar/🌟_EINSTEIN_HALL',
                'en_iyi_loralar/🏛️_NEWTON_HALL',
                'en_iyi_loralar/🧬_DARWIN_HALL',
                'en_iyi_loralar/🌈_HYBRID_HALL',
                'en_iyi_loralar/💎_PERFECT_HYBRID_HALL'
            ]
            
            for hall_dir in halls_to_check:
                if os.path.exists(hall_dir):
                    pt_files = [f for f in os.listdir(hall_dir) if f.endswith('.pt') and lora.id in f]
                    if pt_files:
                        has_file = True
                        break
            
            if has_file:
                status['superhybrids_with_files'] += 1
            else:
                status['superhybrids_without_files'] += 1
                self.missing_files.append(('SUPERHYBRID', lora.id, f"{spec_count} uzmanlık ama dosya yok!"))
        
        return status
    
    def _write_audit_report(self, report: Dict):
        """
        Audit raporunu yaz
        """
        with open(self.audit_log, 'a', encoding='utf-8') as f:
            f.write("\n" + "━" * 120 + "\n")
            f.write(f"🔬 MAÇ #{report['match']} - AUDIT RAPORU\n")
            f.write("━" * 120 + "\n")
            f.write(f"⏰ {report['timestamp']}\n")
            f.write(f"👥 Toplam LoRA: {report['total_loras']}\n")
            f.write("━" * 120 + "\n\n")
            
            # KATEGORİSİZ LoRA'LAR
            f.write(f"❌ KATEGORİSİZ LoRA'LAR: {report['uncategorized_count']}\n")
            f.write("-" * 120 + "\n")
            if self.uncategorized_loras:
                for lora, categories in self.uncategorized_loras[:10]:
                    f.write(f"   • {lora.name} (ID: {lora.id})\n")
                    f.write(f"      Kategoriler: {categories}\n")
                    f.write(f"      Fitness: {lora.get_recent_fitness():.3f}\n")
                    f.write(f"      Yaş: {report['match'] - lora.birth_match} maç\n")
                if len(self.uncategorized_loras) > 10:
                    f.write(f"   ... ve {len(self.uncategorized_loras) - 10} LoRA daha\n")
            f.write("\n")
            
            # SUPERHYBRID'LER
            f.write(f"⭐ SUPERHYBRID LoRA'LAR: {report['superhybrid_count']}\n")
            f.write("-" * 120 + "\n")
            if self.superhybrids:
                for lora, spec_count, categories in self.superhybrids[:10]:
                    f.write(f"   • {lora.name} (ID: {lora.id})\n")
                    f.write(f"      Uzmanlık Sayısı: {spec_count}\n")
                    f.write(f"      Kategoriler: {len(categories)}\n")
                    for cat in categories[:5]:
                        f.write(f"         - {cat}\n")
                    if len(categories) > 5:
                        f.write(f"         ... ve {len(categories) - 5} kategori daha\n")
            f.write("\n")
            
            # YANLIŞ KATEGORİLENDİRMELER
            f.write(f"⚠️ YANLIŞ KATEGORİLENDİRMELER: {report['miscategorized_count']}\n")
            f.write("-" * 120 + "\n")
            if self.miscategorized_loras:
                for lora, hall_name, reason in self.miscategorized_loras[:10]:
                    f.write(f"   • {lora.name} (ID: {lora.id})\n")
                    f.write(f"      Hall: {hall_name}\n")
                    f.write(f"      Sorun: {reason}\n")
            f.write("\n")
            
            # EKSİK DOSYALAR
            f.write(f"📁 EKSİK DOSYALAR: {report['missing_files_count']}\n")
            f.write("-" * 120 + "\n")
            if self.missing_files:
                for category, identifier, reason in self.missing_files[:10]:
                    f.write(f"   • {category}: {identifier}\n")
                    f.write(f"      Sorun: {reason}\n")
            f.write("\n")
            
            # TES HALLs DURUMU
            f.write("🔬 TES HALL DURUMU:\n")
            f.write("-" * 120 + "\n")
            for hall_name, hall_status in report['tes_halls'].items():
                if hall_status['exists']:
                    f.write(f"   {hall_name:20s}: {hall_status['pt_count']} PT dosyası\n")
                    if hall_status.get('missing_count', 0) > 0:
                        f.write(f"      ⚠️ {hall_status['missing_count']} LoRA eksik!\n")
                else:
                    f.write(f"   {hall_name:20s}: ❌ Dizin bulunamadı!\n")
            f.write("\n")
            
            f.write("━" * 120 + "\n")


# Global instance
hall_specialization_auditor = HallSpecializationAuditor()

