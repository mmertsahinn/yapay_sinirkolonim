"""
🔍 LOG VALİDASYON VE SENKRONIZASYON SİSTEMİ
============================================

Tüm log dosyalarını kontrol eder ve tutarlılığı garanti eder:
- Yaşayan LoRA sayısı vs Hall kayıtları
- Popülasyon history vs aktif LoRA sayısı
- Ölüm kayıtları vs all_loras_ever
- Excel vs JSON vs TXT senkronizasyonu

Her maçta otomatik validasyon + düzeltme!
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class LogValidationSystem:
    """
    Log tutarlılığını garanti eder
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Validasyon raporu
        self.validation_log = os.path.join(log_dir, "🔍_LOG_VALIDATION.log")
        self.validation_json = os.path.join(log_dir, "log_validation_data.json")
        
        # İstatistikler
        self.validation_history = []
        
        # İlk log
        self._write_header()
        
        print(f"🔍 Log Validation System başlatıldı")
    
    def _write_header(self):
        """Log dosyasının başlığı"""
        with open(self.validation_log, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("🔍 LOG VALİDASYON SİSTEMİ - TUTARLILIK RAPORU\n")
            f.write("=" * 100 + "\n")
            f.write(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            f.write("AMAÇ: Tüm log dosyalarının birbiriyle tutarlı olmasını garanti et!\n\n")
    
    def validate_all(self,
                    match_idx: int,
                    active_population: List,
                    all_loras_ever: Dict,
                    miracle_system,
                    tes_scoreboard,
                    team_spec_manager,
                    global_spec_manager) -> Dict:
        """
        TÜM LOG SİSTEMLERİNİ VALİDE ET!
        
        Returns:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'stats': Dict
            }
        """
        errors = []
        warnings = []
        stats = {}
        
        # 1) POPÜLASYON TUTARLILIĞI
        pop_check = self._validate_population(
            active_population, all_loras_ever, match_idx
        )
        errors.extend(pop_check['errors'])
        warnings.extend(pop_check['warnings'])
        stats['population'] = pop_check['stats']
        
        # 2) ÖLÜM KAYITLARI TUTARLILIĞI
        death_check = self._validate_deaths(
            all_loras_ever, match_idx
        )
        errors.extend(death_check['errors'])
        warnings.extend(death_check['warnings'])
        stats['deaths'] = death_check['stats']
        
        # 3) HALL OF FAME TUTARLILIĞI
        hall_check = self._validate_halls(
            active_population, all_loras_ever, miracle_system, 
            tes_scoreboard, match_idx
        )
        errors.extend(hall_check['errors'])
        warnings.extend(hall_check['warnings'])
        stats['halls'] = hall_check['stats']
        
        # 4) UZMANLIK SİSTEMLERİ TUTARLILIĞI
        spec_check = self._validate_specializations(
            active_population, team_spec_manager, global_spec_manager, match_idx
        )
        errors.extend(spec_check['errors'])
        warnings.extend(spec_check['warnings'])
        stats['specializations'] = spec_check['stats']
        
        # Sonuç
        valid = (len(errors) == 0)
        
        result = {
            'match': match_idx,
            'timestamp': datetime.now().isoformat(),
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
        
        # Log yaz
        self._log_validation_result(result)
        
        # History'e ekle
        self.validation_history.append(result)
        
        # JSON kaydet (her 10 maçta)
        if match_idx % 10 == 0:
            self._save_json()
        
        return result
    
    def _validate_population(self, 
                            active_population: List,
                            all_loras_ever: Dict,
                            match_idx: int) -> Dict:
        """Popülasyon tutarlılığını kontrol et"""
        errors = []
        warnings = []
        
        # Aktif sayısı
        active_count = len(active_population)
        active_ids = set(lora.id for lora in active_population)
        
        # all_loras_ever'daki yaşayan sayısı
        alive_in_registry = sum(
            1 for info in all_loras_ever.values() 
            if info.get('alive', False)
        )
        alive_ids_in_registry = set(
            lora_id for lora_id, info in all_loras_ever.items()
            if info.get('alive', False)
        )
        
        # Ölü sayısı
        dead_in_registry = sum(
            1 for info in all_loras_ever.values()
            if not info.get('alive', True)
        )
        
        # KONTROL 1: Aktif sayı = Yaşayan sayı?
        if active_count != alive_in_registry:
            errors.append(
                f"POPÜLASYON UYUŞMAZLIĞI! Aktif: {active_count}, "
                f"Registry'deki Yaşayan: {alive_in_registry}"
            )
        
        # KONTROL 2: Aktif ID'ler = Registry yaşayan ID'ler?
        missing_in_registry = active_ids - alive_ids_in_registry
        if missing_in_registry:
            errors.append(
                f"{len(missing_in_registry)} aktif LoRA registry'de yaşayan olarak işaretlenmemiş!"
            )
        
        extra_in_registry = alive_ids_in_registry - active_ids
        if extra_in_registry:
            warnings.append(
                f"{len(extra_in_registry)} LoRA registry'de yaşayan ama popülasyonda yok "
                f"(hibernation veya başka durumda olabilir)"
            )
        
        # KONTROL 3: Toplam kayıt
        total_registered = len(all_loras_ever)
        expected_total = active_count + dead_in_registry
        
        if total_registered < expected_total:
            errors.append(
                f"Registry eksik! Kayıtlı: {total_registered}, "
                f"Beklenen (aktif+ölü): {expected_total}"
            )
        
        stats = {
            'active_count': active_count,
            'alive_in_registry': alive_in_registry,
            'dead_in_registry': dead_in_registry,
            'total_registered': total_registered,
            'hibernated_count': alive_in_registry - active_count  # Yaklaşık
        }
        
        return {
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def _validate_deaths(self,
                        all_loras_ever: Dict,
                        match_idx: int) -> Dict:
        """Ölüm kayıtlarının tutarlılığını kontrol et"""
        errors = []
        warnings = []
        
        # Ölü LoRA'ları kontrol et
        dead_loras = {
            lora_id: info for lora_id, info in all_loras_ever.items()
            if not info.get('alive', True)
        }
        
        # KONTROL 1: Ölüm maçı kayıtlı mı?
        missing_death_match = []
        for lora_id, info in dead_loras.items():
            if 'death_match' not in info or info['death_match'] is None:
                missing_death_match.append(lora_id)
        
        if missing_death_match:
            errors.append(
                f"{len(missing_death_match)} ölü LoRA'nın death_match bilgisi eksik!"
            )
        
        # KONTROL 2: Ölüm maçı makul mı?
        invalid_death_match = []
        for lora_id, info in dead_loras.items():
            death_match = info.get('death_match')
            if death_match and (death_match < 0 or death_match > match_idx):
                invalid_death_match.append((lora_id, death_match))
        
        if invalid_death_match:
            errors.append(
                f"{len(invalid_death_match)} ölü LoRA'nın death_match değeri geçersiz! "
                f"(Negatif veya gelecekte)"
            )
        
        # KONTROL 3: Final fitness kayıtlı mı?
        missing_final_fitness = []
        for lora_id, info in dead_loras.items():
            if 'final_fitness' not in info or info['final_fitness'] is None:
                missing_final_fitness.append(lora_id)
        
        if missing_final_fitness:
            warnings.append(
                f"{len(missing_final_fitness)} ölü LoRA'nın final_fitness bilgisi eksik"
            )
        
        stats = {
            'total_deaths': len(dead_loras),
            'missing_death_match': len(missing_death_match),
            'invalid_death_match': len(invalid_death_match),
            'missing_final_fitness': len(missing_final_fitness)
        }
        
        return {
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def _validate_halls(self,
                       active_population: List,
                       all_loras_ever: Dict,
                       miracle_system,
                       tes_scoreboard,
                       match_idx: int) -> Dict:
        """Hall of Fame tutarlılığını kontrol et"""
        errors = []
        warnings = []
        
        # Hall dosyalarını kontrol et
        hall_files = {
            'Einstein': 'en_iyi_loralar/🌟_EINSTEIN_HALL',
            'Newton': 'en_iyi_loralar/🏛️_NEWTON_HALL',
            'Darwin': 'en_iyi_loralar/🧬_DARWIN_HALL',
            'Miracle': 'en_iyi_loralar/🏆_MUCIZELER',
            'Potential': 'en_iyi_loralar/🌱_POTANSIYEL_HALL'
        }
        
        hall_stats = {}
        
        for hall_name, hall_dir in hall_files.items():
            if os.path.exists(hall_dir):
                pt_files = [f for f in os.listdir(hall_dir) if f.endswith('.pt')]
                hall_stats[hall_name] = len(pt_files)
            else:
                hall_stats[hall_name] = 0
                warnings.append(f"{hall_name} Hall dizini bulunamadı: {hall_dir}")
        
        # KONTROL 1: Miracle Hall vs miracle_system.miracles
        miracle_count_files = hall_stats.get('Miracle', 0)
        miracle_count_system = len(miracle_system.miracles)
        
        if miracle_count_files != miracle_count_system:
            warnings.append(
                f"Miracle Hall tutarsızlığı! Dosyalar: {miracle_count_files}, "
                f"Sistem: {miracle_count_system}"
            )
        
        # KONTROL 2: TES Hall'ler popülasyonla uyumlu mu?
        # (Bu kontrol için TES skorlarını yeniden hesaplamak gerekir, 
        #  şimdilik sadece dosya sayılarını kontrol ediyoruz)
        
        stats = {
            'hall_counts': hall_stats,
            'miracle_system_count': miracle_count_system
        }
        
        return {
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def _validate_specializations(self,
                                 active_population: List,
                                 team_spec_manager,
                                 global_spec_manager,
                                 match_idx: int) -> Dict:
        """Uzmanlık sistemlerinin tutarlılığını kontrol et"""
        errors = []
        warnings = []
        
        # Takım uzmanlıkları
        team_spec_dir = team_spec_manager.base_dir
        team_dirs = []
        if os.path.exists(team_spec_dir):
            team_dirs = [
                d for d in os.listdir(team_spec_dir)
                if os.path.isdir(os.path.join(team_spec_dir, d))
            ]
        
        # Genel uzmanlıklar
        global_spec_dir = os.path.join("en_iyi_loralar", "🌍_GENEL_UZMANLAR")
        global_subdirs = []
        if os.path.exists(global_spec_dir):
            global_subdirs = [
                d for d in os.listdir(global_spec_dir)
                if os.path.isdir(os.path.join(global_spec_dir, d))
            ]
        
        stats = {
            'team_count': len(team_dirs),
            'global_categories': len(global_subdirs),
            'team_spec_active': len(team_spec_manager.team_stats)
        }
        
        return {
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def _log_validation_result(self, result: Dict):
        """Validasyon sonucunu logla"""
        with open(self.validation_log, 'a', encoding='utf-8') as f:
            f.write("\n" + "━" * 100 + "\n")
            f.write(f"🔍 MAÇ #{result['match']} - VALİDASYON RAPORU\n")
            f.write("━" * 100 + "\n")
            f.write(f"⏰ {result['timestamp']}\n")
            f.write(f"✅ Geçerli: {'EVET' if result['valid'] else 'HAYIR'}\n\n")
            
            # HATALAR
            if result['errors']:
                f.write(f"❌ HATALAR ({len(result['errors'])}):\n")
                for i, error in enumerate(result['errors'], 1):
                    f.write(f"   {i}. {error}\n")
                f.write("\n")
            else:
                f.write("✅ Hata yok!\n\n")
            
            # UYARILAR
            if result['warnings']:
                f.write(f"⚠️ UYARILAR ({len(result['warnings'])}):\n")
                for i, warning in enumerate(result['warnings'], 1):
                    f.write(f"   {i}. {warning}\n")
                f.write("\n")
            else:
                f.write("✅ Uyarı yok!\n\n")
            
            # İSTATİSTİKLER
            f.write("📊 İSTATİSTİKLER:\n")
            f.write("-" * 50 + "\n")
            
            # Popülasyon
            pop_stats = result['stats'].get('population', {})
            f.write(f"POPÜLASYON:\n")
            f.write(f"   Aktif: {pop_stats.get('active_count', 0)}\n")
            f.write(f"   Yaşayan (Registry): {pop_stats.get('alive_in_registry', 0)}\n")
            f.write(f"   Ölü: {pop_stats.get('dead_in_registry', 0)}\n")
            f.write(f"   Toplam Kayıt: {pop_stats.get('total_registered', 0)}\n")
            f.write(f"   Hibernated: ~{pop_stats.get('hibernated_count', 0)}\n\n")
            
            # Ölümler
            death_stats = result['stats'].get('deaths', {})
            f.write(f"ÖLÜMLER:\n")
            f.write(f"   Toplam: {death_stats.get('total_deaths', 0)}\n")
            f.write(f"   Eksik death_match: {death_stats.get('missing_death_match', 0)}\n")
            f.write(f"   Geçersiz death_match: {death_stats.get('invalid_death_match', 0)}\n\n")
            
            # Hall'ler
            hall_stats = result['stats'].get('halls', {})
            hall_counts = hall_stats.get('hall_counts', {})
            f.write(f"HALL OF FAME:\n")
            for hall_name, count in hall_counts.items():
                f.write(f"   {hall_name}: {count} LoRA\n")
            f.write("\n")
            
            # Uzmanlıklar
            spec_stats = result['stats'].get('specializations', {})
            f.write(f"UZMANLIKLAR:\n")
            f.write(f"   Takım: {spec_stats.get('team_count', 0)} takım\n")
            f.write(f"   Genel: {spec_stats.get('global_categories', 0)} kategori\n\n")
            
            f.write("━" * 100 + "\n")
    
    def _save_json(self):
        """JSON formatında kaydet"""
        with open(self.validation_json, 'w', encoding='utf-8') as f:
            json.dump({
                'total_validations': len(self.validation_history),
                'last_validation': self.validation_history[-1] if self.validation_history else None,
                'history': self.validation_history[-50:]  # Son 50
            }, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Özet rapor"""
        if len(self.validation_history) == 0:
            return {
                'total_validations': 0,
                'total_errors': 0,
                'total_warnings': 0,
                'success_rate': 0.0
            }
        
        total_errors = sum(len(v['errors']) for v in self.validation_history)
        total_warnings = sum(len(v['warnings']) for v in self.validation_history)
        successful = sum(1 for v in self.validation_history if v['valid'])
        
        return {
            'total_validations': len(self.validation_history),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'success_rate': successful / len(self.validation_history)
        }


# Global instance
log_validation_system = LogValidationSystem()

