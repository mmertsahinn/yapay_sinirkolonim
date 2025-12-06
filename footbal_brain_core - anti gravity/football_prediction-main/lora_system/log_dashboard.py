"""
📊 REAL-TIME LOG DASHBOARD
===========================

Tüm log sistemlerini tek bir yerden izle!
Her maçta otomatik güncellenir ve özet gösterir.
"""

import os
from datetime import datetime
from typing import Dict, List


class LogDashboard:
    """
    Real-time log monitoring dashboard
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.dashboard_file = os.path.join(log_dir, "📊_DASHBOARD.txt")
        
        print(f"📊 Log Dashboard başlatıldı")
    
    def update_dashboard(self,
                        match_idx: int,
                        population: List,
                        all_loras_ever: Dict,
                        validation_result: Dict,
                        ghost_field_summary: Dict,
                        miracle_count: int,
                        tes_distribution: Dict) -> None:
        """
        Dashboard'u güncelle (her maçta!)
        """
        
        with open(self.dashboard_file, 'w', encoding='utf-8') as f:
            # BAŞLIK
            f.write("=" * 100 + "\n")
            f.write("📊 REAL-TIME LOG DASHBOARD\n")
            f.write("=" * 100 + "\n")
            f.write(f"🕐 Son Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"⚽ Maç: #{match_idx}\n")
            f.write("=" * 100 + "\n\n")
            
            # 1) POPÜLASYON DURUMU
            f.write("👥 POPÜLASYON DURUMU:\n")
            f.write("-" * 100 + "\n")
            
            active_count = len(population)
            alive_in_registry = sum(1 for info in all_loras_ever.values() if info.get('alive', False))
            dead_count = sum(1 for info in all_loras_ever.values() if not info.get('alive', True))
            total_registered = len(all_loras_ever)
            
            f.write(f"   ⭐ Aktif Popülasyon: {active_count} LoRA\n")
            f.write(f"   💚 Yaşayan (Registry): {alive_in_registry} LoRA\n")
            f.write(f"   💀 Ölü: {dead_count} LoRA\n")
            f.write(f"   📊 Toplam Kayıt: {total_registered} LoRA (tüm zamanlar)\n")
            f.write(f"   🛌 Hibernated (Tahmini): {alive_in_registry - active_count} LoRA\n")
            
            # Fitness istatistikleri
            if active_count > 0:
                fitnesses = [lora.get_recent_fitness() for lora in population]
                avg_fitness = sum(fitnesses) / len(fitnesses)
                max_fitness = max(fitnesses)
                min_fitness = min(fitnesses)
                
                f.write(f"\n   📈 Fitness İstatistikleri:\n")
                f.write(f"      Ortalama: {avg_fitness:.3f}\n")
                f.write(f"      En Yüksek: {max_fitness:.3f}\n")
                f.write(f"      En Düşük: {min_fitness:.3f}\n")
            
            f.write("\n")
            
            # 2) VALİDASYON DURUMU
            f.write("🔍 LOG VALİDASYON DURUMU:\n")
            f.write("-" * 100 + "\n")
            
            if validation_result['valid']:
                f.write("   ✅ TÜM LOGLAR GEÇERLİ!\n")
            else:
                f.write(f"   ❌ {len(validation_result['errors'])} HATA VAR!\n")
                for error in validation_result['errors'][:5]:
                    f.write(f"      • {error}\n")
            
            if validation_result['warnings']:
                f.write(f"\n   ⚠️ {len(validation_result['warnings'])} UYARI:\n")
                for warning in validation_result['warnings'][:3]:
                    f.write(f"      • {warning}\n")
            
            f.write("\n")
            
            # 3) GHOST FIELD DURUMU
            f.write("👻 GHOST FIELD DURUMU:\n")
            f.write("-" * 100 + "\n")
            
            total_ghosts = ghost_field_summary.get('total_matches', 0)
            avg_ghosts = ghost_field_summary.get('avg_ghosts_per_match', 0)
            avg_affected = ghost_field_summary.get('avg_affected_per_match', 0)
            
            f.write(f"   👻 Toplam Hayalet: {total_ghosts}\n")
            f.write(f"   📊 Maç Başı Ortalama Hayalet: {avg_ghosts:.1f}\n")
            f.write(f"   🎯 Maç Başı Etkilenen LoRA: {avg_affected:.1f}\n")
            f.write("\n")
            
            # 4) HALL OF FAME
            f.write("🏆 HALL OF FAME DURUMU:\n")
            f.write("-" * 100 + "\n")
            
            hall_counts = validation_result['stats'].get('halls', {}).get('hall_counts', {})
            
            f.write(f"   🏆 Mucizeler: {miracle_count} LoRA\n")
            f.write(f"   🌟 Einstein Hall: {hall_counts.get('Einstein', 0)} LoRA\n")
            f.write(f"   🏛️ Newton Hall: {hall_counts.get('Newton', 0)} LoRA\n")
            f.write(f"   🧬 Darwin Hall: {hall_counts.get('Darwin', 0)} LoRA\n")
            f.write(f"   🌱 Potansiyel Hall: {hall_counts.get('Potential', 0)} LoRA\n")
            
            # TES Dağılımı
            if tes_distribution:
                f.write(f"\n   🔬 TES TİPİ DAĞILIMI:\n")
                for tes_type, count in tes_distribution.items():
                    percentage = (count / active_count * 100) if active_count > 0 else 0
                    bar = "█" * int(percentage / 5)
                    f.write(f"      {tes_type:20s}: {count:3d} LoRA ({percentage:5.1f}%) {bar}\n")
            
            f.write("\n")
            
            # 5) UZMANLIK SİSTEMLERİ
            f.write("🎯 UZMANLIK SİSTEMLERİ:\n")
            f.write("-" * 100 + "\n")
            
            spec_stats = validation_result['stats'].get('specializations', {})
            
            f.write(f"   📍 Takım Uzmanlıkları: {spec_stats.get('team_count', 0)} takım\n")
            f.write(f"   🌍 Genel Uzmanlıklar: {spec_stats.get('global_categories', 0)} kategori\n")
            f.write(f"   📊 Aktif Takım İstatistiği: {spec_stats.get('team_spec_active', 0)}\n")
            f.write("\n")
            
            # 6) LOG DOSYALARI DURUMU
            f.write("📁 LOG DOSYALARI DURUMU:\n")
            f.write("-" * 100 + "\n")
            
            log_files = {
                'Evolution Log': 'evolution_log.txt',
                'Match Results': 'match_results.log',
                'Ghost Field': '👻_GHOST_FIELD_EFFECTS.log',
                'Death Report (Excel)': 'OLUM_RAPORU_CANLI.xlsx',
                'Population History (Excel)': 'population_history_DETAYLI.xlsx',
                'Validation Log': '🔍_LOG_VALIDATION.log'
            }
            
            for log_name, log_file in log_files.items():
                full_path = os.path.join(self.log_dir, log_file)
                if os.path.exists(full_path):
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    modified = datetime.fromtimestamp(os.path.getmtime(full_path))
                    f.write(f"   ✅ {log_name:30s}: {size_mb:6.2f} MB (Son: {modified.strftime('%H:%M:%S')})\n")
                else:
                    f.write(f"   ❌ {log_name:30s}: Bulunamadı!\n")
            
            f.write("\n")
            
            # 7) PERFORMANS METRİKLERİ
            f.write("⚡ PERFORMANS METRİKLERİ:\n")
            f.write("-" * 100 + "\n")
            
            # Son 50 maç başarı oranı
            if active_count > 0:
                recent_success = []
                for lora in population:
                    if len(lora.fitness_history) > 0:
                        recent = lora.fitness_history[-50:]
                        success_rate = sum(1 for f in recent if f > 0.5) / len(recent)
                        recent_success.append(success_rate)
                
                if recent_success:
                    avg_success = sum(recent_success) / len(recent_success)
                    f.write(f"   📊 Ortalama Başarı Oranı (Son 50 Maç): {avg_success:.1%}\n")
            
            f.write(f"   📈 Toplam Maç İşlendi: {match_idx}\n")
            f.write(f"   🔄 Toplam LoRA Yaratıldı: {total_registered}\n")
            f.write(f"   💀 Toplam Ölüm: {dead_count}\n")
            f.write(f"   ⚡ Hayatta Kalma Oranı: {(alive_in_registry / total_registered * 100) if total_registered > 0 else 0:.1f}%\n")
            
            f.write("\n")
            f.write("=" * 100 + "\n")
            f.write("📊 Dashboard otomatik güncellenir (her maçta)\n")
            f.write("=" * 100 + "\n")
        
        # Console'a kısa özet
        print(f"\n📊 DASHBOARD GÜNCELLENDİ:")
        print(f"   ⭐ Aktif: {active_count} | 💀 Ölü: {dead_count} | 🏆 Mucize: {miracle_count}")
        if not validation_result['valid']:
            print(f"   ⚠️ {len(validation_result['errors'])} validasyon hatası var!")


# Global instance
log_dashboard = LogDashboard()

