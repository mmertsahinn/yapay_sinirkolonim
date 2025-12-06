"""
🧟 DİRİLTME DEBUG SİSTEMİ
=========================

Her diriltmeyi detaylı loglar ve debug eder!

LOGLAR:
- Hangi LoRA diriltildi?
- Lazarus Lambda skoru neydi?
- Hybrid tier'ı ne?
- Neden dirildi? (öncelik kriteri)
- Nereden geldi? (top list/mucize)
- Hangi klasörlere yerleşti?

AKIŞKAN DİRİLTME SİSTEMİ!
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple


class ResurrectionDebugger:
    """
    Diriltmeleri debug eder
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.log_dir = log_dir
        
        # Log dosyası
        self.debug_log = os.path.join(log_dir, "🧟_RESURRECTION_DEBUG.log")
        
        # Diriltme sayaçları
        self.resurrection_stats = {
            'total_resurrections': 0,
            'from_top_list': 0,
            'from_miracles': 0,
            'perfect_hybrids': 0,
            'strong_hybrids': 0,
            'high_lazarus': 0
        }
        
        self._write_header()
        
        print(f"🧟 Resurrection Debugger başlatıldı")
    
    def _write_header(self):
        """Log başlığı"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(self.debug_log, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("🧟 DİRİLTME DEBUG LOG - AKIŞKAN DİRİLTME SİSTEMİ\n")
            f.write("=" * 120 + "\n")
            f.write(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 120 + "\n\n")
            f.write("AKIŞKAN DİRİLTME:\n")
            f.write("  • Perfect Hybrid: +0.30 Lazarus bonusu (ÖNCELİK!)\n")
            f.write("  • Strong Hybrid: +0.15 Lazarus bonusu\n")
            f.write("  • Yüksek Lazarus Λ: Öğrenme potansiyeli yüksek\n")
            f.write("  • Mucizeler önce, sonra Top List\n")
            f.write("=" * 120 + "\n\n")
    
    def log_resurrection_batch(self,
                               match_idx: int,
                               resurrected_loras: List,
                               source: str,  # 'MIRACLES' | 'TOP_LIST' | 'SPAWN'
                               lazarus_scores: Dict = None):
        """
        Toplu diriltmeyi logla
        
        Args:
            resurrected_loras: Dirilen LoRA listesi
            source: Nereden geldi
            lazarus_scores: {lora_id: (lambda, final_score, type)}
        """
        
        try:
            print(f"   🔍 DEBUG: Diriltme logu yazılıyor...")
            print(f"      • Kaynak: {source}")
            print(f"      • Sayı: {len(resurrected_loras)}")
        except:
            pass
        
        try:
            with open(self.debug_log, 'a', encoding='utf-8') as f:
                f.write("\n" + "━" * 120 + "\n")
                f.write(f"🧟 MAÇ #{match_idx} - DİRİLTME BATCH ({source})\n")
                f.write("━" * 120 + "\n")
                f.write(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"📊 Toplam Dirilen: {len(resurrected_loras)}\n\n")
                
                for i, lora in enumerate(resurrected_loras, 1):
                    f.write(f"#{i}. {lora.name}\n")
                    f.write(f"   📂 ID: {lora.id}\n")
                    f.write(f"   💪 Fitness: {lora.get_recent_fitness():.3f}\n")
                    
                    # Lazarus skoru
                    if lazarus_scores and lora.id in lazarus_scores:
                        lam, final, lora_type = lazarus_scores[lora.id]
                        f.write(f"   🧟 Lazarus Λ: {lam:.3f}\n")
                        f.write(f"   🎯 Final Skor: {final:.3f}\n")
                        f.write(f"   🔬 TES Tip: {lora_type}\n")
                        
                        # Hybrid tier
                        if 'PERFECT HYBRID💎💎💎' in lora_type:
                            f.write(f"   💎 PERFECT HYBRID! (+0.30 bonus)\n")
                            self.resurrection_stats['perfect_hybrids'] += 1
                        elif 'STRONG HYBRID🌟🌟' in lora_type:
                            f.write(f"   🌟 STRONG HYBRID! (+0.15 bonus)\n")
                            self.resurrection_stats['strong_hybrids'] += 1
                        
                        if lam >= 0.70:
                            f.write(f"   ⚡ YÜKSEK LAZARUS! (Yüksek öğrenme potansiyeli)\n")
                            self.resurrection_stats['high_lazarus'] += 1
                    
                    f.write(f"   📍 Kaynak: {source}\n")
                    f.write("   " + "─" * 100 + "\n")
                
                f.write("\n" + "━" * 120 + "\n")
            
            print(f"      ✅ Diriltme logu kaydedildi")
            
        except Exception as e:
            print(f"      ❌ HATA: Diriltme logu yazılamadı!")
            print(f"      ❌ Hata: {str(e)}")
        
        # Sayaçları güncelle
        try:
            self.resurrection_stats['total_resurrections'] += len(resurrected_loras)
            
            if source == 'MIRACLES':
                self.resurrection_stats['from_miracles'] += len(resurrected_loras)
            elif source == 'TOP_LIST':
                self.resurrection_stats['from_top_list'] += len(resurrected_loras)
        except Exception as e:
            print(f"      ⚠️  İstatistik güncellenemedi: {str(e)}")
        
        # Console debug
        print(f"\n   🧟 DİRİLTME DEBUG:")
        print(f"      • Toplam: {len(resurrected_loras)} LoRA")
        print(f"      • Kaynak: {source}")
        if lazarus_scores:
            perfect_count = sum(1 for lam, fin, typ in lazarus_scores.values() if 'PERFECT HYBRID💎💎💎' in typ)
            if perfect_count > 0:
                print(f"      • 💎 Perfect Hybrid: {perfect_count} LoRA (öncelikli!)")
    
    def print_resurrection_summary(self):
        """
        Diriltme özetini print et
        """
        
        print(f"\n🧟 DİRİLTME İSTATİSTİKLERİ:")
        print("─" * 100)
        print(f"   Toplam Dirilen: {self.resurrection_stats['total_resurrections']}")
        print(f"   Mucizelerden: {self.resurrection_stats['from_miracles']}")
        print(f"   Top List'ten: {self.resurrection_stats['from_top_list']}")
        print(f"   💎 Perfect Hybrid: {self.resurrection_stats['perfect_hybrids']}")
        print(f"   🌟 Strong Hybrid: {self.resurrection_stats['strong_hybrids']}")
        print(f"   ⚡ Yüksek Lazarus: {self.resurrection_stats['high_lazarus']}")
        print("─" * 100)


# Global instance
resurrection_debugger = ResurrectionDebugger()

