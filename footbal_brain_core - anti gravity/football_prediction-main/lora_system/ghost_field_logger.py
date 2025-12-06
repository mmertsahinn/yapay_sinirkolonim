"""
👻 GHOST FIELD LOGGER - Hayalet Alan Etkileri
==============================================

Ghost Field (Hayalet Alanlar) etkilerini detaylı loglar:
- Hangi LoRA'lara etki etti?
- Ne kadar etki etti?
- Hangi yönde etki etti?
- En yakın ata kimdi?
- Ghost potential değişimleri

Her maç sonrası etkilenen LoRA'lar rapor edilir!
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd


class GhostFieldLogger:
    """
    Ghost Field etkilerini loglar
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Log dosyaları
        self.ghost_log_file = os.path.join(log_dir, "👻_GHOST_FIELD_EFFECTS.log")
        self.ghost_json_file = os.path.join(log_dir, "ghost_field_data.json")
        self.ghost_excel_file = os.path.join(log_dir, "👻_GHOST_FIELD_EFFECTS.xlsx")
        
        # Hafıza
        self.all_effects = []
        self.match_count = 0
        
        # İlk log
        self._write_header()
        
        print(f"👻 Ghost Field Logger başlatıldı: {log_dir}")
    
    def _write_header(self):
        """Log dosyasının başlığı"""
        with open(self.ghost_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("👻 GHOST FIELD (HAYALET ALANLAR) - ETKİ RAPORU\n")
            f.write("=" * 100 + "\n")
            f.write(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            f.write("MANTIK:\n")
            f.write("- Ölen LoRA'lar 'hayalet alan' oluşturur\n")
            f.write("- Yaşayan LoRA'lar bu alandan etkilenir\n")
            f.write("- Etki: Ataya saygı (çok sapmamak) + Özgürlük\n")
            f.write("- γ = 0.1 (Hafif bağ)\n")
            f.write("=" * 100 + "\n\n")
    
    def log_ghost_effects(self,
                         match_idx: int,
                         affected_loras: List[Dict],
                         total_ghosts: int,
                         strongest_ghost: Optional[Tuple[str, float]] = None):
        """
        Ghost Field etkilerini logla
        
        Args:
            match_idx: Maç numarası
            affected_loras: Etkilenen LoRA'lar
                [
                    {
                        'lora_name': str,
                        'lora_id': str,
                        'ghost_potential': float,
                        'closest_ancestor': (ancestor_id, distance),
                        'ancestor_respect_loss': float,
                        'effect_magnitude': float,  # Etkinin büyüklüğü
                        'effect_direction': str  # 'pull' veya 'push'
                    },
                    ...
                ]
            total_ghosts: Toplam hayalet sayısı
            strongest_ghost: En güçlü hayalet (id, influence_score)
        """
        self.match_count = match_idx
        
        if len(affected_loras) == 0 or total_ghosts == 0:
            return  # Etki yok, loglamaya gerek yok
        
        # Event kaydet
        event = {
            'match': match_idx,
            'timestamp': datetime.now().isoformat(),
            'total_ghosts': total_ghosts,
            'affected_loras_count': len(affected_loras),
            'strongest_ghost': strongest_ghost,
            'affected_loras': affected_loras
        }
        
        self.all_effects.append(event)
        
        # Text log
        with open(self.ghost_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "━" * 100 + "\n")
            f.write(f"👻 MAÇ #{match_idx} - GHOST FIELD ETKİLERİ\n")
            f.write("━" * 100 + "\n")
            f.write(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"👻 Toplam Hayalet: {total_ghosts}\n")
            f.write(f"🎯 Etkilenen LoRA: {len(affected_loras)}\n")
            
            if strongest_ghost:
                f.write(f"💪 En Güçlü Hayalet: {strongest_ghost[0][:12]}... (Etki: {strongest_ghost[1]:.3f})\n")
            
            f.write("\n📊 ETKİLENEN LoRA'LAR (En Yüksek Etkiden Başlayarak):\n")
            f.write("─" * 100 + "\n")
            
            # Etkiye göre sırala
            sorted_loras = sorted(affected_loras, 
                                 key=lambda x: x.get('effect_magnitude', 0), 
                                 reverse=True)
            
            for i, lora_data in enumerate(sorted_loras[:20], 1):  # İlk 20
                f.write(f"\n#{i}. {lora_data['lora_name']}\n")
                f.write(f"   📂 ID: {lora_data['lora_id']}\n")
                f.write(f"   🌊 Ghost Potential: {lora_data.get('ghost_potential', 0):.4f}\n")
                
                if lora_data.get('closest_ancestor'):
                    ancestor_id, distance = lora_data['closest_ancestor']
                    f.write(f"   👤 En Yakın Ata: {ancestor_id[:12]}... (Mesafe: {distance:.3f})\n")
                
                if lora_data.get('ancestor_respect_loss'):
                    f.write(f"   🙏 Ataya Saygı Loss: {lora_data['ancestor_respect_loss']:.6f}\n")
                
                effect_mag = lora_data.get('effect_magnitude', 0)
                effect_dir = lora_data.get('effect_direction', 'unknown')
                
                if effect_dir == 'pull':
                    f.write(f"   ⬅️  ETKİ: Ataya ÇEKİLİYOR (Magnitude: {effect_mag:.4f})\n")
                elif effect_dir == 'push':
                    f.write(f"   ➡️  ETKİ: Atadan UZAKLAŞIYOR (Magnitude: {effect_mag:.4f})\n")
                else:
                    f.write(f"   ↔️  ETKİ: Nötr (Magnitude: {effect_mag:.4f})\n")
                
                f.write("   " + "─" * 50 + "\n")
            
            if len(sorted_loras) > 20:
                f.write(f"\n   ... ve {len(sorted_loras) - 20} LoRA daha etkilendi.\n")
            
            f.write("\n" + "━" * 100 + "\n")
        
        # JSON kaydet (her 10 maçta)
        if match_idx % 10 == 0:
            self._save_json()
        
        # Excel kaydet (her 50 maçta)
        if match_idx % 50 == 0:
            self._save_excel()
    
    def log_ghost_registration(self, 
                              dead_lora_name: str,
                              dead_lora_id: str,
                              influence_score: float,
                              tes_score: float,
                              match_idx: int):
        """
        Yeni hayalet kaydını logla
        """
        with open(self.ghost_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "🆕" * 50 + "\n")
            f.write(f"👻 YENİ HAYALET KAYDI! (Maç #{match_idx})\n")
            f.write("🆕" * 50 + "\n")
            f.write(f"   💀 LoRA: {dead_lora_name}\n")
            f.write(f"   📂 ID: {dead_lora_id}\n")
            f.write(f"   💪 Etki Skoru: {influence_score:.3f}\n")
            f.write(f"   🌀 TES Skoru: {tes_score:.3f}\n")
            f.write(f"   ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("🆕" * 50 + "\n\n")
    
    def _save_json(self):
        """JSON formatında kaydet"""
        with open(self.ghost_json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_matches': self.match_count,
                'total_events': len(self.all_effects),
                'events': self.all_effects
            }, f, indent=2)
    
    def _save_excel(self):
        """Excel formatında kaydet"""
        if len(self.all_effects) == 0:
            return
        
        # DataFrame oluştur
        rows = []
        for event in self.all_effects:
            match_idx = event['match']
            total_ghosts = event['total_ghosts']
            affected_count = event['affected_loras_count']
            
            for lora_data in event['affected_loras']:
                rows.append({
                    'Maç': match_idx,
                    'Toplam Hayalet': total_ghosts,
                    'Etkilenen Toplam': affected_count,
                    'LoRA İsmi': lora_data['lora_name'],
                    'LoRA ID': lora_data['lora_id'],
                    'Ghost Potential': lora_data.get('ghost_potential', 0),
                    'Ataya Saygı Loss': lora_data.get('ancestor_respect_loss', 0),
                    'Etki Büyüklüğü': lora_data.get('effect_magnitude', 0),
                    'Etki Yönü': lora_data.get('effect_direction', 'unknown'),
                    'En Yakın Ata ID': lora_data.get('closest_ancestor', (None, None))[0],
                    'Ata Mesafesi': lora_data.get('closest_ancestor', (None, None))[1],
                    'Zaman': event['timestamp']
                })
        
        df = pd.DataFrame(rows)
        
        # Excel'e yaz (formatting ile!)
        with pd.ExcelWriter(self.ghost_excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Ghost Field Effects', index=False)
            
            # Format ayarları
            workbook = writer.book
            worksheet = writer.sheets['Ghost Field Effects']
            
            # Sütun genişlikleri
            worksheet.column_dimensions['A'].width = 8   # Maç
            worksheet.column_dimensions['B'].width = 15  # Toplam Hayalet
            worksheet.column_dimensions['C'].width = 15  # Etkilenen Toplam
            worksheet.column_dimensions['D'].width = 30  # LoRA İsmi
            worksheet.column_dimensions['E'].width = 20  # LoRA ID
            worksheet.column_dimensions['F'].width = 18  # Ghost Potential
            worksheet.column_dimensions['G'].width = 20  # Ataya Saygı Loss
            worksheet.column_dimensions['H'].width = 18  # Etki Büyüklüğü
            worksheet.column_dimensions['I'].width = 15  # Etki Yönü
            worksheet.column_dimensions['J'].width = 20  # En Yakın Ata
            worksheet.column_dimensions['K'].width = 15  # Ata Mesafesi
            worksheet.column_dimensions['L'].width = 25  # Zaman
        
        print(f"   👻 Ghost Field Excel güncellendi: {self.ghost_excel_file}")
    
    def get_summary(self) -> Dict:
        """Özet istatistikler"""
        if len(self.all_effects) == 0:
            return {
                'total_matches': 0,
                'total_events': 0,
                'total_affected_loras': 0,
                'avg_ghosts_per_match': 0,
                'avg_affected_per_match': 0
            }
        
        total_ghosts = sum(e['total_ghosts'] for e in self.all_effects)
        total_affected = sum(e['affected_loras_count'] for e in self.all_effects)
        
        return {
            'total_matches': len(self.all_effects),
            'total_events': len(self.all_effects),
            'total_affected_loras': total_affected,
            'avg_ghosts_per_match': total_ghosts / len(self.all_effects),
            'avg_affected_per_match': total_affected / len(self.all_effects)
        }


# Global instance
ghost_field_logger = GhostFieldLogger()

