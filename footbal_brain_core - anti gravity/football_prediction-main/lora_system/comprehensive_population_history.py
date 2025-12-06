"""
📚 KAPSAMLI POPÜLASYON TARİHİ
==============================

HER LoRA'NIN YAPTIĞI HER ŞEY BURADA!

İÇERİK:
- Her maç sonrası durum
- Rol değişiklikleri
- Takım uzmanlıkları
- TES skorları
- Fiziksel özellikler
- Hibernation durumu
- Diriltmeler
- Ölümler
- Her şey!

AMAÇ: Hiçbir bilgi kaybolmasın!
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict


class ComprehensivePopulationHistory:
    """
    Her LoRA'nın kapsamlı tarihini tutar
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.log_dir = log_dir
        
        # Ana history dosyası
        self.history_file = os.path.join(log_dir, "📚_POPULATION_HISTORY.json")
        self.txt_file = os.path.join(log_dir, "📚_POPULATION_HISTORY.txt")
        
        # Her LoRA için bireysel history {lora_id: [events]}
        self.lora_histories = defaultdict(list)
        
        # Maç bazlı snapshot {match_idx: population_state}
        self.match_snapshots = {}
        
        # İstatistikler
        self.stats = {
            'total_events': 0,
            'total_loras': 0,
            'match_count': 0
        }
        
        # Mevcut history'yi yükle (varsa)
        self._load_existing_history()
        
        print(f"📚 Comprehensive Population History başlatıldı")
    
    def _load_existing_history(self):
        """Mevcut history'yi yükle"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.lora_histories = defaultdict(list, data.get('lora_histories', {}))
                    self.match_snapshots = data.get('match_snapshots', {})
                    self.stats = data.get('stats', self.stats)
                print(f"   📚 Mevcut history yüklendi: {self.stats['total_events']} olay")
            except:
                pass
    
    def record_match_snapshot(self, match_idx: int, population: List, hibernated_count: int = 0):
        """
        Maç sonrası popülasyon snapshot'ı
        """
        
        # 🔍 DEBUG: HER MAÇTA göster! (Geçici - test için)
        print(f"      🔍 DEBUG [Population History - Snapshot]: Maç #{match_idx}")
        print(f"         • Aktif: {len(population)} | Uyuyan: {hibernated_count}")
        
        snapshot = {
            'match_idx': match_idx,
            'timestamp': datetime.now().isoformat(),
            'active_count': len(population),
            'hibernated_count': hibernated_count,
            'total_count': len(population) + hibernated_count,
            'loras': []
        }
        
        for lora in population:
            lora_data = {
                'id': lora.id,
                'name': lora.name,
                'generation': lora.generation,
                'age': match_idx - lora.birth_match,
                'fitness': round(lora.get_recent_fitness(), 3),
                'life_energy': round(getattr(lora, 'life_energy', 1.0), 3),
                'tes_type': getattr(lora, '_tes_scores', {}).get('lora_type', 'Unknown'),
                'status': 'ACTIVE'
            }
            snapshot['loras'].append(lora_data)
        
        self.match_snapshots[str(match_idx)] = snapshot
        self.stats['match_count'] = match_idx
    
    def record_lora_event(self, lora_id: str, lora_name: str, match_idx: int, event_type: str, details: Dict):
        """
        LoRA için olay kaydet
        
        Event Types:
        - BIRTH: Doğum
        - DEATH: Ölüm
        - RESURRECTION: Diriltme
        - HIBERNATION: Uyumaya gitti
        - WAKE_UP: Uyandı
        - ROLE_CHANGE: Rol değişikliği
        - SPECIALIZATION_GAINED: Takım uzmanlığı kazandı
        - SPECIALIZATION_LOST: Takım uzmanlığı kaybetti
        - TES_UPDATE: TES skoru güncellendi
        - PREDICTION: Tahmin yaptı
        - CORRECT_PREDICTION: Doğru tahmin
        - WRONG_PREDICTION: Yanlış tahmin
        """
        
        # 🔍 DEBUG: Olay kaydediliyor
        if match_idx % 10 == 0 and event_type in ['CORRECT_PREDICTION', 'WRONG_PREDICTION']:
            print(f"      🔍 DEBUG [Population History]: {event_type} kaydediliyor → {lora_name[:20]}")
        
        event = {
            'match_idx': match_idx,
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'lora_name': lora_name,
            'details': details
        }
        
        self.lora_histories[lora_id].append(event)
        self.stats['total_events'] += 1
        
        # Unique LoRA sayısını güncelle
        self.stats['total_loras'] = len(self.lora_histories)
        
        # 🔍 DEBUG: İstatistik
        if match_idx % 10 == 0 and self.stats['total_events'] % 100 == 0:
            print(f"      🔍 DEBUG [Population History]: Toplam {self.stats['total_events']} olay kaydedildi!")
    
    def record_role_change(self, lora, match_idx: int, added_roles: List[str], removed_roles: List[str]):
        """
        Rol değişikliğini kaydet
        """
        
        if not added_roles and not removed_roles:
            return
        
        details = {
            'added': added_roles,
            'removed': removed_roles,
            'tes_type': getattr(lora, '_tes_scores', {}).get('lora_type', 'Unknown'),
            'fitness': round(lora.get_recent_fitness(), 3)
        }
        
        self.record_lora_event(lora.id, lora.name, match_idx, 'ROLE_CHANGE', details)
    
    def record_specialization_change(self, lora, match_idx: int, spec_type: str, team_name: str, gained: bool, score: float):
        """
        Takım uzmanlığı değişikliğini kaydet
        """
        
        event_type = 'SPECIALIZATION_GAINED' if gained else 'SPECIALIZATION_LOST'
        
        details = {
            'spec_type': spec_type,
            'team': team_name,
            'score': round(score, 3),
            'fitness': round(lora.get_recent_fitness(), 3)
        }
        
        self.record_lora_event(lora.id, lora.name, match_idx, event_type, details)
    
    def record_prediction(self, lora, match_idx: int, prediction: str, actual: str, is_correct: bool, confidence: float):
        """
        Tahmin kaydet
        """
        
        # 🔍 DEBUG: İlk birkaç tahmin
        if self.stats['total_events'] < 5:
            print(f"      🔍 DEBUG [record_prediction]: İlk tahmin kaydı!")
            print(f"         LoRA: {lora.name[:25]}")
            print(f"         Tahmin: {prediction} → Gerçek: {actual}")
            print(f"         Doğru: {is_correct}")
        
        event_type = 'CORRECT_PREDICTION' if is_correct else 'WRONG_PREDICTION'
        
        details = {
            'prediction': prediction,
            'actual': actual,
            'confidence': round(confidence, 3),
            'fitness_after': round(lora.get_recent_fitness(), 3)
        }
        
        self.record_lora_event(lora.id, lora.name, match_idx, event_type, details)
    
    def save_history(self, match_idx: int):
        """
        History'yi kaydet (JSON + TXT)
        """
        
        # 🔍 DEBUG: HER MAÇTA göster!
        print(f"      🔍 DEBUG [Population History - Save]: Kaydediliyor...")
        print(f"         • Toplam LoRA: {self.stats['total_loras']}")
        print(f"         • Toplam Olay: {self.stats['total_events']}")
        
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # JSON (tam veri)
            data = {
                'generated_at': datetime.now().isoformat(),
                'current_match': match_idx,
                'stats': self.stats,
                'lora_histories': dict(self.lora_histories),
                'match_snapshots': self.match_snapshots
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"         ✅ JSON kaydedildi: {self.history_file}")
            
            # TXT (insan okunabilir)
            self._generate_txt_report(match_idx)
            print(f"         ✅ TXT kaydedildi: {self.txt_file}")
            
        except Exception as e:
            print(f"         ❌ HATA: Population history kaydedilemedi!")
            print(f"         ❌ Hata: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _generate_txt_report(self, match_idx: int):
        """
        İnsan okunabilir TXT raporu
        """
        
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("📚 KAPSAMLI POPÜLASYON TARİHİ\n")
            f.write("=" * 120 + "\n")
            f.write(f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Güncel Maç: #{match_idx}\n")
            f.write("=" * 120 + "\n\n")
            
            f.write("📊 İSTATİSTİKLER:\n")
            f.write(f"   • Toplam LoRA: {self.stats['total_loras']}\n")
            f.write(f"   • Toplam Olay: {self.stats['total_events']}\n")
            f.write(f"   • Maç Sayısı: {self.stats['match_count']}\n\n")
            
            # En aktif LoRA'lar
            f.write("🌟 EN AKTİF LoRA'LAR (En çok olay):\n")
            sorted_loras = sorted(self.lora_histories.items(), key=lambda x: len(x[1]), reverse=True)
            
            for i, (lora_id, events) in enumerate(sorted_loras[:10], 1):
                lora_name = events[0]['lora_name'] if events else 'Unknown'
                event_count = len(events)
                
                # Olay tipi dağılımı
                event_types = defaultdict(int)
                for event in events:
                    event_types[event['event_type']] += 1
                
                f.write(f"   {i}. {lora_name[:30]:30s} | {event_count:4d} olay\n")
                f.write(f"      {dict(event_types)}\n")
            
            f.write("\n" + "=" * 120 + "\n\n")
            
            # Her LoRA'nın detaylı geçmişi (son 20 olay)
            f.write("📖 DETAYLI LoRA GEÇMİŞLERİ (Her LoRA'nın son 20 olayı):\n\n")
            
            for lora_id, events in sorted_loras[:50]:  # İlk 50 LoRA
                lora_name = events[0]['lora_name'] if events else 'Unknown'
                
                f.write("─" * 120 + "\n")
                f.write(f"LoRA: {lora_name} (ID: {lora_id[:16]}...)\n")
                f.write(f"Toplam Olay: {len(events)}\n")
                f.write("─" * 120 + "\n")
                
                # Son 20 olay
                recent_events = events[-20:] if len(events) > 20 else events
                
                for event in recent_events:
                    match = event['match_idx']
                    event_type = event['event_type']
                    details = event['details']
                    
                    # Emoji seç
                    emoji = self._get_event_emoji(event_type)
                    
                    f.write(f"   Maç #{match:4d} | {emoji} {event_type:25s} | ")
                    
                    # Detayları yaz
                    if event_type == 'ROLE_CHANGE':
                        f.write(f"Added: {details.get('added', [])} | Removed: {details.get('removed', [])}\n")
                    elif event_type in ['SPECIALIZATION_GAINED', 'SPECIALIZATION_LOST']:
                        f.write(f"{details.get('team', '')} {details.get('spec_type', '')} (Skor: {details.get('score', 0):.3f})\n")
                    elif event_type in ['CORRECT_PREDICTION', 'WRONG_PREDICTION']:
                        f.write(f"{details.get('prediction', '')} → {details.get('actual', '')} (Güven: {details.get('confidence', 0):.3f})\n")
                    else:
                        f.write(f"{str(details)[:80]}\n")
                
                f.write("\n")
            
            f.write("=" * 120 + "\n")
    
    def _get_event_emoji(self, event_type: str) -> str:
        """Event tipine göre emoji"""
        emojis = {
            'BIRTH': '👶',
            'DEATH': '💀',
            'RESURRECTION': '⚡',
            'HIBERNATION': '😴',
            'WAKE_UP': '👁️',
            'ROLE_CHANGE': '🎭',
            'SPECIALIZATION_GAINED': '🎯',
            'SPECIALIZATION_LOST': '📉',
            'TES_UPDATE': '🔬',
            'CORRECT_PREDICTION': '✅',
            'WRONG_PREDICTION': '❌',
            'PREDICTION': '🔮'
        }
        return emojis.get(event_type, '📌')
    
    def get_lora_summary(self, lora_id: str) -> Dict:
        """
        Bir LoRA'nın özetini al
        """
        
        events = self.lora_histories.get(lora_id, [])
        
        if not events:
            return None
        
        # İstatistikler
        event_types = defaultdict(int)
        for event in events:
            event_types[event['event_type']] += 1
        
        # Son durum
        last_event = events[-1] if events else None
        
        return {
            'lora_name': events[0]['lora_name'],
            'total_events': len(events),
            'event_distribution': dict(event_types),
            'last_event': last_event,
            'first_event': events[0] if events else None
        }


# Global instance
comprehensive_history = ComprehensivePopulationHistory()

