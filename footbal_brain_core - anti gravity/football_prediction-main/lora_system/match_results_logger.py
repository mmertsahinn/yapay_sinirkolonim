"""
MAÇ SONUÇLARI LOGGER
====================

Sadece maç sonuçlarını detaylı loglar.
Her maç için:
- Tarih, saat, takımlar
- Tahmin (kazanan + skor)
- Gerçek sonuç (kazanan + skor)
- Doğru/yanlış
- Fitness puanları
- Popülasyon durumu

Log dosyası APPEND mode'da açılır (üzerine yazmaz).
"""

import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List

class MatchResultsLogger:
    """
    Maç sonuçları için özel logger
    """
    
    def __init__(self, log_file: str = "match_results.log"):
        self.log_file = log_file
        
        # HER ÇALIŞTIRMADA SIFIRDAN BAŞLA! (üzerine yaz)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("MAÇ SONUÇLARI LOG DEFTERİ\n")
            f.write("="*100 + "\n")
            f.write(f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
        
        print(f"📊 Maç sonuçları logger başlatıldı: {log_file}")
    
    def log_match(self,
                  match_idx: int,
                  home_team: str,
                  away_team: str,
                  match_date: str,
                  match_time: str,  # ✅ SAAT eklendi!
                  predicted_winner: str,
                  predicted_score: Optional[Tuple[int, int]],
                  actual_winner: str,
                  actual_score: Optional[Tuple[int, int]],
                  winner_correct: bool,
                  score_fitness: Optional[Dict] = None,
                  confidence: float = 0.0,
                  population_size: int = 0,
                  base_proba: Optional[list] = None,
                  final_proba: Optional[list] = None,
                  lora_thoughts: Optional[List[Dict]] = None,
                  nature_context: Optional[Dict] = None):
        """
        Tek bir maçın sonucunu logla
        """
        if nature_context:
            self.current_context = nature_context
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            # Kaç LoRA doğru bildi hesapla
            if lora_thoughts:
                correct_count = sum(1 for t in lora_thoughts if t['result'] == 'CORRECT')
                total_loras = len(lora_thoughts)
                lora_accuracy = f"{correct_count}/{total_loras} LoRA bildi (%{correct_count/total_loras*100:.0f})"
            else:
                lora_accuracy = "LoRA bilgisi yok"
            
            # Başlık - NET FORMAT (SAAT + ÖZET!)
            result_text = "✅ DOĞRU!" if winner_correct else "❌ YANLIŞ!"
            
            # 🌡️ NATURE & SOCIAL STATS (Context)
            nature_info = ""
            if base_proba is not None and hasattr(self, 'last_nature_state'): 
                # Not: Bu değerleri dışarıdan almak lazım, şimdilik placeholder
                pass
            
            f.write("\n" + "="*100 + "\n")
            f.write(f"MAÇ #{match_idx + 1} - {match_date} {match_time} | {result_text} | {lora_accuracy}\n")
            
            # Ekstra Context Satırı (Varsa)
            if hasattr(self, 'current_context'):
                f.write(f"🌍 DOĞA: {self.current_context.get('temperature', 0.0):.2f}°C (Kaos: {self.current_context.get('chaos', 0.0):.2f}) | 💕 SOSYAL BAĞ: {self.current_context.get('active_bonds', 0)}\n")
            
            f.write("="*100 + "\n")
            f.write(f"🏟️  {home_team} vs {away_team}\n\n")
            
            # TAHMİN - NET FORMAT
            f.write("🔮 TAHMİN:\n")
            f.write("-"*50 + "\n")
            
            # Kim kazanır?
            winner_text = "EV SAHİBİ" if 'home' in predicted_winner.lower() else ("DEPLASMAN" if 'away' in predicted_winner.lower() else "BERABERE")
            f.write(f"   • Kim kazanır? {winner_text}\n")
            f.write(f"   • Güven: {confidence:.0%}\n")
            
            if predicted_score:
                f.write(f"   • Skor tahmini: {predicted_score[0]}-{predicted_score[1]}\n")
            
            f.write("\n")
            
            # GERÇEK SONUÇ - NET FORMAT
            f.write("📥 GERÇEK SONUÇ:\n")
            f.write("-"*50 + "\n")
            
            actual_winner_text = "EV SAHİBİ" if 'home' in actual_winner.lower() else ("DEPLASMAN" if 'away' in actual_winner.lower() else "BERABERE")
            f.write(f"   • Kazanan: {actual_winner_text}\n")
            
            if actual_score:
                f.write(f"   • Maç sonucu: {actual_score[0]}-{actual_score[1]}\n")
            
            f.write("\n")
            
            # SONUÇ DEĞERLENDİRME - NET FORMAT
            f.write("🎯 SONUÇ:\n")
            f.write("-"*50 + "\n")
            
            if winner_correct:
                f.write("   ✅ DOĞRU TAHMİN!\n")
            else:
                f.write("   ❌ YANLIŞ TAHMİN!\n")
            
            # Skor fitness - SADECE TOPLAM
            if score_fitness:
                total_fitness = score_fitness.get('total_fitness', 0)
                if total_fitness > 0:
                    f.write(f"   📈 Toplam Puan: {total_fitness:.0f}\n")
            
            # Popülasyon durumu
            f.write(f"\n🧬 POPÜLASYON: {population_size} LoRA\n")
            
            # 🧠 LoRA DÜŞÜNCELERI (DETAYLI - HER LoRA SKOR TAHMİNİ DE YAPSIN!)
            if lora_thoughts:
                f.write(f"\n💭 LoRA DÜŞÜNCELERI ({len(lora_thoughts)} LoRA):\n")
                f.write("="*100 + "\n")
                
                for i, thought in enumerate(lora_thoughts):
                    # Kazanan tahmini
                    winner_tr = "EV SAHİBİ" if 'home' in thought['prediction'].lower() else ("DEPLASMAN" if 'away' in thought['prediction'].lower() else "BERABERE")
                    
                    # Gerçek kazanan
                    actual_tr = "EV SAHİBİ" if 'home' in actual_winner.lower() else ("DEPLASMAN" if 'away' in actual_winner.lower() else "BERABERE")
                    
                    # Sonuç
                    result_icon = "✅" if thought['result'] == 'CORRECT' else "❌"
                    
                    f.write(f"\n{result_icon} {thought['lora_name']} [{thought['temperament_type']}] | Fitness: {thought.get('old_fitness', 0.5):.3f}\n")
                    f.write(f"   → Kazanan: {winner_tr} ({thought['confidence']*100:.0f}%)\n")
                    
                    # ⚽ HER LoRA KENDİ SKOR TAHMİNİ!
                    lora_score = thought.get('predicted_score', None)
                    
                    if lora_score and lora_score is not None:
                        f.write(f"   → Skor tahmini: {lora_score[0]}-{lora_score[1]}\n")
                    else:
                        f.write(f"   → Skor tahmini: Veri yok (xG eksik)\n")
                    
                    # Gerçek sonuçla karşılaştır
                    if actual_score:
                        f.write(f"   → Gerçek: {actual_score[0]}-{actual_score[1]} ({actual_tr})\n")
                        
                        # Skor analizi (lora_score None olabilir!)
                        if lora_score and lora_score == actual_score:
                            f.write(f"   💬 \"Hem kazananı hem skoru TAM bildiim! 🎯\"\n")
                        elif thought['result'] == 'CORRECT':
                            f.write(f"   💬 \"Kazananı doğru bilsem de skor biraz farklıymış.\"\n")
                        else:
                            if lora_score:  # None değilse
                                score_diff = abs((lora_score[0] - lora_score[1]) - (actual_score[0] - actual_score[1]))
                            f.write(f"   💬 \"Yanıldım. {actual_tr} kazandı {actual_score[0]}-{actual_score[1]} ile.\"\n")
                
                # Özet stats
                f.write("\n" + "─"*100 + "\n")
                correct_count = sum(1 for t in lora_thoughts if t['result'] == 'CORRECT')
                f.write(f"📊 ÖZET: {correct_count}/{len(lora_thoughts)} LoRA doğru bildi (%{correct_count/len(lora_thoughts)*100:.0f})\n")
                
                # En iyi/kötü
                if correct_count > 0:
                    best = max([t for t in lora_thoughts if t['result'] == 'CORRECT'], key=lambda x: x['confidence'])
                    f.write(f"🌟 En iyi: {best['lora_name']} ({best['confidence']*100:.0f}% güvenle doğru!)\n")
                
                wrong = [t for t in lora_thoughts if t['result'] == 'WRONG']
                if wrong:
                    worst = max(wrong, key=lambda x: x['confidence'])
                    f.write(f"⚠️ Aşırı emin yanlış: {worst['lora_name']} ({worst['confidence']*100:.0f}% emindi ama yanlış!)\n")
            
            f.write("\n")
    
    def log_session_start(self, total_matches: int, resume: bool = False):
        """Oturum başlangıcını logla"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "#"*100 + "\n")
            f.write(f"YENİ OTURUM BAŞLADI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if resume:
                f.write("🔄 RESUME MODE: Önceki durumdan devam ediliyor\n")
            f.write(f"📊 Toplam Maç: {total_matches}\n")
            f.write("#"*100 + "\n\n")
    
    def log_session_end(self, total_matches: int, population_size: int):
        """Oturum bitişini logla"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "#"*100 + "\n")
            f.write(f"OTURUM TAMAMLANDI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"✅ İşlenen Maç: {total_matches}\n")
            f.write(f"🧬 Final Popülasyon: {population_size} LoRA\n")
            f.write("#"*100 + "\n\n")

