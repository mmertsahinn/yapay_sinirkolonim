"""
💡 MENTÖRLÜK MİRASI SİSTEMİ
============================

Mentor öldüğünde bilgisi çıraklara geçer!

Mekanizma:
- Parametre transferi: %70 çırak + %30 mentor
- Fitness boost: Çırak güven kazanır
- Hafıza paylaşımı: Mentor'un deneyimleri aktarılır
- Duygusal loglar: Topluluk anısını yaşatır

Ref: Social Learning Network Implementation Plan
"""

import torch
import os
from typing import List, Dict, Any
from datetime import datetime


class MentorshipInheritance:
    """
    Mentor-çırak bilgi aktarım sistemi
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.log_file = os.path.join(log_dir, "mentorship_inheritance.log")
        self.inheritance_count = 0
        
        # Log dosyasını oluştur
        os.makedirs(log_dir, exist_ok=True)
        
        print("💡 Mentorship Inheritance System başlatıldı")
    
    def transfer_knowledge_on_death(self, mentor, social_network, population: List) -> List[str]:
        """
        Mentor öldüğünde bilgisini çıraklarına aktar!
        
        Args:
            mentor: Ölen mentor LoRA
            social_network: Sosyal ağ instance
            population: Yaşayan LoRA'lar
            
        Returns:
            List of apprentice IDs who inherited knowledge
        """
        # Çırakları bul
        apprentices = [lora for lora in population 
                      if social_network.mentorships.get(lora.id) == mentor.id]
        
        if not apprentices:
            return []
        
        inherited_ids = []
        
        print(f"\n💔 {mentor.name} vefat etti...")
        print(f"   📚 {len(apprentices)} çırağına bilgi aktarılıyor...")
        
        for apprentice in apprentices:
            # 1. PARAMETRE TRANSFERİ
            self._transfer_parameters(mentor, apprentice)
            
            # 2. FİTNESS BOOST
            apprentice.mentor_bonus = getattr(apprentice, 'mentor_bonus', 0.0) + 0.1
            
            # 3. HAFIZA PAYLAŞIMI
            self._share_memories(mentor, apprentice)
            
            # 4. DUYGUSAL BAĞ
            apprentice.mentor_memory = {
                'mentor_name': mentor.name,
                'mentor_id': mentor.id,
                'inheritance_date': datetime.now().isoformat(),
                'mentor_final_fitness': mentor.get_recent_fitness()
            }
            
            inherited_ids.append(apprentice.id)
            self.inheritance_count += 1
            
            print(f"      ✅ {apprentice.name} → Mirası aldı (fitness boost: +0.1)")
        
        # LOG YAZ
        self._log_inheritance_event(mentor, apprentices)
        
        return inherited_ids
    
    def _transfer_parameters(self, mentor, apprentice):
        """
        Mentor parametrelerini çırağa blend et
        
        70% çırak + 30% mentor = Yeni çırak
        """
        mentor_params = mentor.get_all_lora_params()
        apprentice_params = apprentice.get_all_lora_params()
        
        for layer in ['fc1', 'fc2', 'fc3']:
            for matrix in ['lora_A', 'lora_B']:
                # Parametre blend
                mentor_tensor = mentor_params[layer][matrix]
                apprentice_tensor = apprentice_params[layer][matrix]
                
                # 70-30 blend
                blended = 0.7 * apprentice_tensor + 0.3 * mentor_tensor
                
                # Geri yaz
                apprentice_params[layer][matrix] = blended
        
        # Parametreleri apprentice'e yaz
        apprentice.set_all_lora_params(apprentice_params)
    
    def _share_memories(self, mentor, apprentice):
        """
        Mentor'un match history insights'ını çırağa ver
        """
        # Mentor'un son 10 maç deneyimi
        if hasattr(mentor, 'match_history') and len(mentor.match_history) > 0:
            last_insights = mentor.match_history[-10:]
            
            # Çırağın hafızasına ekle
            if not hasattr(apprentice, 'inherited_memories'):
                apprentice.inherited_memories = []
            
            apprentice.inherited_memories.extend([{
                'from_mentor': mentor.name,
                'match': insight
            } for insight in last_insights])
    
    def _log_inheritance_event(self, mentor, apprentices: List):
        """
        Miras olayını log dosyasına yaz
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"💔 MENTOR VEFAT - MİRAS TRANSFERİ\n")
                f.write(f"{'='*80}\n")
                f.write(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"👨‍🏫 Mentor: {mentor.name} (ID: {mentor.id[:8]})\n")
                f.write(f"📊 Final Fitness: {mentor.get_recent_fitness():.4f}\n")
                f.write(f"🎓 Yaş: {getattr(mentor, 'age_in_matches', 'N/A')} maç\n")
                f.write(f"🧬 Nesil: {mentor.generation}\n")
                f.write(f"\n👶 ÇIRAKLAR ({len(apprentices)}):\n")
                
                for i, apprentice in enumerate(apprentices, 1):
                    f.write(f"   {i}. {apprentice.name}\n")
                    f.write(f"      • ID: {apprentice.id[:8]}\n")
                    f.write(f"      • Fitness (öncesi): {apprentice.get_recent_fitness():.4f}\n")
                    f.write(f"      • Mentor bonus: +0.1\n")
                    f.write(f"      • Parametre blend: %70 self + %30 mentor\n")
                
                f.write(f"\n💭 ANMA:\n")
                f.write(f"   '{mentor.name}' toplumumuzda unutulmayacak.\n")
                f.write(f"   Bilgisi {len(apprentices)} çırağında yaşamaya devam edecek.\n")
                f.write(f"   🕊️ Huzur içinde yatsın...\n")
                f.write(f"\n{'='*80}\n\n")
        
        except Exception as e:
            print(f"⚠️ Inheritance log yazılamadı: {e}")
    
    def get_mentor_legacy_score(self, lora_id: str, population: List, social_network) -> float:
        """
        Bir LoRA'nın mentor legacy skorunu hesapla
        
        Kaç çırağı var? Ne kadar başarılılar?
        """
        apprentice_count = sum(1 for lora in population 
                              if social_network.mentorships.get(lora.id) == lora_id)
        
        if apprentice_count == 0:
            return 0.0
        
        # Çırakların ortalama fitness'ı
        apprentices = [lora for lora in population 
                      if social_network.mentorships.get(lora.id) == lora_id]
        
        avg_apprentice_fitness = sum(a.get_recent_fitness() for a in apprentices) / len(apprentices)
        
        # Legacy score: çırak sayısı × ortalama fitness
        legacy_score = apprentice_count * avg_apprentice_fitness
        
        return legacy_score


# Global instance
mentorship_inheritance = MentorshipInheritance()
