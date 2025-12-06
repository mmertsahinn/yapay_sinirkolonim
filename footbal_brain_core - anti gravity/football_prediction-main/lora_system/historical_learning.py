"""
📚 TARİHSEL ÖĞRENME SİSTEMİ
============================

LoRA'lar başlangıçta TÜM GEÇMİŞİ okur ve kendi mizacına göre yorumlar!

GEÇMİŞ KAYNAKLAR:
1. Ortak hafıza (collective_memory) - 500+ maç
2. Ölü LoRA'ların deneyimleri (all_loras_ever)
3. Diğer LoRA'ların walletları

Her LoRA bu verileri okur ve:
- Kendi mizacına göre çıkarım yapar
- Atalardan öğrenir
- Başkalarının hatalarından ders çıkarır
"""

from typing import Dict, List
import numpy as np


class HistoricalLearningSystem:
    """
    Tarihsel öğrenme sistemi
    """
    
    @staticmethod
    def lora_reads_collective_history(lora, collective_memory: Dict, all_loras_ever: Dict) -> Dict:
        """
        LoRA BAŞLANGIÇTA TÜM GEÇMİŞİ OKUR!
        
        Args:
            lora: LoRA instance
            collective_memory: Ortak hafıza
            all_loras_ever: Tüm zamanlar LoRA kayıtları
        
        Returns:
            {
                'learned_insights': [...],
                'ancestor_wisdom': [...],
                'personal_conclusion': str
            }
        """
        temp = lora.temperament
        
        print(f"\n📚 {lora.name} GEÇMİŞİ OKUYOR...")
        
        insights = []
        ancestor_wisdom = []
        
        # ============================================
        # 1. ORTAK HAFIZAYI OKU (500+ maç!)
        # ============================================
        
        total_matches = len(collective_memory)
        
        if total_matches > 0:
            # Pattern başarı oranları
            pattern_success = {}
            
            for match_key, match_data in collective_memory.items():
                # Bu maçta hangi pattern'ler vardı?
                # Hangi LoRA'lar doğru bildi?
                
                lora_insights_data = match_data.get('lora_insights', {})
                
                for lora_id, insight in lora_insights_data.items():
                    # Bu LoRA'nın öğrenmesi
                    learning = insight.get('learning', '')
                    correct = insight.get('correct', False)
                    
                    if learning and correct:
                        # Başarılı bir öğrenme!
                        insights.append({
                            'from_lora': insight.get('name', 'Unknown'),
                            'learning': learning,
                            'match': match_data.get('match_idx', 0)
                        })
            
            print(f"   📖 {total_matches} maçın geçmişini okudu")
            print(f"   💡 {len(insights)} başarılı öğrenme buldu")
        
        # ============================================
        # 2. ATALARIN BİLGELİĞİ (Ölü LoRA'lar!)
        # ============================================
        
        if all_loras_ever:
            # En başarılı ölüleri bul
            dead_legends = []
            
            for lora_id, lora_data in all_loras_ever.items():
                if not lora_data.get('alive', True):  # Ölü
                    final_fitness = lora_data.get('final_fitness', 0.0)
                    
                    if final_fitness > 0.65:  # Başarılıydı!
                        dead_legends.append({
                            'lora_id': lora_id,
                            'name': lora_data.get('lora', {}).name if 'lora' in lora_data else 'Unknown',
                            'fitness': final_fitness,
                            'specialization': lora_data.get('lora', {}).specialization if 'lora' in lora_data else None
                        })
            
            # En iyi 10 atayı al
            dead_legends.sort(key=lambda x: x['fitness'], reverse=True)
            top_ancestors = dead_legends[:10]
            
            for ancestor in top_ancestors:
                ancestor_wisdom.append({
                    'name': ancestor['name'],
                    'fitness': ancestor['fitness'],
                    'specialization': ancestor['specialization']
                })
            
            print(f"   🏛️ {len(top_ancestors)} atanın bilgeliğini okudu")
        
        # ============================================
        # 3. MİZAÇ BAZLI YORUM!
        # ============================================
        
        personal_conclusion = HistoricalLearningSystem._interpret_history(
            lora, insights, ancestor_wisdom, total_matches
        )
        
        print(f"   💭 Kişisel Sonuç: \"{personal_conclusion}\"")
        
        return {
            'learned_insights': insights,
            'ancestor_wisdom': ancestor_wisdom,
            'personal_conclusion': personal_conclusion,
            'total_history_size': total_matches
        }
    
    @staticmethod
    def _interpret_history(lora, insights: List, ancestors: List, total_matches: int) -> str:
        """
        Geçmişi mizaç bazlı yorumla
        """
        temp = lora.temperament
        
        independence = temp.get('independence', 0.5)
        social_intelligence = temp.get('social_intelligence', 0.5)
        ambition = temp.get('ambition', 0.5)
        contrarian = temp.get('contrarian_score', 0.5)
        
        # BAĞIMSIZ
        if independence > 0.8:
            return f"{total_matches} maç geçmişi var. İlginç ama kendi yolumu bulacağım."
        
        # SOSYAL ZEKİ
        elif social_intelligence > 0.7:
            return f"{len(insights)} başarılı strateji, {len(ancestors)} ata bilgeliği. Hepsinden öğreneceğim!"
        
        # KARŞIT
        elif contrarian > 0.7:
            return f"Herkes böyle yapmış ama ben farklı düşünüyorum. Kendi yolumu deneyeceğim."
        
        # HIRSLI
        elif ambition > 0.7:
            return f"Atalarımın başarısını geçeceğim! {len(ancestors)} atadan daha iyi olacağım!"
        
        # DENGELI
        else:
            return f"{total_matches} maç deneyimi. Dengeli bir yaklaşım benimseyeceğim."


# Global instance
historical_learning = HistoricalLearningSystem()



