"""
🧟 LAZARUS POTENTIAL (Diriltme Potansiyeli!)
=============================================

"EN YÜKSEK ÖĞRENME KAPASİTESİ" OLANLAR DİRİLİR!

Fisher Information Matrix (FIM) ile hesaplanan "Lazarus Λ":

Λ(lora) = det(F)^(1/k) × exp(-β × Entropy)

Nerede:
  • F: Fisher Information Matrix (Parametre hassasiyeti!)
  • k: Parametre sayısı
  • Entropy: Sistemin düzensizliği
  • β: Entropi ceza katsayısı

YÜksek Λ → "Çok öğrenmiş ama kötü zamanda öldü!" → DİRİLT!
Düşük Λ → "Az deneyim, dar uzman!" → DİRİLTME!
"""

import torch
import math
from typing import Dict, List, Tuple


class LazarusPotential:
    """
    Fisher Information bazlı diriltme potansiyeli
    """
    
    def __init__(self, beta: float = 0.5):
        """
        Args:
            beta: Entropi ceza katsayısı (0.5 = orta)
        """
        self.beta = beta
        print(f"🧟 Lazarus Potential başlatıldı (β={beta})")
    
    def calculate_lazarus_lambda(
        self,
        lora,
        fisher_info_matrix: torch.Tensor = None
    ) -> Dict:
        """
        Lazarus Λ hesapla!
        
        Args:
            lora: LoRA instance
            fisher_info_matrix: Fisher Info (None ise K-FAC ile hesapla!)
        
        Returns:
            {
                'lambda': Lazarus Λ değeri,
                'fisher_det': Fisher determinantı,
                'entropy': Entropi,
                'learning_capacity': Öğrenme kapasitesi (Fisher!)
            }
        """
        # 1) FISHER INFO HESAPLA (K-FAC ile!)
        if fisher_info_matrix is None:
            from lora_system.kfac_fisher import kfac_fisher
            fisher_data = kfac_fisher.compute_fisher_kfac(lora)
            
            # Log-Determinant kullan (Daha stabil!)
            if 'fisher_logdet' in fisher_data:
                log_det = fisher_data['fisher_logdet']
                det_F = fisher_data.get('fisher_det', 0.0)
                
                # Geometrik ortalama yerine LOG-FISHER SCORE kullan!
                # det(F)^(1/k) yerine log(det(F)) / k
                rank = 16  # LoRA rank
                k = rank * 3  # 3 layer (fc1, fc2, fc3)
                
                # Log-space'de işlem yap (Fisher Score ≈ 40-60 arası çıkar)
                fisher_score = log_det / k
                
                # 🔍 DEBUG: Fisher hesaplama detayları
                if hasattr(lora, 'birth_match'):
                    match_age = getattr(lora, '_current_match', 0) - lora.birth_match
                    if match_age % 50 == 0 or match_age < 5:
                        print(f"      🔍 Fisher Debug ({lora.name[:20]}):")
                        print(f"         • Log-Det: {log_det:.2f}")
                        print(f"         • Fisher Score (Log/k): {fisher_score:.3f}")
                        
                        # Yeni Eşikler (Log-Scale)
                        if fisher_score < 40.0:
                            print(f"         💬 Yorum: 'Düşük Fisher - Az deneyim'")
                        elif fisher_score < 48.0:
                            print(f"         💬 Yorum: 'Orta Fisher - Standart öğrenme'")
                        elif fisher_score < 55.0:
                            print(f"         💬 Yorum: 'Yüksek Fisher - Çok iyi öğrenmiş!'")
                        else:
                            print(f"         🌟 Yorum: 'EFSANE FISHER - Muazzam bilgi!'")
            else:
                # Fallback (Eski yöntem - çok nadir)
                det_F = fisher_data.get('fisher_det', 1e-10)
                if det_F <= 0: det_F = 1e-10
                rank = 16
                k = rank * 3
                fisher_score = math.log(det_F) / k
        else:
            # Eğer Fisher matrisi verilmişse
            try:
                det_F = torch.det(fisher_info_matrix).item()
                if det_F <= 0: det_F = 1e-10
                k = fisher_info_matrix.shape[0]
                fisher_score = math.log(det_F) / k
            except:
                fisher_score = 40.0  # Hesaplanamazsa default (orta)
        
        # 3) ENTROPİ HESAPLA
        entropy = self._calculate_entropy(lora)
        
        # 🔍 DEBUG: Entropy yorumu
        if hasattr(lora, 'birth_match'):
            match_age = getattr(lora, '_current_match', 0) - lora.birth_match
            if match_age % 50 == 0 or match_age < 5:
                print(f"         • Entropy: {entropy:.4f}")
                if entropy < 0.02:
                    print(f"         ⚠️ Uyarı: 'Çok düşük entropy - Parametreler tekdüze!'")
                elif entropy < 0.05:
                    print(f"         💬 Yorum: 'Düşük entropy - Genetik çeşitlilik az'")
                elif entropy < 0.15:
                    print(f"         💬 Yorum: 'Orta entropy - Normal çeşitlilik'")
                else:
                    print(f"         ✅ Yorum: 'Yüksek entropy - İyi çeşitlilik!'")
        
        # 4) LAZARUS Λ (Yeni Formül)
        # Fisher Score 40-60 arası değişir.
        # Bunu 0-1 arasına normalize etmeye çalışalım ama ucu açık kalsın.
        # Referans: 50.0 = İyi
        
        # Normalize Score: (Fisher - 30) / 20  => 30->0.0, 50->1.0, 60->1.5
        normalized_fisher = max(0.0, (fisher_score - 30.0) / 20.0)
        
        # Lambda = Normalized_Fisher * Entropy_Penalty
        lambda_value = normalized_fisher * math.exp(-self.beta * entropy)
        
        # 🔍 DEBUG: Final Lazarus Lambda yorumu
        if hasattr(lora, 'birth_match'):
            match_age = getattr(lora, '_current_match', 0) - lora.birth_match
            if match_age % 50 == 0 or match_age < 5:
                print(f"         • Lazarus Λ: {lambda_value:.3f}")
                if lambda_value < 0.5:
                    print(f"         📉 'DÜŞÜK - Diriltme önceliği düşük'")
                elif lambda_value < 0.8:
                    print(f"         📊 'ORTA - Standart aday'")
                elif lambda_value < 1.1:
                    print(f"         📈 'İYİ - Güçlü aday'")
                else:
                    print(f"         🌟 'YÜKSEK - Efsane! Mutlaka dirilt!'")
        
        # Fisher determinant değerini belirle
        # ÖNEMLİ: det_F neden 0.0 olabilir?
        # 1. K-FAC kullanıldığında: det_F hesaplanmaz, sadece logdet kullanılır (normal!)
        #    → fisher_data.get('fisher_det', 0.0) = 0.0 (K-FAC logdet kullanır, det_F gerekmez)
        # 2. Fallback'te: det_F = 1e-10 (default değer)
        # 3. Fisher matrisi verilmişse: det_F = torch.det(...) hesaplanır
        # 4. Hesaplanamazsa: det_F tanımlı değil, 0.0 döner
        # 
        # SONUÇ: fisher_det = 0.0 NORMALDİR! K-FAC kullanıldığında logdet kullanılır, det_F gerekmez.
        # Asıl önemli olan fisher_score (log-scale) değeridir!
        if 'det_F' in locals():
            fisher_det_value = det_F
        else:
            # Fisher determinant hesaplanamadı (K-FAC kullanıldığında normal)
            # K-FAC logdet kullanır, det_F hesaplanmaz (0.0 = K-FAC kullanıldı, NORMAL!)
            fisher_det_value = 0.0
        
        return {
            'lambda': lambda_value,
            'fisher_det': fisher_det_value,  # Fisher determinant (0.0 = K-FAC kullanıldı [NORMAL!], 1e-10 = fallback default)
            'fisher_term': fisher_score,  # Log-Scale Fisher Score (40-60 arası normal, 50 = iyi) - ASIL ÖNEMLİ OLAN BU!
            'entropy': entropy,
            'learning_capacity': fisher_score,
            'formula': f"Λ = ({fisher_score:.1f}-30)/20 × exp(-{self.beta}×{entropy:.2f}) = {lambda_value:.3f}"
        }
    
    def check_population_diversity(self, population: List, match_idx: int):
        """
        Popülasyon çeşitliliğini kontrol et ve UYAR!
        
        Her 50 maçta çağrılmalı
        """
        if match_idx % 50 != 0 or match_idx == 0:
            return
        
        # Tüm LoRA'ların Lazarus Lambda değerlerini topla
        lambdas = [getattr(lora, '_lazarus_lambda', 0.5) for lora in population]
        
        # İstatistikler
        import numpy as np
        mean_lambda = np.mean(lambdas)
        std_lambda = np.std(lambdas)
        unique_values = len(set([round(l, 2) for l in lambdas]))
        
        print(f"\n🧬 GENETİK ÇEŞİTLİLİK RAPORU (Maç #{match_idx}):")
        print(f"   {'═'*60}")
        print(f"   • Popülasyon: {len(population)} LoRA")
        print(f"   • Ortalama Lazarus Λ: {mean_lambda:.3f}")
        print(f"   • Standart Sapma: {std_lambda:.3f}")
        print(f"   • Benzersiz Değer: {unique_values}/{len(population)}")
        
        # YORUMLAR VE UYARILAR!
        if std_lambda < 0.05:
            print(f"\n   🚨 KRİTİK UYARI: GENETİK ÇEŞİTLİLİK ÇOK DÜŞÜK!")
            print(f"      💬 Yorum: 'Tüm LoRA'lar birbirine çok benziyor!'")
            print(f"      💬 Sebep: Koloni mantığı - Kimse ölmüyor, baskı yok")
            print(f"      💡 İleride düşünülecek:")
            print(f"         • Mutasyon oranını artır")
            print(f"         • Diversity spawn ekle")
            print(f"         • Kara Veba'yı bekle (doğal eleme)")
        
        elif std_lambda < 0.10:
            print(f"\n   ⚠️  UYARI: Genetik çeşitlilik az")
            print(f"      💬 Yorum: 'LoRA'lar benzeşiyor'")
            print(f"      💡 İleride düşünülecek: Çeşitlilik artırma")
        
        elif std_lambda < 0.20:
            print(f"\n   ✅ Genetik çeşitlilik normal")
            print(f"      💬 Yorum: 'Sağlıklı popülasyon çeşitliliği'")
        
        else:
            print(f"\n   🌟 Genetik çeşitlilik YÜKSEK!")
            print(f"      💬 Yorum: 'Çok çeşitli popülasyon - Mükemmel!'")
        
        print(f"   {'═'*60}\n")
    
    def _calculate_entropy(self, lora) -> float:
        """
        LoRA'nın entropisini hesapla!
        
        Entropy = -Σ p_i log(p_i)
        
        p_i = fitness dağılımı (başarı çeşitliliği!)
        """
        # Fitness geçmişi
        if not hasattr(lora, 'fitness_history') or len(lora.fitness_history) < 5:
            return 0.5  # Default
        
        fitness_hist = lora.fitness_history[-100:]  # Son 100 maç
        
        # Histogram (10 bin)
        hist, _ = torch.histogram(
            torch.tensor(fitness_hist, dtype=torch.float32),
            bins=10,
            range=(0.0, 1.0)
        )
        
        # Normalize et (olasılık!)
        p = hist.float() / (hist.sum() + 1e-8)
        
        # Shannon entropy
        entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
        
        # Normalize (0-1 arası)
        # Max entropy = log(10) ≈ 2.30
        entropy_normalized = entropy / 2.30
        
        return entropy_normalized
    
    def rank_for_resurrection(
        self,
        dead_loras: List,
        top_n: int = 10
    ) -> List[Tuple]:
        """
        Ölü LoRA'ları Lazarus Λ'ya göre sırala!
        
        Args:
            dead_loras: Ölü LoRA listesi
            top_n: İlk kaç tane?
        
        Returns:
            [(lora, lambda_data), ...] (Sıralı!)
        """
        results = []
        
        for lora in dead_loras:
            try:
                lambda_data = self.calculate_lazarus_lambda(lora)
                results.append((lora, lambda_data))
            except Exception as e:
                # Hesaplanamazsa atla
                continue
        
        # Λ'ya göre sırala (Büyükten küçüğe!)
        results.sort(key=lambda x: x[1]['lambda'], reverse=True)
        
        return results[:top_n]
    
    def print_resurrection_ranking(self, ranked_loras: List[Tuple]):
        """
        Diriltme sıralamasını yazdır!
        """
        print("\n" + "="*80)
        print("🧟 LAZARUS POTENTIAL SIRALAMA (Diriltme Önceliği!)")
        print("="*80)
        print(f"{'Rank':<6} {'LoRA':<25} {'Λ':<8} {'Fisher':<10} {'Entropy':<10} {'Kapasite':<10}")
        print("-"*80)
        
        for idx, (lora, data) in enumerate(ranked_loras, start=1):
            print(f"#{idx:<5} {lora.name[:24]:<25} {data['lambda']:<8.3f} "
                  f"{data['fisher_term']:<10.3f} {data['entropy']:<10.2f} "
                  f"{'Yüksek!' if data['learning_capacity'] > 1.0 else 'Orta':<10}")
        
        print("="*80)
        print(f"💡 YORUM:")
        print(f"   • Yüksek Λ = Çok deneyim + Düşük entropi = DİRİLT!")
        print(f"   • Fisher > 1.0 = Geniş parametre uzayı keşfetti!")
        print(f"   • Entropi düşük = Organize öğrenmiş!")
        print("="*80 + "\n")


# Global instance
lazarus_potential = LazarusPotential(beta=0.5)


