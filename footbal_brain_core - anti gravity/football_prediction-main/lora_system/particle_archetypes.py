"""
🌊 PARÇACIK FİZİĞİ ARKETİPLERİ
================================

LoRA'ları parçacık fiziği özelliklerine göre sınıflandır!

ARKETİPLER:
-----------
Sıcaklık (T) bazlı:
  • Yüksek T → Kaotik, Volatil, Keşifçi
  • Düşük T → Sakin, Stabil, Muhafazakar

Sürtünme (ξ) bazlı:
  • Yüksek ξ → Dirençli, Yavaş, Dikkatli
  • Düşük ξ → Hızlı, Atik, Çevik

Onsager-Machlup (S_OM) bazlı:
  • Düşük S_OM → Verimli, Optimize, Zarif
  • Yüksek S_OM → Zorlanan, Verimsiz, Karmaşık

Lazarus (Λ) bazlı:
  • Yüksek Λ → Öğrenebilir, Potansiyel, Değerli
  • Düşük Λ → Dar Uzman, Sınırlı, Tekrarcı

Ghost (U) bazlı:
  • Yüksek U → Ataya Bağlı, Gelenekçi, Muhafazakar
  • Düşük U → Yenilikçi, Özgür, Devrimci
"""

from typing import Dict, Tuple
import math


class ParticleArchetypes:
    """
    Parçacık fiziği bazlı arketip sistemi
    """
    
    def __init__(self):
        # Eşik değerleri
        self.thresholds = {
            'T_high': 0.02,      # Yüksek sıcaklık
            'T_low': 0.005,      # Düşük sıcaklık
            'xi_high': 0.1,      # Yüksek sürtünme
            'xi_low': -0.05,     # Düşük sürtünme (negatif = ivme!)
            'som_low': 1.0,      # Düşük eylem (verimli!)
            'som_high': 3.0,     # Yüksek eylem (verimsiz!)
            'lambda_high': 0.7,  # Yüksek Lazarus
            'lambda_low': 0.3,   # Düşük Lazarus
            'ghost_high': 0.1,   # Yüksek hayalet etkisi
            'ghost_low': 0.01    # Düşük hayalet etkisi
        }
        
        print("🌊 Particle Archetypes başlatıldı")
    
    def determine_archetype(
        self,
        T: float,
        xi: float,
        som: float,
        lazarus_lambda: float,
        ghost_u: float,
        energy: float
    ) -> Dict:
        """
        LoRA'nın parçacık fiziği arketipini belirle!
        
        Args:
            T: Sıcaklık (Langevin)
            xi: Sürtünme (Nosé-Hoover)
            som: Onsager-Machlup eylemi
            lazarus_lambda: Lazarus potansiyeli
            ghost_u: Hayalet potansiyeli
            energy: Life energy
        
        Returns:
            {
                'primary_archetype': Ana arketip,
                'secondary_traits': İkincil özellikler,
                'description': Açıklama,
                'emoji': Emoji
            }
        """
        # 1) SICAKLIK BAZLI (Ana Özellik!)
        if T > self.thresholds['T_high']:
            temp_trait = "Kaotik 🌪️"
            temp_desc = "Yüksek gürültü, çok keşif yapıyor!"
        elif T < self.thresholds['T_low']:
            temp_trait = "Stabil 🗿"
            temp_desc = "Düşük gürültü, kararlı hareket!"
        else:
            temp_trait = "Dengeli 🌊"
            temp_desc = "Orta sıcaklık, dengeli keşif."
        
        # 2) SÜRTÜNME BAZLI
        if xi > self.thresholds['xi_high']:
            friction_trait = "Dirençli 🛑"
            friction_desc = "Yüksek sürtünme, yavaş değişiyor."
        elif xi < self.thresholds['xi_low']:
            friction_trait = "Hızlı ⚡"
            friction_desc = "Düşük sürtünme, hızlı adapte oluyor!"
        else:
            friction_trait = "Orta Hız 🚶"
            friction_desc = "Normal sürtünme."
        
        # 3) VERİMLİLİK (Onsager-Machlup)
        if som < self.thresholds['som_low']:
            efficiency_trait = "Verimli ✨"
            efficiency_desc = "Düşük eylem, zarif yörünge!"
        elif som > self.thresholds['som_high']:
            efficiency_trait = "Zorlanan 💦"
            efficiency_desc = "Yüksek eylem, verimsiz yörünge."
        else:
            efficiency_trait = "Normal Verim 📊"
            efficiency_desc = "Orta verimlilik."
        
        # 4) ÖĞRENEBİLİRLİK (Lazarus)
        if lazarus_lambda > self.thresholds['lambda_high']:
            learning_trait = "Potansiyel Deha 🧠"
            learning_desc = "Yüksek Fisher Info, çok öğrenebilir!"
        elif lazarus_lambda < self.thresholds['lambda_low']:
            learning_trait = "Dar Uzman 🎯"
            learning_desc = "Düşük Fisher Info, sınırlı deneyim."
        else:
            learning_trait = "Orta Öğrenen 📚"
            learning_desc = "Normal öğrenme kapasitesi."
        
        # 5) YENİLİKÇİLİK (Ghost)
        if ghost_u > self.thresholds['ghost_high']:
            innovation_trait = "Gelenekçi 🏛️"
            innovation_desc = "Atalara çok bağlı, muhafazakar."
        elif ghost_u < self.thresholds['ghost_low']:
            innovation_trait = "Devrimci 🔥"
            innovation_desc = "Atalardan uzak, yenilikçi!"
        else:
            innovation_trait = "Dengeli Yenilikçi 🌱"
            innovation_desc = "Ataları dinler ama özgür."
        
        # 6) BİRLEŞİK ARKETİP BELİRLE!
        primary = self._determine_combined_archetype(
            T, xi, som, lazarus_lambda, ghost_u, energy
        )
        
        return {
            'primary_archetype': primary['name'],
            'emoji': primary['emoji'],
            'description': primary['description'],
            'secondary_traits': {
                'temperature': temp_trait,
                'friction': friction_trait,
                'efficiency': efficiency_trait,
                'learning': learning_trait,
                'innovation': innovation_trait
            },
            'trait_descriptions': {
                'temperature': temp_desc,
                'friction': friction_desc,
                'efficiency': efficiency_desc,
                'learning': learning_desc,
                'innovation': innovation_desc
            }
        }
    
    def _determine_combined_archetype(
        self, T: float, xi: float, som: float, 
        lazarus_lambda: float, ghost_u: float, energy: float
    ) -> Dict:
        """
        Birleşik arketip belirle (En baskın özelliklere göre!)
        """
        # ÖZEL ARKETİPLER (Nadir kombinasyonlar!)
        
        # 1) EINSTEIN TİPİ: Yüksek T + Yüksek Λ + Düşük Ghost
        if T > 0.02 and lazarus_lambda > 0.7 and ghost_u < 0.05:
            return {
                'name': "Dâhi Einstein 🌟",
                'emoji': "🌟",
                'description': "Kaotik ama öğrenebilir, yenilikçi deha!"
            }
        
        # 2) NEWTON TİPİ: Düşük T + Düşük S_OM + Düşük ξ
        if T < 0.01 and som < 1.5 and abs(xi) < 0.05:
            return {
                'name': "İstikrarlı Newton 🏛️",
                'emoji': "🏛️",
                'description': "Stabil, verimli, düzenli hareket!"
            }
        
        # 3) DARWIN TİPİ: Orta her şey + Yüksek Λ
        if 0.01 < T < 0.02 and lazarus_lambda > 0.6:
            return {
                'name': "Adaptif Darwin 🧬",
                'emoji': "🧬",
                'description': "Dengeli ama yüksek öğrenme kapasitesi!"
            }
        
        # 4) KAOTIK DEHA: Çok yüksek T + Yüksek Λ + Yüksek S_OM
        if T > 0.03 and lazarus_lambda > 0.6 and som > 3.0:
            return {
                'name': "Kaotik Deha 🌪️",
                'emoji': "🌪️",
                'description': "Aşırı kaotik ama çok deneyim kazanıyor!"
            }
        
        # 5) MUHAFAZAKAR USTA: Düşük T + Yüksek Ghost + Düşük S_OM
        if T < 0.008 and ghost_u > 0.1 and som < 2.0:
            return {
                'name': "Muhafazakar Usta 🗿",
                'emoji': "🗿",
                'description': "Ataları takip eden, verimli, stabil!"
            }
        
        # 6) YENİLİKÇİ KEŞİFÇİ: Yüksek T + Düşük Ghost
        if T > 0.015 and ghost_u < 0.03:
            return {
                'name': "Yenilikçi Keşifçi 🔥",
                'emoji': "🔥",
                'description': "Atalardan kopuk, çok keşif yapıyor!"
            }
        
        # 7) ZORLU SAVAŞÇI: Yüksek S_OM + Yüksek Energy + Yüksek ξ
        if som > 3.5 and energy > 1.5 and xi > 0.1:
            return {
                'name': "Zorlu Savaşçı ⚔️",
                'emoji': "⚔️",
                'description': "Verimsiz ama dayanıklı, mücadeleci!"
            }
        
        # 8) ZARİF USTA: Düşük S_OM + Düşük T + Yüksek Λ
        if som < 1.0 and T < 0.01 and lazarus_lambda > 0.7:
            return {
                'name': "Zarif Usta ✨",
                'emoji': "✨",
                'description': "Verimli, stabil, yüksek öğrenme kapasitesi!"
            }
        
        # 9) DÜŞÜK ENERJİ MÜCADELECI: Düşük Energy + Yüksek S_OM
        if energy < 0.5 and som > 2.5:
            return {
                'name': "Bitkin Savaşçı 💀",
                'emoji': "💀",
                'description': "Enerjisi düşük ama savaşıyor!"
            }
        
        # 10) DEFAULT: Dengeli
        return {
            'name': "Dengeli Parçacık ⚖️",
            'emoji': "⚖️",
            'description': "Ortalama özellikler, dengeli hareket."
        }
    
    def get_archetype_from_lora(self, lora) -> Dict:
        """
        LoRA'dan direkt arketip belirle!
        """
        T = getattr(lora, '_langevin_temp', 0.01)
        xi = getattr(lora, '_nose_hoover_xi', 0.0)
        som = getattr(lora, '_om_action', 0.0)
        lazarus_lambda = getattr(lora, '_lazarus_lambda', 0.5)
        ghost_u = getattr(lora, '_ghost_potential', 0.0)
        energy = getattr(lora, 'life_energy', 1.0)
        
        return self.determine_archetype(T, xi, som, lazarus_lambda, ghost_u, energy)


# Global instance
particle_archetypes = ParticleArchetypes()



