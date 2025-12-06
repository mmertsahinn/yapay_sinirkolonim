"""
🎭 FİZİK BAZLI ARKETİPLER
==========================

Eski arketipler: "Hırslı Savaşçı", "Sakin Bilge"
Yeni arketipler: Frequency + Amplitude + Phase kombinasyonu!

FREQUENCY (Ne kadar hızlı değişir?):
  Yüksek → Volatil Ruh (Hızlı değişen!)
  Düşük → Sabit Ruh (Yavaş değişen!)

AMPLITUDE (Ne kadar salınır?):
  Yüksek → Dalgalı Karakter (Çok canlı!)
  Düşük → Stabil Karakter (Sakin!)

PHASE (Diğerleriyle senkron mu?):
  0 → Senkron (Sürü ile!)
  π → Asenkron (Sürüden kopuk!)
"""

from typing import Dict
import numpy as np
from math import pi


class PhysicsBasedArchetypes:
    """
    Fizik bazlı arketip sistemi
    """
    
    # ARKETİP TANIMLARI (Frequency + Amplitude kombinasyonu!)
    ARCHETYPES = {
        # ============================================
        # YÜK SEK FREQUENCY (Hızlı değişen!)
        # ============================================
        
        'Volatil Ateş': {
            'description': 'Çok hızlı değişir, çok canlı, öngörülemez!',
            'frequency_range': (0.15, 0.25),
            'amplitude_range': (0.15, 0.25),
            'traits': 'Dürtüsel, Sinirli, Duygusal',
            'analogy': 'Ateş gibi! Hızlı yanar, hızlı söner!',
            'emoji': '🔥'
        },
        
        'Hızlı Gezgin': {
            'description': 'Hızlı değişir ama kontrollü',
            'frequency_range': (0.10, 0.15),
            'amplitude_range': (0.10, 0.15),
            'traits': 'Hırslı, Rekabetçi, Adaptif',
            'analogy': 'Rüzgar gibi! Hızlı ama yönlü!',
            'emoji': '💨'
        },
        
        # ============================================
        # ORTA FREQUENCY (Dengeli!)
        # ============================================
        
        'Dalgalı Okyanus': {
            'description': 'Orta hızda değişir, canlı!',
            'frequency_range': (0.06, 0.10),
            'amplitude_range': (0.12, 0.18),
            'traits': 'Sosyal, Empatik, Dengeli',
            'analogy': 'Okyanus gibi! Dalgalar var ama tahmin edilebilir!',
            'emoji': '🌊'
        },
        
        'Dengeli Merkür': {
            'description': 'Orta hızda, orta salınım',
            'frequency_range': (0.05, 0.08),
            'amplitude_range': (0.08, 0.12),
            'traits': 'Dengeli, Ölçülü, Normal',
            'analogy': 'Normal insan! Ne çok hızlı ne çok yavaş!',
            'emoji': '⚖️'
        },
        
        # ============================================
        # DÜŞÜK FREQUENCY (Yavaş değişen!)
        # ============================================
        
        'Sakin Dağ': {
            'description': 'Yavaş değişir, sakin, istikrarlı',
            'frequency_range': (0.02, 0.05),
            'amplitude_range': (0.05, 0.10),
            'traits': 'Sabırlı, Dayanıklı, Bilge',
            'analogy': 'Dağ gibi! Yavaş erozyon, ama hiç değişmez değil!',
            'emoji': '⛰️'
        },
        
        'Katı Kaya': {
            'description': 'Çok yavaş değişir, neredeyse sabit!',
            'frequency_range': (0.01, 0.03),
            'amplitude_range': (0.03, 0.06),
            'traits': 'Çok Sabırlı, Çok Dayanıklı, Katı',
            'analogy': 'Kaya gibi! Neredeyse hiç değişmez!',
            'emoji': '🗿'
        },
        
        # ============================================
        # ÖZEL KOMBİNASYONLAR
        # ============================================
        
        'Kaotik Yıldırım': {
            'description': 'Yüksek freq + Yüksek amp = TAM KAOS!',
            'frequency_range': (0.20, 0.30),
            'amplitude_range': (0.20, 0.30),
            'traits': 'Dürtüsel, Risk Sever, Öngörülemez',
            'analogy': 'Yıldırım gibi! Hiç belli olmaz!',
            'emoji': '⚡'
        },
        
        'Kutup Yıldızı': {
            'description': 'Düşük freq + Düşük amp = SABİT!',
            'frequency_range': (0.01, 0.02),
            'amplitude_range': (0.02, 0.04),
            'traits': 'Bağımsız, Sabit, Güvenilir',
            'analogy': 'Kutup Yıldızı gibi! Hep aynı yerde!',
            'emoji': '⭐'
        },
        
        'Gelgit Dansçısı': {
            'description': 'Düşük freq + Yüksek amp = YAVAŞ AMA GÜÇLÜ!',
            'frequency_range': (0.02, 0.04),
            'amplitude_range': (0.15, 0.25),
            'traits': 'Duygusal Derinlik, Yavaş ama Güçlü Değişim',
            'analogy': 'Gelgit gibi! Yavaş ama çok etkili!',
            'emoji': '🌙'
        }
    }
    
    @staticmethod
    def determine_archetype_from_physics(lora) -> str:
        """
        LoRA'nın fizik parametrelerinden arketip belirle!
        
        Args:
            lora: LoRA (fluid_temperament dynamics'i olmalı!)
        
        Returns:
            Arketip adı (örn: "Volatil Ateş 🔥")
        """
        from lora_system.fluid_temperament import fluid_temperament
        
        if lora.id not in fluid_temperament.temperament_dynamics:
            return "Dengeli Merkür ⚖️"  # Varsayılan
        
        dynamics = fluid_temperament.temperament_dynamics[lora.id]
        
        # En baskın özellikten frekans ve amplitude al
        # (Örnek: independence)
        if 'independence' in dynamics:
            freq = dynamics['independence']['frequency']
            amp = dynamics['independence']['amplitude']
        else:
            freq = 0.05
            amp = 0.10
        
        # ARKETİP BELİRLE (Frequency + Amplitude kombinasyonu!)
        
        # Kaotik Yıldırım (Yüksek freq + Yüksek amp)
        if freq >= 0.15 and amp >= 0.15:
            return "Kaotik Yıldırım ⚡"
        
        # Volatil Ateş (Yüksek freq)
        elif freq >= 0.12:
            return "Volatil Ateş 🔥"
        
        # Hızlı Gezgin
        elif freq >= 0.08:
            return "Hızlı Gezgin 💨"
        
        # Gelgit Dansçısı (Düşük freq + Yüksek amp)
        elif freq <= 0.04 and amp >= 0.15:
            return "Gelgit Dansçısı 🌙"
        
        # Kutup Yıldızı (Düşük freq + Düşük amp)
        elif freq <= 0.02 and amp <= 0.05:
            return "Kutup Yıldızı ⭐"
        
        # Sakin Dağ
        elif freq <= 0.05:
            return "Sakin Dağ ⛰️"
        
        # Dalgalı Okyanus
        elif amp >= 0.12:
            return "Dalgalı Okyanus 🌊"
        
        # Dengeli
        else:
            return "Dengeli Merkür ⚖️"
    
    @staticmethod
    def get_archetype_description(archetype_name: str) -> Dict:
        """Arketip detaylarını al"""
        for name, details in PhysicsBasedArchetypes.ARCHETYPES.items():
            if name in archetype_name:
                return details
        
        return {
            'description': 'Bilinmiyor',
            'traits': 'Karışık',
            'analogy': 'Normal',
            'emoji': '❓'
        }


# Global instance
physics_archetypes = PhysicsBasedArchetypes()



