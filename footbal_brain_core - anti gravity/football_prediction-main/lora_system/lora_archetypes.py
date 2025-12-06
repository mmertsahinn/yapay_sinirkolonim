"""
🎭 LoRA ARKETİPLERİ (Kişilik Şablonları)
========================================

Her arketip = Uç bir kişilik profili

Spawn sırasında arketiplerden seç → Çeşitlilik garantili!
"""

import random


class LoRAArchetypes:
    """
    LoRA kişilik arketipleri
    """
    
    ARCHETYPES = {
        # 1. ZEN MASTER (Sabırlı & Düşük Risk)
        "zen_master": {
            "name": "Zen Master",
            "emoji": "🧘",
            "description": "Aşırı sabırlı, riskten kaçınır, uzun vadeli düşünür",
            "temperament": {
                'patience': 0.95,              # Çok sabırlı!
                'risk_tolerance': 0.10,        # Risk almaz
                'stress_tolerance': 0.90,      # Strese dayanıklı
                'impulsiveness': 0.05,         # Dürtüsel değil
                'hype_sensitivity': 0.20,      # Hype'a aldırmaz
                'independence': 0.70,          # Bağımsız
                'social_intelligence': 0.50,   # Orta sosyal
                'herd_tendency': 0.15,         # Sürü takipçisi değil
                'contrarian_score': 0.30,      # Biraz karşıt
                'ambition': 0.60               # Orta hırs
            }
        },
        
        # 2. MAD WARRIOR (Agresif & Yüksek Risk)
        "mad_warrior": {
            "name": "Mad Warrior",
            "emoji": "⚔️",
            "description": "Aşırı agresif, yüksek risk, hızlı kazanç peşinde",
            "temperament": {
                'patience': 0.10,              # Sabırsız!
                'risk_tolerance': 0.95,        # Çok riskli!
                'stress_tolerance': 0.40,      # Strese zayıf
                'impulsiveness': 0.90,         # Çok dürtüsel!
                'hype_sensitivity': 0.85,      # Hype'a çok duyarlı
                'independence': 0.60,          # Orta bağımsız
                'social_intelligence': 0.30,   # Sosyal zeka düşük
                'herd_tendency': 0.20,         # Sürü takipçisi değil
                'contrarian_score': 0.70,      # Karşıt
                'ambition': 0.95               # Çok hırslı!
            }
        },
        
        # 3. LONE WOLF (Aşırı Bağımsız)
        "lone_wolf": {
            "name": "Lone Wolf",
            "emoji": "🐺",
            "description": "Aşırı bağımsız, anti-sosyal, tek başına çalışır",
            "temperament": {
                'patience': 0.60,              # Orta sabır
                'risk_tolerance': 0.55,        # Orta risk
                'stress_tolerance': 0.70,      # İyi dayanıklı
                'impulsiveness': 0.40,         # Orta dürtüsel
                'hype_sensitivity': 0.15,      # Hype'a aldırmaz
                'independence': 0.98,          # Çok bağımsız!
                'social_intelligence': 0.10,   # Anti-sosyal!
                'herd_tendency': 0.05,         # Sürüye karşı!
                'contrarian_score': 0.80,      # Çok karşıt!
                'ambition': 0.70               # Yüksek hırs
            }
        },
        
        # 4. SOCIAL BUTTERFLY (Aşırı Sosyal)
        "social_butterfly": {
            "name": "Social Butterfly",
            "emoji": "🦋",
            "description": "Aşırı sosyal, sürü takipçisi, bağa önem verir",
            "temperament": {
                'patience': 0.70,              # İyi sabır
                'risk_tolerance': 0.35,        # Düşük risk
                'stress_tolerance': 0.60,      # Orta dayanıklı
                'impulsiveness': 0.45,         # Orta dürtüsel
                'hype_sensitivity': 0.70,      # Hype'a duyarlı
                'independence': 0.15,          # Çok bağımlı!
                'social_intelligence': 0.95,   # Çok sosyal!
                'herd_tendency': 0.90,         # Sürü takipçisi!
                'contrarian_score': 0.10,      # Karşıt değil
                'ambition': 0.50               # Orta hırs
            }
        },
        
        # 5. CONTRARIAN REBEL (Aşırı Karşıt)
        "contrarian_rebel": {
            "name": "Contrarian Rebel",
            "emoji": "🤘",
            "description": "Herkese inat, karşıt düşünür, mainstream'e karşı",
            "temperament": {
                'patience': 0.50,              # Orta sabır
                'risk_tolerance': 0.75,        # Yüksek risk
                'stress_tolerance': 0.55,      # Orta dayanıklı
                'impulsiveness': 0.65,         # Yüksek dürtüsel
                'hype_sensitivity': 0.30,      # Hype'a karşı!
                'independence': 0.85,          # Çok bağımsız
                'social_intelligence': 0.40,   # Düşük sosyal
                'herd_tendency': 0.05,         # Anti-sürü!
                'contrarian_score': 0.98,      # Çok karşıt!
                'ambition': 0.75               # Yüksek hırs
            }
        },
        
        # 6. PERFECTIONIST (Aşırı Titiz)
        "perfectionist": {
            "name": "Perfectionist",
            "emoji": "🎯",
            "description": "Aşırı titiz, düşük risk, yüksek standartlar",
            "temperament": {
                'patience': 0.85,              # Çok sabırlı
                'risk_tolerance': 0.20,        # Çok düşük risk!
                'stress_tolerance': 0.50,      # Orta stres (mükemmellik baskısı)
                'impulsiveness': 0.10,         # Çok düşük dürtü!
                'hype_sensitivity': 0.25,      # Hype'a az duyarlı
                'independence': 0.70,          # Bağımsız
                'social_intelligence': 0.45,   # Orta sosyal
                'herd_tendency': 0.25,         # Düşük sürü
                'contrarian_score': 0.40,      # Orta karşıt
                'ambition': 0.90               # Çok hırslı!
            }
        },
        
        # 7. GAMBLER (Aşırı Kumar)
        "gambler": {
            "name": "Gambler",
            "emoji": "🎲",
            "description": "Her şey bahis, aşırı risk seven, şans oyunları",
            "temperament": {
                'patience': 0.25,              # Sabırsız
                'risk_tolerance': 0.98,        # Çok çok riskli!
                'stress_tolerance': 0.70,      # Stresi sever!
                'impulsiveness': 0.95,         # Çok dürtüsel!
                'hype_sensitivity': 0.90,      # Hype'a çok duyarlı
                'independence': 0.50,          # Orta bağımsız
                'social_intelligence': 0.35,   # Düşük sosyal
                'herd_tendency': 0.30,         # Düşük sürü
                'contrarian_score': 0.60,      # Orta karşıt
                'ambition': 0.85               # Çok hırslı
            }
        },
        
        # 8. ANALYST (Aşırı Analitik)
        "analyst": {
            "name": "Analyst",
            "emoji": "📊",
            "description": "Veri odaklı, soğuk mantık, duygusuz",
            "temperament": {
                'patience': 0.80,              # Çok sabırlı
                'risk_tolerance': 0.40,        # Düşük-orta risk
                'stress_tolerance': 0.75,      # İyi dayanıklı
                'impulsiveness': 0.08,         # Çok düşük dürtü!
                'hype_sensitivity': 0.10,      # Hype'a karşı!
                'independence': 0.85,          # Çok bağımsız
                'social_intelligence': 0.25,   # Düşük sosyal (duygusuz)
                'herd_tendency': 0.15,         # Anti-sürü
                'contrarian_score': 0.50,      # Orta karşıt
                'ambition': 0.70               # Yüksek hırs
            }
        },
        
        # 9. OPTIMIST (Aşırı İyimser)
        "optimist": {
            "name": "Optimist",
            "emoji": "😊",
            "description": "Her şey güzel, pozitif, naif",
            "temperament": {
                'patience': 0.75,              # İyi sabır
                'risk_tolerance': 0.65,        # Orta-yüksek risk (iyimserlik)
                'stress_tolerance': 0.85,      # Çok dayanıklı (pozitif)
                'impulsiveness': 0.55,         # Orta dürtü
                'hype_sensitivity': 0.80,      # Hype'a çok duyarlı
                'independence': 0.40,          # Düşük bağımsız
                'social_intelligence': 0.70,   # İyi sosyal
                'herd_tendency': 0.65,         # Sürü takipçisi
                'contrarian_score': 0.15,      # Karşıt değil
                'ambition': 0.60               # Orta hırs
            }
        },
        
        # 10. PESSIMIST (Aşırı Karamsar)
        "pessimist": {
            "name": "Pessimist",
            "emoji": "😔",
            "description": "Her şey kötü gidecek, negatif, şüpheci",
            "temperament": {
                'patience': 0.45,              # Düşük sabır (sinirli)
                'risk_tolerance': 0.15,        # Çok düşük risk!
                'stress_tolerance': 0.30,      # Düşük dayanıklı
                'impulsiveness': 0.35,         # Düşük-orta dürtü
                'hype_sensitivity': 0.20,      # Hype'a şüpheci
                'independence': 0.60,          # Orta bağımsız
                'social_intelligence': 0.35,   # Düşük sosyal
                'herd_tendency': 0.25,         # Düşük sürü
                'contrarian_score': 0.75,      # Yüksek karşıt
                'ambition': 0.35               # Düşük hırs (ne olacak ki?)
            }
        },
        
        # 11. CHAOS AGENT (Kaos Temsilcisi)
        "chaos_agent": {
            "name": "Chaos Agent",
            "emoji": "🌪️",
            "description": "Tamamen rastgele, tahmin edilemez, kaotik",
            "temperament": {
                'patience': random.uniform(0.0, 1.0),     # Tamamen rastgele!
                'risk_tolerance': random.uniform(0.0, 1.0),
                'stress_tolerance': random.uniform(0.0, 1.0),
                'impulsiveness': random.uniform(0.5, 1.0),  # En az orta dürtü
                'hype_sensitivity': random.uniform(0.0, 1.0),
                'independence': random.uniform(0.0, 1.0),
                'social_intelligence': random.uniform(0.0, 1.0),
                'herd_tendency': random.uniform(0.0, 1.0),
                'contrarian_score': random.uniform(0.0, 1.0),
                'ambition': random.uniform(0.0, 1.0)
            }
        },
        
        # 12. HYPE BEAST (Tren Takipçisi)
        "hype_beast": {
            "name": "Hype Beast",
            "emoji": "🔥",
            "description": "Trendleri takip eder, popüler olana yönelir",
            "temperament": {
                'patience': 0.30,              # Sabırsız
                'risk_tolerance': 0.70,        # Yüksek risk
                'stress_tolerance': 0.45,      # Düşük-orta
                'impulsiveness': 0.85,         # Çok dürtüsel!
                'hype_sensitivity': 0.98,      # Hype'a aşırı duyarlı!
                'independence': 0.20,          # Çok bağımlı
                'social_intelligence': 0.75,   # Yüksek sosyal
                'herd_tendency': 0.95,         # Aşırı sürü takipçisi!
                'contrarian_score': 0.05,      # Anti-karşıt
                'ambition': 0.80               # Yüksek hırs
            }
        }
    }
    
    @classmethod
    def get_random_archetype(cls):
        """Rastgele arketip seç"""
        archetype_key = random.choice(list(cls.ARCHETYPES.keys()))
        return archetype_key, cls.ARCHETYPES[archetype_key]
    
    @classmethod
    def get_diverse_archetypes(cls, count: int):
        """
        Çeşitli arketipleri seç (tekrar yok!)
        
        Args:
            count: Kaç arketip?
        
        Returns:
            List of (key, archetype_dict)
        """
        all_keys = list(cls.ARCHETYPES.keys())
        
        if count <= len(all_keys):
            # Yeterli arketip var, hepsini kullan
            selected_keys = random.sample(all_keys, count)
        else:
            # Yeterli arketip yok, tekrar et
            selected_keys = []
            while len(selected_keys) < count:
                remaining = count - len(selected_keys)
                batch = random.sample(all_keys, min(remaining, len(all_keys)))
                selected_keys.extend(batch)
        
        return [(key, cls.ARCHETYPES[key]) for key in selected_keys]
    
    @classmethod
    def create_balanced_version(cls, archetype_key: str):
        """
        Arketipin dengeli versiyonunu oluştur
        
        Her özellik orta seviyeye çekilir (0.4-0.6 arası)
        
        Args:
            archetype_key: Arketip anahtarı
        
        Returns:
            (key, balanced_archetype_dict)
        """
        if archetype_key not in cls.ARCHETYPES:
            archetype_key = random.choice(list(cls.ARCHETYPES.keys()))
        
        original = cls.ARCHETYPES[archetype_key]
        
        # Dengeli mizaç (hepsi orta seviye)
        balanced_temperament = {
            'patience': random.uniform(0.45, 0.55),
            'risk_tolerance': random.uniform(0.45, 0.55),
            'stress_tolerance': random.uniform(0.45, 0.55),
            'impulsiveness': random.uniform(0.45, 0.55),
            'hype_sensitivity': random.uniform(0.45, 0.55),
            'independence': random.uniform(0.45, 0.55),
            'social_intelligence': random.uniform(0.45, 0.55),
            'herd_tendency': random.uniform(0.45, 0.55),
            'contrarian_score': random.uniform(0.45, 0.55),
            'ambition': random.uniform(0.45, 0.55)
        }
        
        balanced = {
            'name': f"Balanced_{original['name']}",
            'emoji': "⚖️",
            'description': f"Dengeli versiyon: {original['name']} - Orta seviye özellikler",
            'temperament': balanced_temperament
        }
        
        return (f"balanced_{archetype_key}", balanced)
    
    @classmethod
    def get_all_balanced_versions(cls):
        """
        Tüm arketiplerin dengeli versiyonlarını al
        
        Returns:
            List of (key, balanced_archetype_dict)
        """
        balanced_list = []
        for key in cls.ARCHETYPES.keys():
            if key != 'chaos_agent':  # Chaos agent'in dengeli versiyonu olamaz!
                balanced_list.append(cls.create_balanced_version(key))
        
        return balanced_list


# Global instance
lora_archetypes = LoRAArchetypes()

