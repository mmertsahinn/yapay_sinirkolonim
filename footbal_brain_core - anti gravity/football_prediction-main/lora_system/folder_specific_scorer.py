"""
🎯 FOLDER SPECIFIC SCORER - Klasör Bazlı Puanlama
==================================================

Her klasör (Einstein, Takım, H2H) için özel puanlama mantığı.
"""

from typing import Dict, Any

class FolderSpecificScorer:
    """
    Klasöre özel puan hesaplayıcı
    """

    def calculate_score_for_folder(self, lora: Any, folder_type: str, match_count: int = 0, collective_memory: Any = None) -> float:
        """
        Belirli bir klasör tipi için LoRA'nın uygunluk puanını hesapla.
        """
        if folder_type == "EINSTEIN":
            # Zeka ve potansiyel odaklı
            # Lazarus potansiyeli yüksek, öğrenme hızı yüksek
            lazarus = getattr(lora, '_lazarus_lambda', 0.5)
            fitness = lora.get_recent_fitness()
            return (lazarus * 0.7) + (fitness * 0.3)

        elif folder_type.startswith("Team_"):
            # Takım uzmanlığı
            team_name = folder_type.replace("Team_", "")
            # Collective memory üzerinden bu takımla ilgili performansını bulmak gerekir
            # Şimdilik basit bir placeholder
            if hasattr(lora, 'specialization') and lora.specialization and team_name in str(lora.specialization):
                return lora.get_recent_fitness() * 1.5
            return lora.get_recent_fitness() * 0.5 # Uzman değilse düşük puan

        return lora.get_recent_fitness()

    def calculate_h2h_score(self, lora: Any, team1: str, team2: str, collective_memory: Any) -> float:
        """
        İki takım arasındaki maçlardaki başarısı
        """
        # Placeholder: Rastgele veya genel fitness
        return lora.get_recent_fitness()

    def get_h2h_details(self, lora: Any, team1: str, team2: str, collective_memory: Any) -> Dict:
        """
        H2H detayları
        """
        return {
            "matches": 0,
            "wins": 0,
            "score": 0.0
        }

# Global instance
folder_specific_scorer = FolderSpecificScorer()
