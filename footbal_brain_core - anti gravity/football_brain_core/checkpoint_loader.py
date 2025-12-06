"""
Checkpoint sistemi - Son çekilen maçların kaydı
Eksik maç kontrolü için kullanılır
"""
import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional

class CheckpointManager:
    """Maç çekme ilerlemesini takip eder"""
    
    def __init__(self, checkpoint_file: str = "loading_checkpoint.json"):
        self.project_root = Path(__file__).parent
        self.checkpoint_file = self.project_root / checkpoint_file
    
    def save_checkpoint(
        self,
        league_name: str,
        season: int,
        last_match_date: datetime,
        total_matches_loaded: int,
        api_requests_used: int,
        api_requests_remaining: int
    ):
        """Checkpoint kaydet"""
        checkpoint_data = {
            "last_update": datetime.now().isoformat(),
            "league": league_name,
            "season": season,
            "last_match_date": last_match_date.isoformat(),
            "total_matches_loaded": total_matches_loaded,
            "api_requests_used": api_requests_used,
            "api_requests_remaining": api_requests_remaining,
            "status": "in_progress"
        }
        
        # Mevcut checkpoint'leri oku
        all_checkpoints = self.load_all_checkpoints()
        all_checkpoints[f"{league_name}_{season}"] = checkpoint_data
        
        # Kaydet
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(all_checkpoints, f, indent=2, ensure_ascii=False)
    
    def load_all_checkpoints(self) -> Dict:
        """Tüm checkpoint'leri yükle"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def get_checkpoint(self, league_name: str, season: int) -> Optional[Dict]:
        """Belirli bir lig/sezon için checkpoint al"""
        all_checkpoints = self.load_all_checkpoints()
        key = f"{league_name}_{season}"
        return all_checkpoints.get(key)
    
    def mark_completed(self, league_name: str, season: int):
        """Lig/sezon tamamlandı olarak işaretle"""
        all_checkpoints = self.load_all_checkpoints()
        key = f"{league_name}_{season}"
        if key in all_checkpoints:
            all_checkpoints[key]["status"] = "completed"
            all_checkpoints[key]["completed_at"] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(all_checkpoints, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Checkpoint özetini yazdır"""
        all_checkpoints = self.load_all_checkpoints()
        
        if not all_checkpoints:
            print("[BILGI] Henuz checkpoint kaydi yok.")
            return
        
        print("=" * 80)
        print("CHECKPOINT OZETI - Son Cekilen Maclar")
        print("=" * 80)
        
        for key, checkpoint in sorted(all_checkpoints.items()):
            league = checkpoint.get("league", "N/A")
            season = checkpoint.get("season", "N/A")
            last_date = checkpoint.get("last_match_date", "N/A")
            total = checkpoint.get("total_matches_loaded", 0)
            status = checkpoint.get("status", "unknown")
            last_update = checkpoint.get("last_update", "N/A")
            
            try:
                if last_date != "N/A":
                    dt = datetime.fromisoformat(last_date)
                    last_date = dt.strftime("%Y-%m-%d")
            except:
                pass
            
            try:
                if last_update != "N/A":
                    dt = datetime.fromisoformat(last_update)
                    last_update = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
            
            status_icon = "✓" if status == "completed" else "→"
            print(f"\n{status_icon} {league} - Sezon {season}")
            print(f"   Son mac tarihi: {last_date}")
            print(f"   Toplam mac: {total}")
            print(f"   Durum: {status}")
            print(f"   Son guncelleme: {last_update}")
        
        print("\n" + "=" * 80)
        print("NOT: Eksik mac kontrolu icin bu checkpoint'leri kullanabilirsiniz.")
        print("=" * 80)

if __name__ == "__main__":
    manager = CheckpointManager()
    manager.print_summary()






