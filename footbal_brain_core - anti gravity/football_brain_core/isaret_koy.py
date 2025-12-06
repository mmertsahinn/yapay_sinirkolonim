"""
Şu ana kadar çekilen maçların sonuna işaret koy
Bu işaretten önceki maçlar kontrol edilecek
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match

def mark_loaded_matches():
    """Şu ana kadar çekilen tüm maçları işaretle"""
    session = get_session()
    
    try:
        # Toplam maç sayısı
        total_matches = session.query(Match).count()
        print(f"Toplam mac sayisi: {total_matches}")
        
        if total_matches == 0:
            print("[UYARI] Veritabaninda mac yok!")
            return
        
        # Son maçı bul (tarih sırasına göre)
        last_match = session.query(Match).order_by(Match.match_date.desc()).first()
        
        if not last_match:
            print("[UYARI] Son mac bulunamadi!")
            return
        
        # İşaret bilgilerini kaydet
        marker_info = {
            "total_matches_marked": total_matches,
            "last_match_id": last_match.id,
            "last_match_date": last_match.match_date.isoformat(),
            "last_match_home_team": last_match.home_team.name if last_match.home_team else "N/A",
            "last_match_away_team": last_match.away_team.name if last_match.away_team else "N/A",
            "marked_at": datetime.now().isoformat(),
            "status": "CHECK_REQUIRED"  # Kontrol gerekiyor
        }
        
        # Dosyaya kaydet
        marker_file = project_root / "loaded_matches_marker.json"
        import json
        with open(marker_file, 'w', encoding='utf-8') as f:
            json.dump(marker_info, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("ISARET KONULDU!")
        print("=" * 80)
        print(f"Toplam isaretlenen mac: {total_matches}")
        print(f"Son mac ID: {last_match.id}")
        print(f"Son mac tarihi: {last_match.match_date.strftime('%Y-%m-%d')}")
        print(f"Son mac: {last_match.home_team.name if last_match.home_team else 'N/A'} vs {last_match.away_team.name if last_match.away_team else 'N/A'}")
        print(f"Isaret dosyasi: {marker_file}")
        print("\n[NOT] Bu isaretten onceki {total_matches} mac kontrol edilecek.")
        print("      Eksik mac var mi kontrol etmek icin: python eksik_mac_kontrol.py")
        print("=" * 80)
        
    except Exception as e:
        print(f"[HATA] {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    mark_loaded_matches()






