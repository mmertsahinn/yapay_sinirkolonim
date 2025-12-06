"""
İşaretten önceki maçları kontrol et - Eksik maç var mı?
"""
import sys
import os
from pathlib import Path
from datetime import datetime, date
import json

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.schema import Match, League
from football_brain_core.src.db.repositories import MatchRepository

def check_missing_matches():
    """İşaretten önceki maçları kontrol et"""
    
    # İşaret dosyasını oku
    marker_file = project_root / "loaded_matches_marker.json"
    if not marker_file.exists():
        print("[HATA] Isaret dosyasi bulunamadi!")
        print("Once isaret koymak icin: python isaret_koy.py")
        return
    
    with open(marker_file, 'r', encoding='utf-8') as f:
        marker = json.load(f)
    
    print("=" * 80)
    print("EKSIK MAC KONTROLU")
    print("=" * 80)
    print(f"Isaret bilgileri:")
    print(f"  Toplam isaretlenen mac: {marker['total_matches_marked']}")
    print(f"  Son mac tarihi: {marker['last_match_date']}")
    print(f"  Son mac: {marker['last_match_home_team']} vs {marker['last_match_away_team']}")
    print(f"  Isaret zamani: {marker['marked_at']}")
    print()
    
    session = get_session()
    
    try:
        # İşaret tarihine kadar olan maçları al
        marker_date = datetime.fromisoformat(marker['last_match_date'])
        
        # Tüm ligleri al
        leagues = session.query(League).all()
        
        print("Lig bazinda kontrol ediliyor...\n")
        
        total_issues = 0
        
        for league in leagues:
            # Bu ligdeki maçları al (işaret tarihine kadar)
            matches = session.query(Match).filter(
                Match.league_id == league.id,
                Match.match_date <= marker_date
            ).order_by(Match.match_date).all()
            
            if not matches:
                continue
            
            # Tarih aralığını bul
            first_match = matches[0]
            last_match = matches[-1]
            
            # Tarih sırasında eksik var mı kontrol et
            match_dates = sorted([m.match_date.date() for m in matches])
            
            # Eksik tarihleri bul (basit kontrol: aynı tarihte birden fazla maç olabilir, bu normal)
            # Daha detaylı kontrol: her gün için beklenen maç sayısı vs.
            
            print(f"{league.name}:")
            print(f"  Toplam mac: {len(matches)}")
            print(f"  Tarih araligi: {first_match.match_date.date()} - {last_match.match_date.date()}")
            
            # Aynı tarihte tekrar eden maçlar var mı? (duplicate kontrolü)
            match_ids = [m.match_id for m in matches if m.match_id]
            unique_ids = set(match_ids)
            if len(match_ids) != len(unique_ids):
                duplicates = len(match_ids) - len(unique_ids)
                print(f"  [UYARI] {duplicates} tekrar eden mac ID bulundu!")
                total_issues += duplicates
            
            # NULL/boş değerler var mı?
            null_scores = sum(1 for m in matches if m.home_score is None or m.away_score is None)
            if null_scores > 0:
                print(f"  [BILGI] {null_scores} macin skoru henuz yok (normal olabilir)")
            
            print()
        
        # Genel özet
        all_matches_until_marker = session.query(Match).filter(
            Match.match_date <= marker_date
        ).count()
        
        print("=" * 80)
        print("OZET")
        print("=" * 80)
        print(f"Isaret tarihine kadar toplam mac: {all_matches_until_marker}")
        print(f"Isaret dosyasindaki sayi: {marker['total_matches_marked']}")
        
        if all_matches_until_marker == marker['total_matches_marked']:
            print("\n[OK] Mac sayilari eslesiyor!")
        else:
            diff = abs(all_matches_until_marker - marker['total_matches_marked'])
            print(f"\n[UYARI] Mac sayilari farkli! Fark: {diff}")
        
        if total_issues == 0:
            print("\n[OK] Tekrar eden mac bulunamadi. Veriler temiz gorunuyor.")
        else:
            print(f"\n[UYARI] {total_issues} sorun bulundu!")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"[HATA] {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    check_missing_matches()






