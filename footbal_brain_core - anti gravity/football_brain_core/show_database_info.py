"""
Veritabanı dosyası bilgilerini göster
"""
import sys
import os
from pathlib import Path

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.config import Config

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    config = Config()
    
    # Veritabanı yolu
    db_url = config.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
    else:
        db_path = db_url
    
    # Tam yol
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)
    
    db_path = os.path.abspath(db_path)
    
    print("=" * 60)
    print("VERITABANI DOSYASI BILGILERI")
    print("=" * 60)
    print(f"\nDosya yolu:")
    print(f"  {db_path}")
    print()
    
    if os.path.exists(db_path):
        # Dosya bilgileri
        size = os.path.getsize(db_path)
        size_mb = size / (1024 * 1024)
        
        from datetime import datetime
        mod_time = os.path.getmtime(db_path)
        mod_date = datetime.fromtimestamp(mod_time)
        
        print(f"Dosya durumu: [OK] Mevcut")
        print(f"Dosya boyutu: {size_mb:.2f} MB ({size:,} bytes)")
        print(f"Son guncelleme: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Windows'ta dosyayı açmak için
        print("Dosyayi acmak icin:")
        print(f'  explorer /select,"{db_path}"')
        print()
        print("Veya direkt:")
        print(f'  "{db_path}"')
        
    else:
        print("Dosya durumu: [HATA] Dosya bulunamadi!")
        print("Veritabani henuz olusturulmamis olabilir.")
        print("\nOlusturmak icin:")
        print("  python init_db.py")
    
    print("=" * 60)
    
    # Veritabanı içeriği hızlı özet
    if os.path.exists(db_path):
        print("\nVeritabani icerik ozeti:")
        try:
            from football_brain_core.src.db.connection import get_session
            from football_brain_core.src.db.schema import Match, Team, League
            
            session = get_session()
            try:
                match_count = session.query(Match).count()
                team_count = session.query(Team).count()
                league_count = session.query(League).count()
                
                print(f"  Ligler: {league_count}")
                print(f"  Takimlar: {team_count}")
                print(f"  Maclar: {match_count}")
            finally:
                session.close()
        except Exception as e:
            print(f"  [HATA] Veritabani okunamadi: {e}")







