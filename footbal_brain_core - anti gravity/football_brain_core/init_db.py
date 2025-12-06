"""
Veritabanını initialize et - Kolay başlatma scripti
"""
import sys
from pathlib import Path

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.db.connection import get_engine
from football_brain_core.src.db.schema import Base

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("Veritabani olusturuluyor...")
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("[OK] Veritabani hazir!")

