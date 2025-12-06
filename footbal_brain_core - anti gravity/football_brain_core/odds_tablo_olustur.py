"""
MatchOdds tablosunu oluşturur (eğer yoksa)
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_session
from src.db.schema import Base, MatchOdds
from sqlalchemy import create_engine
import os

# Database path
db_path = os.path.join(Path(__file__).parent, "football_brain.db")
engine = create_engine(f"sqlite:///{db_path}", echo=False)

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 80)
print("📊 MATCH_ODDS TABLOSU OLUŞTURULUYOR")
print("=" * 80)
print()

try:
    # Tabloyu oluştur
    MatchOdds.__table__.create(engine, checkfirst=True)
    print("✅ match_odds tablosu oluşturuldu (veya zaten var)")
    
    # Kontrol et
    session = get_session()
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if 'match_odds' in tables:
            print("✅ match_odds tablosu veritabanında mevcut")
            
            # Kolonları göster
            columns = inspector.get_columns('match_odds')
            print(f"📋 Toplam {len(columns)} kolon:")
            for col in columns[:10]:  # İlk 10'unu göster
                print(f"   - {col['name']} ({col['type']})")
            if len(columns) > 10:
                print(f"   ... ve {len(columns) - 10} kolon daha")
        else:
            print("❌ match_odds tablosu bulunamadı")
    finally:
        session.close()
    
    print()
    print("=" * 80)
    print("✅ TABLO OLUŞTURMA TAMAMLANDI")
    print("=" * 80)
    print()
    print("📝 Şimdi odds_yukle.py scriptini çalıştırabilirsiniz")
    
except Exception as e:
    print(f"❌ Hata: {e}")
    import traceback
    traceback.print_exc()

