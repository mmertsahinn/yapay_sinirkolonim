import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None

# BURAYI SABİTLEDİK: Senin verdiğin tam adres
# Klasör adındaki 'footbal' (tek l) yazımına sadık kalarak yolu yazdım
DB_PATH = r"C:\Users\muham\Desktop\footbal_brain_core\football_brain_core\football_brain.db"

def get_engine() -> Engine:
    global _engine
    
    if _engine is None:
        # Önce dosya orada mı diye kontrol edelim (Hata ayıklamak için)
        if os.path.exists(DB_PATH):
            print(f"✅ Veritabanı bulundu: {DB_PATH}")
        else:
            print(f"❌ HATA: Veritabanı dosyası burada YOK: {DB_PATH}")
            print("Lütfen klasör adını (football vs footbal) kontrol et.")

        # URL'yi tam yol ile oluşturuyoruz
        database_url = f"sqlite:///{DB_PATH}"

        _engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
        )
    
    return _engine


def get_session() -> Session:
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    
    return _session_factory()