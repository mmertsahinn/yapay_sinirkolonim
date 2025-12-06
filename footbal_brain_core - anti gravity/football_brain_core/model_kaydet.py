"""Model versiyonunu veritabanına kaydet"""
import sys
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.repositories import ModelVersionRepository

session = get_session()
try:
    # Önceki aktif versiyonları deaktif et
    ModelVersionRepository.deactivate_all(session)
    
    # Yeni model versiyonunu oluştur
    version = "v1.0"
    description = f"PRD Model - Eğitim: 2020-2022, Validation: 2022, Epochs: 50, Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    model_version = ModelVersionRepository.create(session, version, description)
    session.commit()
    
    print("=" * 60)
    print("MODEL VERSIYONU KAYDEDILDI")
    print("=" * 60)
    print(f"Versiyon: {model_version.version}")
    print(f"ID: {model_version.id}")
    print(f"Aktif: {model_version.is_active}")
    print(f"Aciklama: {model_version.description}")
    print("=" * 60)
    print("\n[OK] Model versiyonu basariyla kaydedildi!")
    
except Exception as e:
    session.rollback()
    print(f"[HATA] Model versiyonu kaydedilemedi: {e}")
    import traceback
    traceback.print_exc()
finally:
    session.close()






