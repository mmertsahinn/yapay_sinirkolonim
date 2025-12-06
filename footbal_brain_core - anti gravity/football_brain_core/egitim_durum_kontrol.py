"""Eğitim durumunu kontrol et"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("EĞİTİM DURUMU KONTROLÜ")
print("=" * 60)
print(f"Kontrol zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 1. Python süreçleri
print("1. ÇALIŞAN PYTHON SÜREÇLERİ:")
print("-" * 60)
try:
    import subprocess
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                          capture_output=True, text=True, encoding='utf-8')
    lines = result.stdout.strip().split('\n')
    python_processes = [l for l in lines if 'python.exe' in l.lower()]
    print(f"   Toplam Python süreci: {len(python_processes)}")
    if python_processes:
        for proc in python_processes[:3]:  # İlk 3'ü göster
            parts = proc.split(',')
            if len(parts) >= 2:
                print(f"   - PID: {parts[1].strip('\"')}")
except:
    print("   Kontrol edilemedi")

# 2. Model dosyası
print("\n2. MODEL DOSYASI:")
print("-" * 60)
model_path = Path(__file__).parent / "model_prd_v1.0.pth"
if model_path.exists():
    file_size = model_path.stat().st_size
    mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
    print(f"   [OK] Model dosyasi var")
    print(f"   [INFO] Boyut: {file_size / (1024*1024):.2f} MB")
    print(f"   [INFO] Son guncelleme: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ne kadar süre önce güncellendi
    time_diff = datetime.now() - mod_time
    if time_diff.total_seconds() < 60:
        print(f"   [INFO] {int(time_diff.total_seconds())} saniye once guncellendi")
    elif time_diff.total_seconds() < 3600:
        print(f"   [INFO] {int(time_diff.total_seconds() / 60)} dakika once guncellendi")
    else:
        print(f"   [INFO] {int(time_diff.total_seconds() / 3600)} saat once guncellendi")
else:
    print("   [HATA] Model dosyasi henuz olusturulmamis")

# 3. Veritabanı durumu
print("\n3. VERİTABANI DURUMU:")
print("-" * 60)
try:
    from src.db.connection import get_session
    from src.db.schema import Match
    from sqlalchemy import extract, func
    
    session = get_session()
    try:
        # Eğitim için maç sayısı (2022 ve öncesi)
        train_count = session.query(Match).filter(
            extract('year', Match.match_date) <= 2022
        ).count()
        
        # Validation için maç sayısı (2022)
        val_count = session.query(Match).filter(
            extract('year', Match.match_date) == 2022
        ).count()
        
        # Toplam maç
        total_count = session.query(Match).count()
        
        print(f"   [INFO] Egitim maclari (<=2022): {train_count:,}")
        print(f"   [INFO] Validation maclari (2022): {val_count:,}")
        print(f"   [INFO] Toplam mac: {total_count:,}")
        
    finally:
        session.close()
except Exception as e:
    print(f"   [HATA] Hata: {e}")

print("\n" + "=" * 60)
print("[BILGI] Egitim devam ediyorsa model dosyasi duzenli olarak guncellenecek")
print("[BILGI] Egitim tamamlandiginda 'Model egitimi tamamlandi!' mesaji gorunecek")
print("=" * 60)

