"""
Eğitim İlerlemesini İzleme Scripti
train_enhance_v2.py'nin çalışmasını izler ve ilerlemeyi gösterir
"""
import os
import time
import sys
from pathlib import Path
from datetime import datetime

def format_size(size_bytes):
    """Dosya boyutunu okunabilir formata çevir"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_file_info(filepath):
    """Dosya bilgilerini al"""
    if os.path.exists(filepath):
        stat = os.stat(filepath)
        size = format_size(stat.st_size)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        return size, mtime
    return None, None

def monitor_training():
    """Eğitim sürecini izle"""
    script_dir = Path(__file__).parent
    
    # İzlenecek dosyalar
    files_to_watch = {
        'Model': script_dir / 'football_prediction_ensemble.joblib',
        'Confusion Matrix': script_dir / 'confusion_matrix.png',
        'Feature Importance': script_dir / 'feature_importance.png',
        'Learning Curve': script_dir / 'learning_curve.png',
        'Home Goals Model': script_dir / 'home_goals_model.joblib',
        'Away Goals Model': script_dir / 'away_goals_model.joblib'
    }
    
    print("="*70)
    print("EGITIM IZLEME EKRANI")
    print("="*70)
    print(f"Baslangic zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nIzlenecek dosyalar:")
    for name, path in files_to_watch.items():
        status = "[VAR]" if path.exists() else "[YOK]"
        print(f"  {status} {name}")
    print("\n" + "="*70)
    print("Ilerleme takibi basladi... (Ctrl+C ile cikis)\n")
    
    last_sizes = {}
    check_count = 0
    start_time = time.time()
    
    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            elapsed = int(time.time() - start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            
            # Ekranı temizle
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("="*70)
            print(f"EGITIM IZLEME - {current_time} (Calisma suresi: {hours:02d}:{minutes:02d}:{seconds:02d})")
            print("="*70)
            
            any_update = False
            for name, filepath in files_to_watch.items():
                size, mtime = get_file_info(filepath)
                
                if size:
                    status = "[OK]"
                    # Boyut değişikliği kontrolü
                    current_size = os.path.getsize(filepath) if filepath.exists() else 0
                    if name in last_sizes and current_size != last_sizes[name]:
                        status = "[GUNCELLENIYOR]"
                        any_update = True
                    last_sizes[name] = current_size
                    
                    print(f"{status} {name:30s} | Boyut: {size:>10s} | Son guncelleme: {mtime}")
                else:
                    print(f"[BEKLENIYOR] {name:30s} | Dosya henuz olusturulmadi")
            
            # CSV kontrolü
            csv_path = script_dir / 'football_match_data.csv'
            if csv_path.exists():
                csv_size, csv_mtime = get_file_info(csv_path)
                print(f"\n[CSV] football_match_data.csv | Boyut: {csv_size:>10s} | Son guncelleme: {csv_mtime}")
            
            print("\n" + "="*70)
            print(f"Kontrol sayisi: {check_count} | Son kontrol: {current_time}")
            
            if any_update:
                print(">>> Dosya guncellemesi tespit edildi! <<<")
            
            print("\nNot: Her 10 saniyede bir kontrol ediliyor...")
            print("Ctrl+C ile cikis yapabilirsiniz.")
            
            time.sleep(10)  # 10 saniyede bir kontrol
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("IZLEME DURDURULDU")
        print("="*70)
        print(f"Toplam izleme suresi: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Toplam kontrol sayisi: {check_count}")
        print("\nEgitim arka planda devam ediyor olabilir.")
        print("Model dosyalarini kontrol etmek icin tekrar calistirabilirsiniz.")

if __name__ == "__main__":
    # Windows encoding sorunu için
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    monitor_training()
