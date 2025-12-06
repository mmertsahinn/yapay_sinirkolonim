import os
import shutil
import glob
import sys

# Windows konsolunda emoji desteği için
sys.stdout.reconfigure(encoding='utf-8')

def reset_system():
    """
    🌌 GENESIS PROTOCOL (System Reset)
    
    Tüm verileri siler ve sistemi "Büyük Patlama" öncesine döndürür.
    """
    print(f"{'='*60}")
    print(f"🌌 GENESIS PROTOCOL BAŞLATILIYOR...")
    print(f"{'='*60}")
    
    # 1. Klasörler
    dirs_to_clean = [
        "evolution_logs",
        "checkpoints",
        "lora_models",
        "visualizations",
        "best_loras"  # 🌟 YENİ EXPORT KLASÖRÜ
    ]
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"🧹 Temizleniyor: {dir_name}...")
            # İçindeki tüm dosyaları sil
            files = glob.glob(os.path.join(dir_path, "*"))
            for f in files:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                except Exception as e:
                    print(f"   ❌ Hata: {f} silinemedi ({e})")
        else:
            print(f"✨ Oluşturuluyor: {dir_name}...")
            os.makedirs(dir_path, exist_ok=True)
            
    # 2. Özel Dosyalar (Varsa sil)
    files_to_delete = [
        "population_history.csv",
        "events_log.csv",
        "detailed_lora_history.csv"
    ]
    
    for fname in files_to_delete:
        fpath = os.path.join(root_dir, "evolution_logs", fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                print(f"🗑️ Silindi: {fname}")
            except Exception as e:
                print(f"   ❌ Hata: {fname} silinemedi ({e})")

    print(f"\n{'='*60}")
    print(f"✨ SİSTEM SIFIRLANDI (TABULA RASA)")
    print(f"🚀 'run_evolutionary_learning.py' çalıştırılarak İLK İNSANLAR yaratılabilir!")
    print(f"{'='*60}")

if __name__ == "__main__":
    confirm = input("⚠️ TÜM VERİLER SİLİNECEK! Onaylıyor musunuz? (evet/hayır): ")
    if confirm.lower() == "evet":
        reset_system()
    else:
        print("❌ İptal edildi.")
