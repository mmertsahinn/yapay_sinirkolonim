"""
MASTER CONTROLLER - TÜM SİSTEMLERİ YÖNETİR
Hype ve Odds sistemlerini sürekli çalıştırır, hataları otomatik çözer
"""
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_controller.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def start_hype_system():
    """Hype sistemini başlat"""
    script_path = Path(__file__).parent / "hype_surekli_calis.py"
    return subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )

def start_odds_system():
    """Odds sistemini başlat"""
    script_path = Path(__file__).parent / "odds_surekli_calis.py"
    return subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )

def main():
    print("=" * 80)
    print("🎯 MASTER CONTROLLER - TÜM SİSTEMLER BAŞLATILIYOR")
    print("=" * 80)
    print("📋 Sistemler:")
    print("  ✅ Hype Sürekli Çalışan Sistem")
    print("  ✅ Odds Sürekli Çalışan Sistem")
    print("=" * 80)
    print()
    
    hype_process = None
    odds_process = None
    
    try:
        # Hype sistemini başlat
        logger.info("🔥 Hype sistemi başlatılıyor...")
        hype_process = start_hype_system()
        print("✅ Hype sistemi başlatıldı!")
        
        time.sleep(2)
        
        # Odds sistemini başlat
        logger.info("🎲 Odds sistemi başlatılıyor...")
        odds_process = start_odds_system()
        print("✅ Odds sistemi başlatıldı!")
        
        print("\n" + "=" * 80)
        print("🚀 TÜM SİSTEMLER ÇALIŞIYOR!")
        print("=" * 80)
        print("📊 İlerleme log dosyalarında takip edilebilir:")
        print("   - hype_surekli.log")
        print("   - odds_surekli.log")
        print("=" * 80)
        print("\n⚠️ Sistemler arka planda çalışıyor. Kapatmak için Ctrl+C basın.")
        
        # Sürekli kontrol et
        while True:
            time.sleep(60)  # Her 60 saniyede bir kontrol et
            
            # Process'lerin çalışıp çalışmadığını kontrol et
            if hype_process and hype_process.poll() is not None:
                logger.warning("⚠️ Hype sistemi durdu! Yeniden başlatılıyor...")
                hype_process = start_hype_system()
                print("✅ Hype sistemi yeniden başlatıldı!")
            
            if odds_process and odds_process.poll() is not None:
                logger.warning("⚠️ Odds sistemi durdu! Yeniden başlatılıyor...")
                odds_process = start_odds_system()
                print("✅ Odds sistemi yeniden başlatıldı!")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Sistemler durduruluyor...")
        if hype_process:
            hype_process.terminate()
        if odds_process:
            odds_process.terminate()
        print("✅ Sistemler durduruldu!")

if __name__ == "__main__":
    main()





