"""
JSON HYPE FILLER - HIZLI KAYIT (INSTANT SAVE)
- Listenin EN SONUNDAN başlar.
- Her 5 maçta bir diske kaydeder (Anlık değişim görünür).
- Eksik verileri (null) doldurur.
- 10 Thread ile hızlı çalışır.
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket

# Konsol çıktı ayarları
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Proje dizin ayarları
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))
sys.path.insert(0, str(project_root))

from src.ingestion.alternative_hype_scraper import AlternativeHypeScraper

# --- AYARLAR ---
JSON_FILE = "football_brain_export.json"
MAX_WORKERS = 100   # Hız: 10 Motorlu
BATCH_SIZE = 1000     # DÜZELTME: Her 5 maçta bir kaydeder (Sonucu hemen görmek için)

def get_json_path():
    """Dosyanın tam yolunu bulur"""
    # 1. Çalışma dizinine bak
    path = Path(os.getcwd()) / JSON_FILE
    if path.exists():
        return path
    
    # 2. Dosyanın yanına bak
    path = Path(__file__).parent / JSON_FILE
    if path.exists():
        return path
        
    # 3. Proje köküne bak
    path = Path(project_root) / JSON_FILE
    return path

def save_to_disk(full_data: Dict):
    """Veriyi diske yazar ve bilgi verir"""
    path = get_json_path()
    try:
        # Geçici bir dosyaya yazıp sonra ismini değiştirmek daha güvenlidir ama
        # Windows'ta bazen sorun çıkarır, direkt yazıyoruz.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except PermissionError:
        print(f"❌ HATA: Dosya şu an açık! Lütfen JSON dosyasını kapatın.", flush=True)
        return False
    except Exception as e:
        print(f"❌ KAYIT HATASI: {e}", flush=True)
        return False

# İnternet bağlantısını kontrol eden fonksiyon
def check_internet_connection():
    try:
        # Google DNS sunucusuna bağlanmayı dene
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False

def scrape_match_data(match: Dict):
    """
    Tek bir maç için veriyi çeker.
    """
    home = match.get('home_team_name')
    away = match.get('away_team_name')
    league = match.get('league_name')
    date_str = match.get('match_date')
    
    if not home or not away:
        return None

    print(f"➡️ Başladı: {home} vs {away}", flush=True)

    # İNATÇI MOD
    while True:
        try:
            # İnternet bağlantısını kontrol et
            if not check_internet_connection():
                print("⚠️ İnternet bağlantısı yok. Yeniden bağlanmayı bekliyor...", flush=True)
                time.sleep(10)  # 10 saniye bekle
                continue

            scraper = AlternativeHypeScraper()
            match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            result = scraper.get_match_hype(league, home, away, match_date)
            return result
        except Exception as e:
            print(f"⚠️ HATA ({home} vs {away}): {e}. Bekleniyor...", flush=True)
            time.sleep(3)  # Hata durumunda bekleme süresi

def main():
    print("="*60)
    print("🚀 JSON HYPE DOLDURUCU (HIZLI KAYIT MODU)")
    print(f"📂 Hedef Dosya: {get_json_path().name}")
    print(f"💾 Kayıt Sıklığı: Her {BATCH_SIZE} maçta bir")
    print("="*60, flush=True)

    # 1. DOSYAYI YÜKLE
    path = get_json_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"❌ Kritik Hata: Dosya okunamadı! {e}")
        return
    
    # Matches referansını al
    all_matches = full_data.get('data', {}).get('matches', [])
    
    if not all_matches:
        print("❌ Dosyada maç bulunamadı!")
        return

    # 2. İŞLENECEK LİSTEYİ HAZIRLA (TERS ÇEVİR)
    print("🔄 Liste analiz ediliyor (Sondan başa)...")
    
    # (Index, Match) çiftleri
    indexed_matches = list(enumerate(all_matches))
    reversed_matches = indexed_matches[::-1]
    
    # Eksik hype verisi olan ilk kayıttan başlamak için listeyi filtrele
    first_missing_index = None
    for idx, m in indexed_matches:
        is_hype_missing = (m.get('hype_updated_at') is None) or \
                          (m.get('home_support') is None) or \
                          (m.get('total_tweets') == 0)

        if is_hype_missing:
            first_missing_index = idx
            break

    if first_missing_index is not None:
        indexed_matches = indexed_matches[first_missing_index:]
        reversed_matches = indexed_matches[::-1]
    else:
        print("🎉 Yapılacak iş kalmadı! Tüm veriler zaten dolu.", flush=True)
        return

    # Devam eden işlemler için güncellenmiş listeyi kullan
    todo_list = []
    for idx, m in reversed_matches:
        is_hype_missing = (m.get('hype_updated_at') is None) or \
                          (m.get('home_support') is None) or \
                          (m.get('total_tweets') == 0)

        has_score = (m.get('home_score') is not None)

        if has_score and is_hype_missing:
            todo_list.append((idx, m))

    if not todo_list:
        print("🎉 Yapılacak iş kalmadı! Tüm veriler zaten dolu.", flush=True)
        return

    print(f"📋 Doldurulacak {len(todo_list)} eksik maç bulundu.", flush=True)
    print("-" * 40)

    # 3. İŞLEME BAŞLA
    processed_count = 0
    
    # Batch döngüsü
    for i in range(0, len(todo_list), BATCH_SIZE):
        batch = todo_list[i : i + BATCH_SIZE]
        matches_modified_count = 0
        
        print(f"\n⚡ Grup İşleniyor ({i+1} - {i+len(batch)})...", flush=True)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(scrape_match_data, item[1]): item[0] 
                for item in batch
            }
            
            for future in as_completed(future_to_idx):
                original_idx = future_to_idx[future]
                result = future.result()
                
                if result:
                    # DİKKAT: Doğrudan hafızadaki ana listeyi güncelliyoruz
                    target_match = all_matches[original_idx]
                    
                    target_match['home_support'] = result.get("home_support", 0.5)
                    target_match['away_support'] = result.get("away_support", 0.5)
                    target_match['sentiment_score'] = result.get("sentiment_score", 0.0)
                    target_match['total_tweets'] = result.get("total_mentions", 0)
                    target_match['hype_updated_at'] = datetime.now().isoformat()
                    
                    matches_modified_count += 1
                    
                    # Log
                    teams = f"{target_match['home_team_name']} vs {target_match['away_team_name']}"
                    mentions = target_match['total_tweets']
                    print(f"✅ [{original_idx}] {teams} | 📢 {mentions}", flush=True)

        # 4. KAYIT (HER 5 MAÇTA BİR)
        if matches_modified_count > 0:
            print("💾 Dosyaya yazılıyor...", flush=True)
            if save_to_disk(full_data):
                processed_count += matches_modified_count
                print(f"✅ JSON GÜNCELLENDİ! (Toplam: {processed_count} maç işlendi)", flush=True)
            else:
                print("⚠️ DİKKAT: Kayıt yapılamadı (Dosya açık olabilir)", flush=True)
        
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Durduruldu.")


