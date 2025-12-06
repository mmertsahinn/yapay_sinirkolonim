import json
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from src.ingestion.alternative_hype_scraper import AlternativeHypeScraper
import argparse
import os
import threading
import signal
import sys

# Global değişkenler
scraper_lock = threading.Lock()
global_scraper = None
full_data_global = None  # Son işlenen veriyi saklamak için


def signal_handler(sig, frame):
    """Terminal kapatılırken çalışan handler"""
    print("\n\n⚠️ PROGRAM KAPATILIYOR...", flush=True)

    if full_data_global is not None:
        print("💾 Son veriler diske yazılıyor...", flush=True)
        try:
            save_to_disk(full_data_global)
            print("✅ Veriler başarıyla kaydedildi!", flush=True)
        except Exception as e:
            print(f"❌ Kayıt hatası: {e}", flush=True)

    print("🛑 Program kapatıldı.", flush=True)
    sys.exit(0)


# Signal handler'ı kayıt et
signal.signal(signal.SIGINT, signal_handler)
if sys.platform == "win32":
    signal.signal(signal.SIGTERM, signal_handler)


def get_scraper():
    """Global scraper nesnesini döndür (thread-safe)"""
    global global_scraper
    with scraper_lock:
        if global_scraper is None:
            global_scraper = AlternativeHypeScraper()
    return global_scraper

# Takım listesi
TEAMS = [
    "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir FK", "Kasımpaşa", "Çaykur Rizespor", "Sivasspor", "Antalyaspor", "Alanyaspor", "Gaziantep FK", "MKE Ankaragücü", "Hatayspor", "Kayserispor", "Adana Demirspor", "Göztepe", "Eyüpspor", "Bodrum FK",
    "Manchester City", "Arsenal", "Liverpool", "Manchester United", "Chelsea", "Tottenham", "Newcastle United", "Aston Villa", "West Ham United", "Brighton", "Bournemouth", "Wolverhampton", "Crystal Palace", "Fulham", "Brentford", "Everton", "Nottingham Forest", "Ipswich Town", "Leicester City", "Southampton",
    "Real Madrid", "FC Barcelona", "Atlético Madrid", "Athletic Bilbao", "Real Sociedad", "Villarreal", "Real Betis", "Valencia", "Sevilla", "Girona", "Osasuna", "Rayo Vallecano", "Getafe", "Mallorca", "Las Palmas", "Deportivo Alavés", "Celta Vigo", "Leganés", "Valladolid", "Espanyol",
    "Inter", "AC Milan", "Juventus", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina", "Bologna", "Torino", "Udinese", "Genoa", "Sampdoria", "Lecce", "Empoli", "Verona", "Monza", "Parma", "Venezia", "Como",
    "Bayern Münih", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Stuttgart", "Wolfsburg", "Mönchengladbach", "Eintracht Frankfurt", "Werder Bremen", "Freiburg", "Mainz 05", "Augsburg", "Köln", "Bochum", "Hoffenheim", "Union Berlin", "Heidenheim", "St. Pauli",
    "Paris Saint-Germain", "AS Monaco", "Olympique Lyon", "Olympique Marseille", "Lille", "Rennes", "Nice", "Montpellier", "Nantes", "Reims", "Strasbourg", "Lens", "Toulouse", "Lorient", "Auxerre", "Brest", "Metz", "Angers",
    "Benfica", "Porto", "Sporting CP", "Braga", "Vitória de Guimarães", "Famalicão", "Boavista", "Casa Pia", "Portimonense", "Rio Ave", "Estoril", "Moreirense", "Farense", "Gil Vicente", "Estrela da Amadora", "Nacional", "Santa Clara", "AVS Futebol SAD"
]

JSON_FILE = "football_brain_export.json"
MAX_WORKERS = 10


def get_json_path():
    """JSON dosyasının yolunu döndürür."""
    return Path(__file__).parent / JSON_FILE


def save_to_disk(full_data: Dict):
    """Veriyi diske yazar."""
    path = get_json_path()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2, default=str)


def scrape_match_data(match: Dict):
    """Tek bir maç için hype verisini detaylı şekilde çeker (olmuyorsa atlar, tekrar denemez)."""
    home = match.get('home_team_name')
    away = match.get('away_team_name')
    league = match.get('league_name')
    date_str = match.get('match_date')

    if not home or not away:
        print(f"⚠️ Geçersiz maç verisi: {home} vs {away}", flush=True)
        return None

    try:
        print(f"🔍 {home} vs {away} için hype verisi çekiliyor (detaylı araştırma)...", flush=True)
        scraper = get_scraper()
        match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        result = scraper.get_match_hype(league, home, away, match_date)

        if result and result.get("total_mentions", 0) > 0:
            print(f"✅ {home} vs {away}: {result.get('total_mentions', 0)} tweet bulundu", flush=True)
            return result
        else:
            print(f"⚠️ {home} vs {away}: Veri bulunamadı, atlanıyor", flush=True)
            return None

    except Exception as e:
        print(f"❌ Hata ({home} vs {away}): {e} - Atlanıyor", flush=True)
        return None


# Komut satırı argümanlarını işlemek için argparse kullanımı
def parse_arguments():
    parser = argparse.ArgumentParser(description="JSON Hype Detaylı Doldurucu")
    parser.add_argument("--threads", type=int, default=10, help="Thread sayısını belirtin (varsayılan: 10)")
    parser.add_argument("--log_interval", type=int, default=25, help="Yazdırma sıklığını maç sayısı cinsinden belirtin (varsayılan: 25 maç)")
    return parser.parse_args()


def main():
    global full_data_global

    args = parse_arguments()
    max_workers = args.threads
    log_interval = args.log_interval

    print("=" * 60)
    print("🚀 JSON HYPE DETAYLI DOLDURUCU (UZUN SÜRELİ MOD)")
    print(f"📂 Hedef Dosya: {get_json_path().name}")
    print(f"🔧 Thread Sayısı: {max_workers}")
    print(f"🔄 Retry Sayısı: 5 (detaylı çekim için)")
    print("=" * 60, flush=True)

    while True:
        # JSON dosyasını yükle
        path = get_json_path()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
                full_data_global = full_data  # Global değişkene ata
        except Exception as e:
            print(f"❌ Kritik Hata: Dosya okunamadı! {e}", flush=True)
            time.sleep(5)
            continue

        all_matches = full_data.get('data', {}).get('matches', [])

        if not all_matches:
            print("❌ Dosyada maç bulunamadı!", flush=True)
            time.sleep(5)
            continue

        # Belirtilen takımları filtrele
        todo_list = []
        for idx, match in enumerate(all_matches):
            home = match.get('home_team_name')
            away = match.get('away_team_name')

            if home in TEAMS or away in TEAMS:
                is_hype_missing = (match.get('hype_updated_at') is None) or \
                                  (match.get('home_support') is None) or \
                                  (match.get('total_tweets') == 0)

                if is_hype_missing:
                    todo_list.append((idx, match))

        if not todo_list:
            print("🎉 Yapılacak iş kalmadı!", flush=True)
            time.sleep(5)
            continue

        total_matches = len(all_matches)
        missing_count = len(todo_list)
        print(f"📋 Toplam {total_matches} maçtan {missing_count} tanesinin hype verisi yok", flush=True)
        print(f"🏃 İşlenecek maç sayısı: {missing_count}/{total_matches}", flush=True)

        # İşleme başla
        processed_count = 0
        batch_size = max(max_workers, 5)  # Detaylı çekim için batch size'i azalttık
        start_time = time.time()
        last_log_time = time.time()

        for i in range(0, len(todo_list), batch_size):
            batch = todo_list[i:i + batch_size]

            # Her batch'ten sonra garbage collection yapıp bellek temizle
            import gc
            gc.collect()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(scrape_match_data, item[1]): item[0]
                    for item in batch
                }

                for future in as_completed(future_to_idx):
                    original_idx = future_to_idx[future]
                    result = future.result()

                    if result:
                        target_match = all_matches[original_idx]
                        target_match['home_support'] = result.get("home_support", 0.5)
                        target_match['away_support'] = result.get("away_support", 0.5)
                        target_match['sentiment_score'] = result.get("sentiment_score", 0.0)
                        target_match['total_tweets'] = result.get("total_mentions", 0)
                        target_match['hype_updated_at'] = datetime.now().isoformat()
                        processed_count += 1

                        # Her log_interval maçta bir geri dönüt ver
                        if processed_count % log_interval == 0:
                            elapsed = time.time() - last_log_time
                            minutes = elapsed / 60
                            remaining = len(todo_list) - processed_count
                            tweets = target_match.get('total_tweets', 0)
                            print(f"✅ {processed_count}/{len(todo_list)} | ⏱️ {minutes:.2f}dk | 📊 Kalan: {remaining} / Toplam maçlar: {total_matches} | 📢 {tweets} tweet", flush=True)
                            last_log_time = time.time()

            save_to_disk(full_data)
            time.sleep(1)  # Batch'ler arasında daha uzun mola

        total_time = (time.time() - start_time) / 60
        print(f"✅ {len(todo_list)} maç tamamlandı | ⏱️ Toplam: {total_time:.1f}dk", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
