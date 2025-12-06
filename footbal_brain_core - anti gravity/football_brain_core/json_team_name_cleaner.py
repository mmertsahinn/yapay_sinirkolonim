import json
from pathlib import Path
import re
from typing import Dict

JSON_FILE = "football_brain_export.json"

def get_json_path():
    """JSON dosyasının yolunu döndürür."""
    return Path(__file__).parent / JSON_FILE

def clean_team_name(team_name: str) -> str:
    """Takım isminden başındaki maç skorunu temizler."""
    if not team_name:
        return team_name

    # Regex pattern: Başında parantez içinde skor olan kısmı bul (örn: (2-0), (1-1), (0-3))
    pattern = r'^\(\d+-\d+\)\s*'
    cleaned = re.sub(pattern, '', team_name.strip())

    return cleaned

def main():
    print("=" * 60)
    print("🧹 JSON TAKIM İSİM TEMİZLEYİCİ")
    print(f"📂 Hedef Dosya: {get_json_path().name}")
    print("=" * 60)

    # JSON dosyasını yükle
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

    print(f"📋 Toplam {len(all_matches)} maç kontrol edilecek...")

    cleaned_count = 0

    for idx, match in enumerate(all_matches):
        original_home = match.get('home_team_name', '')
        original_away = match.get('away_team_name', '')

        cleaned_home = clean_team_name(original_home)
        cleaned_away = clean_team_name(original_away)

        # Eğer değişiklik olduysa güncelle
        if cleaned_home != original_home or cleaned_away != original_away:
            if cleaned_home != original_home:
                print(f"✅ [{idx}] Home: '{original_home}' → '{cleaned_home}'")
                match['home_team_name'] = cleaned_home
            if cleaned_away != original_away:
                print(f"✅ [{idx}] Away: '{original_away}' → '{cleaned_away}'")
                match['away_team_name'] = cleaned_away
            cleaned_count += 1

    if cleaned_count == 0:
        print("🎉 Temizlenecek takım ismi bulunamadı!")
        return

    # Değişiklikleri kaydet
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 {cleaned_count} takım ismi temizlendi ve kaydedildi!")
    except Exception as e:
        print(f"❌ Kayıt hatası: {e}")

if __name__ == "__main__":
    main()
