"""API limit bildirimi testi"""
import os
import sys
from datetime import date

# API key'i ayarla
os.environ["API_FOOTBALL_KEY"] = "647f5de88a29d150a9d4e2c0c7b636fb"

# Path ayarla
sys.path.insert(0, os.path.dirname(__file__))

from src.ingestion.api_client import APIFootballClient

print("=" * 70)
print("API Limit Bildirimi Testi")
print("=" * 70)

# Callback fonksiyonu tanımla
def on_limit_warning(remaining: int, limit: int):
    """Limit uyarısı callback'i"""
    print(f"\n[CALLBACK] Limit uyarisi: {remaining}/{limit} kaldi")

# API client oluştur
client = APIFootballClient(on_limit_warning=on_limit_warning)

print(f"\nAPI Client olusturuldu")
print(f"Gunluk limit: {client.daily_limit}")

# Bir test isteği yap (limit durumunu görmek için)
print("\nTest istegi yapiliyor...")
try:
    fixtures = client.get_fixtures(date_from=date(2025, 11, 28), date_to=date(2025, 11, 28))
    print(f"[OK] Test basarili! {len(fixtures) if fixtures else 0} fikstur bulundu.")
    
    # Limit durumunu göster
    remaining = client.daily_limit - client.requests_today
    print(f"\nLimit Durumu:")
    print(f"  Kullanilan: {client.requests_today}/{client.daily_limit}")
    print(f"  Kalan: {remaining}/{client.daily_limit}")
    
    if remaining <= 10:
        print(f"\n[UYARI] Limit azaliyor! {remaining} kaldi.")
    elif remaining == 0:
        print(f"\n[UYARI] LIMIT DOLDU!")
    else:
        print(f"\n[OK] Limit yeterli.")
        
except Exception as e:
    print(f"[HATA] {e}")

print("\n" + "=" * 70)
print("Test tamamlandi!")
print("=" * 70)
print("\nNot: Limit doldugunda veya azaldiginda otomatik bildirim gonderilir.")
print("Windows'ta toast bildirimi gorunecektir.")






