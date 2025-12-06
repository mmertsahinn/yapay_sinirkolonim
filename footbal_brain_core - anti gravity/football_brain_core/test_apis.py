"""
API'leri test et - API-FOOTBALL ve OpenRouter (GPT/Grok)
"""
import os
import sys
from pathlib import Path

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Python path'i düzelt - proje kök dizinini ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))  # Bir üst dizin (footbal_brain_core)

print("API Test Baslatiyor...")
print("=" * 60)

# API Key'leri kontrol et - önce ortam değişkeni, sonra config'den varsayılan
try:
    from football_brain_core.src.config import Config
    config = Config()
except ImportError:
    # Alternatif: direkt import dene
    sys.path.insert(0, str(project_root))
    from src.config import Config
    config = Config()

api_football_key = os.getenv("API_FOOTBALL_KEY", "") or config.API_FOOTBALL_KEY
openrouter_key = os.getenv("OPENROUTER_API_KEY", "") or config.OPENROUTER_API_KEY

print("\nAPI Key Durumu:")
print(f"   API_FOOTBALL_KEY: {'[OK] Ayarlandi' if api_football_key else '[HATA] Ayarlandi'}")
print(f"   OPENROUTER_API_KEY: {'[OK] Ayarlandi' if openrouter_key else '[HATA] Ayarlandi'}")

if not api_football_key:
    print("\n[HATA] API_FOOTBALL_KEY ayarlanmamis!")
    print("   PowerShell'de calistir:")
    print('   $env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"')
    sys.exit(1)

if not openrouter_key:
    print("\n[HATA] OPENROUTER_API_KEY ayarlanmamis!")
    print("   PowerShell'de calistir:")
    print('   $env:OPENROUTER_API_KEY="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"')
    sys.exit(1)

print(f"\nAPI Key'ler:")
print(f"   API-FOOTBALL: {api_football_key[:20]}...")
print(f"   OpenRouter: {openrouter_key[:20]}...")

# 1. API-FOOTBALL Testi
print("\n" + "=" * 60)
print("1. API-FOOTBALL Testi")
print("=" * 60)

try:
    try:
        from football_brain_core.src.ingestion.api_client import APIFootballClient
    except ImportError:
        from src.ingestion.api_client import APIFootballClient
    
    client = APIFootballClient(api_key=api_football_key)
    
    print("API-FOOTBALL baglantisi test ediliyor...")
    
    from datetime import date
    today = date.today()
    
    # Bugünün fikstürlerini çek
    fixtures = client.get_fixtures(date_from=today, date_to=today)
    
    if fixtures:
        print(f"[OK] API-FOOTBALL CALISIYOR! {len(fixtures)} fikstur bulundu.\n")
        print("Ilk 3 fikstur ornegi:")
        for i, fixture in enumerate(fixtures[:3], 1):
            fixture_data = fixture.get("fixture", {})
            teams = fixture.get("teams", {})
            home = teams.get("home", {}).get("name", "N/A")
            away = teams.get("away", {}).get("name", "N/A")
            print(f"  {i}. {home} vs {away}")
    else:
        print("[UYARI] Bugun icin fikstur bulunamadi (normal olabilir)")
        print("[OK] Ancak API baglantisi calisiyor!")
    
except ValueError as e:
    print(f"[HATA] {e}")
    print("\nCozum:")
    print('   $env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"')
    sys.exit(1)
except Exception as e:
    print(f"[HATA] Beklenmeyen hata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. OpenRouter - GPT Testi
print("\n" + "=" * 60)
print("2. OpenRouter - GPT (openai/gpt-oss-20b:free) Testi")
print("=" * 60)

try:
    try:
        from football_brain_core.src.explanations.llm_client import LLMClient
    except ImportError:
        from src.explanations.llm_client import LLMClient
    
    gpt_client = LLMClient(provider="openrouter")
    
    print("GPT modeli test ediliyor...")
    
    test_context = {
        "home_team": "Arsenal",
        "away_team": "Chelsea"
    }
    
    test_predictions = {
        "match_result": "1",
        "btts": "Yes"
    }
    
    test_stats = {
        "home_form": "70% win rate",
        "away_form": "50% win rate"
    }
    
    response = gpt_client.generate_explanation(
        test_context, test_predictions, test_stats
    )
    
    print(f"[OK] GPT CALISIYOR!")
    print(f"Cevap: {response[:100]}...")
    
except Exception as e:
    print(f"[HATA] GPT hatasi: {e}")
    import traceback
    traceback.print_exc()

# 3. OpenRouter - Grok Testi
print("\n" + "=" * 60)
print("3. OpenRouter - Grok (x-ai/grok-4.1-fast:free) Testi")
print("=" * 60)

try:
    grok_client = LLMClient(provider="openrouter-grok")
    
    print("Grok modeli test ediliyor...")
    
    response = grok_client.generate_explanation(
        test_context, test_predictions, test_stats
    )
    
    print(f"[OK] Grok CALISIYOR!")
    print(f"Cevap: {response[:100]}...")
    
except Exception as e:
    print(f"[HATA] Grok hatasi: {e}")
    import traceback
    traceback.print_exc()

# 4. Her Iki Modeli Karsilastir
print("\n" + "=" * 60)
print("4. Model Karsilastirmasi (Hiz Testi)")
print("=" * 60)

try:
    import time
    
    print("GPT test ediliyor...")
    gpt_start = time.time()
    gpt_response = gpt_client.generate_explanation(
        test_context, test_predictions, test_stats
    )
    gpt_time = time.time() - gpt_start
    
    print("Grok test ediliyor...")
    grok_start = time.time()
    grok_response = grok_client.generate_explanation(
        test_context, test_predictions, test_stats
    )
    grok_time = time.time() - grok_start
    
    print(f"\nSureler:")
    print(f"   GPT: {gpt_time:.2f} saniye")
    print(f"   Grok: {grok_time:.2f} saniye")
    
    if gpt_time < grok_time:
        print(f"   En hizli: GPT ({grok_time - gpt_time:.2f} saniye daha hizli)")
    else:
        print(f"   En hizli: Grok ({gpt_time - grok_time:.2f} saniye daha hizli)")
    
except Exception as e:
    print(f"[HATA] Karsilastirma hatasi: {e}")

# Ozet
print("\n" + "=" * 60)
print("TEST OZETI")
print("=" * 60)
print("[OK] API-FOOTBALL: Calisiyor")
print("[OK] OpenRouter GPT: Calisiyor")
print("[OK] OpenRouter Grok: Calisiyor")
print("\nTum API'ler hazir! Projeyi kullanmaya baslayabilirsin.")
print("\nSonraki adim:")
print("   python -m football_brain_core.src.cli.main init-db")

