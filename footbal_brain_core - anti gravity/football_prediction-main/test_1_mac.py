"""
🔍 1 MAÇLIK TEST - TÜM SİSTEMLERİ KONTROL ET!
===========================================

Hızlı test için sadece 1 maç!
Tüm debug mesajları görünür!
"""

import sys
sys.argv = [
    'test_1_mac.py',
    '--csv', '2025_temmuz_sonrasi_SONUCLAR.csv',
    '--results', '2025_temmuz_sonrasi_SONUCLAR.csv',
    '--max', '1',  # SADECE 1 MAÇ!
    '--start', '0'
]

print("=" * 100)
print("🔍 1 MAÇLIK HIZLI TEST BAŞLIYOR!")
print("=" * 100)
print("\nKONTROL EDİLECEKLER:")
print("  ✅ Population History çalışıyor mu?")
print("  ✅ Dynamic Relocation çalışıyor mu?")
print("  ✅ Hall Audit çalışıyor mu?")
print("  ✅ Team Spec çalışıyor mu?")
print("  ✅ Sync çalışıyor mu?")
print("  ✅ Loglar yazılıyor mu?")
print("=" * 100)
print()

# Ana sistemi çalıştır
from run_evolutionary_learning import main
main()

print()
print("=" * 100)
print("🔍 TEST BİTTİ!")
print("=" * 100)
print("\nLOG DOSYALARINI KONTROL ET:")
print("  • evolution_logs/📚_POPULATION_HISTORY.txt")
print("  • evolution_logs/🔄_DYNAMIC_RELOCATION.log")
print("  • evolution_logs/🔬_HALL_SPEC_AUDIT.log")
print("=" * 100)

