"""
Hızlı test scripti - 10 maç ile sistemi test et
"""
import sys
import traceback

try:
    from run_evolutionary_learning import EvolutionaryLearningSystem
    import argparse
    
    print("="*80)
    print("🧪 HIZLI TEST: 10 MAÇ")
    print("="*80)
    
    # Sistemi başlat
    system = EvolutionaryLearningSystem(config_path="evolutionary_config.yaml")
    
    # 10 maç çalıştır
    print("\n🚀 10 maç çalıştırılıyor...\n")
    system.run(
        csv_path="prediction_matches.csv",
        start_match=0,
        max_matches=10,
        results_csv="results_matches.csv"
    )
    
    print("\n✅ Test tamamlandı!")
    
except Exception as e:
    print(f"\n❌ HATA: {type(e).__name__}: {e}")
    print("\n📋 Traceback:")
    traceback.print_exc()
    sys.exit(1)

