"""
Sürekli 10'ar maçlık eğitim döngüsü
Her session tamamlanana kadar devam eder, hataları yakalar ve düzeltir
"""
import sys
import traceback
import time
from run_evolutionary_learning import EvolutionaryLearningSystem

def run_continuous_training(max_sessions=None, matches_per_session=10):
    """
    Sürekli 10'ar maçlık sessionlar çalıştır
    
    Args:
        max_sessions: Maksimum session sayısı (None = sınırsız)
        matches_per_session: Her session'da kaç maç (default: 10)
    """
    session_count = 0
    total_matches = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    print("="*80)
    print("🔄 SÜREKLI EĞİTİM BAŞLATILIYOR")
    print("="*80)
    print(f"   Her session: {matches_per_session} maç")
    print(f"   Maksimum session: {'Sınırsız' if max_sessions is None else max_sessions}")
    print("="*80)
    
    while True:
        # Maksimum session kontrolü
        if max_sessions is not None and session_count >= max_sessions:
            print(f"\n✅ Maksimum session sayısına ulaşıldı: {max_sessions}")
            break
        
        session_count += 1
        print(f"\n{'='*80}")
        print(f"📊 SESSION #{session_count} BAŞLIYOR ({matches_per_session} maç)")
        print(f"{'='*80}")
        
        try:
            # Sistemi başlat (her session'da yeni instance)
            print(f"\n🔧 Sistem başlatılıyor...")
            system = EvolutionaryLearningSystem(config_path="evolutionary_config.yaml")
            
            # Mevcut durumu yükle (varsa)
            import os
            if os.path.exists(system.paths['lora_population']):
                print(f"📂 Kaydedilmiş durum yükleniyor...")
                system.load_state()
                start_match = system.evolution_manager.match_count
                print(f"   ✅ Kaldığı yerden devam: Maç #{start_match}")
            else:
                start_match = 0
                print(f"   🆕 Yeni koloni başlatılıyor")
            
            # Bu session için maksimum maç sayısı
            session_max_matches = matches_per_session
            
            print(f"\n🚀 Session #{session_count} çalıştırılıyor...")
            print(f"   Başlangıç maçı: {start_match}")
            print(f"   Bu session'da: {session_max_matches} maç")
            
            # Session'ı çalıştır
            system.run(
                csv_path="prediction_matches.csv",
                start_match=start_match,
                max_matches=session_max_matches,
                results_csv="results_matches.csv"
            )
            
            # Session başarılı
            total_matches += session_max_matches
            consecutive_errors = 0  # Hata sayacını sıfırla
            
            print(f"\n✅ SESSION #{session_count} TAMAMLANDI!")
            print(f"   Bu session: {session_max_matches} maç")
            print(f"   Toplam maç: {total_matches}")
            print(f"   Popülasyon: {len(system.evolution_manager.population)} LoRA")
            
            # Durumu kaydet
            print(f"\n💾 Durum kaydediliyor...")
            system.save_state()
            print(f"   ✅ Durum kaydedildi!")
            
            # Kısa bir bekleme (sistemin stabilize olması için)
            print(f"\n⏳ Sonraki session için 2 saniye bekleniyor...")
            time.sleep(2)
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Kullanıcı tarafından durduruldu!")
            print(f"   Tamamlanan session: {session_count}")
            print(f"   Toplam maç: {total_matches}")
            break
            
        except Exception as e:
            consecutive_errors += 1
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"\n❌ SESSION #{session_count} HATASI!")
            print(f"   Hata tipi: {error_type}")
            print(f"   Mesaj: {error_msg}")
            print(f"   Ardışık hata: {consecutive_errors}/{max_consecutive_errors}")
            
            # Traceback'i göster
            print(f"\n📋 Detaylı hata:")
            traceback.print_exc()
            
            # Çok fazla ardışık hata varsa dur
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n🛑 Çok fazla ardışık hata ({consecutive_errors})!")
                print(f"   Sistem durduruluyor. Lütfen hataları kontrol edin.")
                break
            
            # Hata sonrası bekleme (sistemin toparlanması için)
            print(f"\n⏳ Hata sonrası 5 saniye bekleniyor...")
            time.sleep(5)
            
            # Sonraki session'a geç (aynı session'ı tekrar deneme)
            continue
    
    # Final özet
    print(f"\n{'='*80}")
    print(f"📊 EĞİTİM ÖZETİ")
    print(f"{'='*80}")
    print(f"   Tamamlanan session: {session_count}")
    print(f"   Toplam maç: {total_matches}")
    print(f"   Ortalama maç/session: {total_matches/session_count if session_count > 0 else 0:.1f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sürekli 10\'ar maçlık eğitim döngüsü')
    parser.add_argument('--sessions', type=int, default=None, 
                       help='Maksimum session sayısı (None = sınırsız)')
    parser.add_argument('--matches', type=int, default=10,
                       help='Her session\'da kaç maç (default: 10)')
    
    args = parser.parse_args()
    
    try:
        run_continuous_training(
            max_sessions=args.sessions,
            matches_per_session=args.matches
        )
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Program kullanıcı tarafından durduruldu!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ KRİTİK HATA: {e}")
        traceback.print_exc()
        sys.exit(1)

