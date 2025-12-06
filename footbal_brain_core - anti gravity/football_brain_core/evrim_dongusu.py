"""
PRD: Evolution Core - Evrim Döngüsü Çalıştırıcı
Hataları toplar, cluster'lar, çözmeye çalışır, çözemediğinde kullanıcıya sorar
"""
import sys
from pathlib import Path

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.models.evolution_core import EvolutionCore
from src.reporting.learning_notebook_excel import LearningNotebookExporter

print("=" * 80)
print("EVOLUTION CORE - EVRİM DÖNGÜSÜ")
print("=" * 80)
print("PRD: Hataları toplar, cluster'lar, çözmeye çalışır")
print("=" * 80)
print()

try:
    # Evolution Core oluştur
    evolution = EvolutionCore()
    
    # Evrim döngüsünü çalıştır
    print("🔄 Evrim döngüsü başlıyor...\n")
    results = evolution.process_evolution_cycle()
    
    print("\n" + "=" * 80)
    print("EVRİM DÖNGÜSÜ SONUÇLARI")
    print("=" * 80)
    print(f"📥 Error Inbox'a eklenen hata: {results.get('errors_collected', 0)}")
    print(f"📊 Oluşturulan cluster: {results.get('clusters_created', 0)}")
    print(f"✅ Seviye 1'de çözülen: {results.get('solved_level1', 0)}")
    print(f"❓ Kullanıcıya sorulan soru: {results.get('questions_asked', 0)}")
    print(f"⏳ Çözülemeyen: {results.get('unresolved', 0)}")
    print("=" * 80)
    
    # Excel Öğrenme Defteri oluştur
    print("\n📋 Excel Öğrenme Defteri oluşturuluyor...")
    exporter = LearningNotebookExporter()
    notebook_path = exporter.export_learning_notebook()
    print(f"✅ Öğrenme Defteri: {notebook_path}")
    
    print("\n💡 Sonraki adımlar:")
    print("  1. Excel dosyasını aç ve hataları incele")
    print("  2. Kullanıcıya sorulan soruları cevapla (human_feedback tablosu)")
    print("  3. Evolution plan'ları uygula (evolution_plans tablosu)")
    
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()






