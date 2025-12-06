"""
Etkileşimli Öğrenme CLI - Kullanıcıyla birlikte öğrenme
"""
import sys
from football_brain_core.src.models.interactive_learning import InteractiveLearning
from football_brain_core.src.models.self_learning import SelfLearningBrain
from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import ModelVersionRepository
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def interactive_learning_session(season: int):
    """
    Kullanıcıyla etkileşimli öğrenme oturumu başlatır.
    Model hata yaptığında kullanıcıya sorar ve birlikte öğrenir.
    """
    print("🧠 Etkileşimli Öğrenme Oturumu Başlatılıyor...")
    print("=" * 60)
    
    session = get_session()
    try:
        # Model yükle
        active_model = ModelVersionRepository.get_active(session)
        if not active_model:
            print("❌ Aktif model bulunamadı! Önce model eğitmelisin.")
            return
        
        # Model yükleme kodu buraya eklenecek
        # Şimdilik placeholder
        print("⚠️  Model yükleme kısmı implement edilecek")
        
        market_types = [
            MarketType.MATCH_RESULT,
            MarketType.BTTS,
            MarketType.OVER_UNDER_25,
        ]
        
        # SelfLearningBrain oluştur
        # brain = SelfLearningBrain(model, market_types)
        interactive_learner = InteractiveLearning()
        
        print(f"\n📊 Sezon {season} üzerinde öğrenme başlıyor...")
        print("💡 Model yanlış tahmin yaptığında sana soracak.\n")
        
        # Öğrenme döngüsü
        # Bu kısım self_learning.py'deki learn_from_past_matches ile entegre edilecek
        
        print("\n✅ Öğrenme oturumu tamamlandı!")
        
        # Öğrenme özeti
        summary = interactive_learner.get_learning_summary()
        print(f"\n📚 Toplam öğrenme: {summary['total_learnings']}")
        print(f"💡 Öğrenilen konular: {', '.join(summary['learning_topics'])}")
        
    finally:
        session.close()


def process_user_feedback_cli(match_id: int, market_type: str, feedback: str):
    """Kullanıcı geri bildirimini işler"""
    interactive_learner = InteractiveLearning()
    
    from football_brain_core.src.features.market_targets import MarketType as MT
    market_enum = getattr(MT, market_type.upper(), MT.MATCH_RESULT)
    
    result = interactive_learner.process_user_feedback(
        match_id, market_enum, feedback, {}
    )
    
    print(f"✅ Geri bildirim kaydedildi!")
    print(f"💡 Öğrenme noktaları: {result['learning_points']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  python interactive_learn.py learn --season 2023")
        print("  python interactive_learn.py feedback --match-id 123 --market match_result --feedback '...'")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "learn":
        season = int(sys.argv[3]) if "--season" in sys.argv else 2023
        interactive_learning_session(season)
    elif command == "feedback":
        # Feedback işleme
        match_id = int(sys.argv[sys.argv.index("--match-id") + 1])
        market = sys.argv[sys.argv.index("--market") + 1]
        feedback = sys.argv[sys.argv.index("--feedback") + 1]
        process_user_feedback_cli(match_id, market, feedback)







