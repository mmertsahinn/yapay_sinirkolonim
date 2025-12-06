"""
EVRİMLEŞEN MODEL SİSTEMİ
Model her hata yaptığında otomatik olarak kendini evrimleştirir
- Hata analizi
- Model güncelleme
- Feature iyileştirme
- Parametre optimizasyonu
"""
import sys
from pathlib import Path
import torch
from datetime import datetime

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.config import Config
from src.models.train_offline import OfflineTrainer
from src.models.self_learning import SelfLearningBrain
from src.models.multi_task_model import MultiTaskModel
from src.features.market_targets import MarketType
from src.db.connection import get_session
from src.db.repositories import LeagueRepository, ModelVersionRepository
from src.features.feature_builder import FeatureBuilder

print("=" * 80)
print("EVRİMLEŞEN MODEL SİSTEMİ")
print("=" * 80)
print("Model her hata yaptığında otomatik olarak kendini evrimleştirecek")
print("=" * 80)
print()

config = Config()
config.MODEL_CONFIG.epochs = 50
config.MODEL_CONFIG.batch_size = 32

# Marketler
market_types = [
    MarketType.MATCH_RESULT,
    MarketType.BTTS,
    MarketType.OVER_UNDER_25,
    MarketType.GOAL_RANGE,
    MarketType.CORRECT_SCORE,
    MarketType.DOUBLE_CHANCE,
]

print(f"📊 Marketler: {[m.value for m in market_types]}")
print(f"⚙️  Epochs: {config.MODEL_CONFIG.epochs}")
print()

# Ligler
session = get_session()
try:
    league_ids = [
        LeagueRepository.get_or_create(session, league.name).id
        for league in config.TARGET_LEAGUES
    ]
    print(f"🏆 Ligler: {len(league_ids)} lig\n")
finally:
    session.close()

# Eğitim yıllarını bul
session = get_session()
try:
    from sqlalchemy import func, extract
    from src.db.schema import Match
    
    years_query = session.query(
        extract('year', Match.match_date).label('year')
    ).distinct().order_by('year').all()
    
    available_years = sorted([int(y[0]) for y in years_query])
    train_years = [y for y in available_years if y <= 2022]
    
    if not train_years:
        train_years = available_years[:3] if len(available_years) >= 3 else available_years
    
    print(f"📅 Eğitim yılları: {train_years[0]} - {train_years[-1]}")
    print(f"📅 Validation: {train_years[-1]}\n")
    
finally:
    session.close()

try:
    print("=" * 80)
    print("1. ADIM: İLK MODEL EĞİTİMİ")
    print("=" * 80)
    
    # İlk model eğitimi
    trainer = OfflineTrainer(market_types, config, model_config={
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3
    })
    
    model = trainer.train(train_years, [train_years[-1]], league_ids)
    
    # Modeli kaydet
    initial_model_path = "model_evolution_v1.0.pth"
    torch.save(model.state_dict(), initial_model_path)
    print(f"\n✅ İlk model kaydedildi: {initial_model_path}")
    
    print("\n" + "=" * 80)
    print("2. ADIM: EVRİMLEŞME SÜRECİ")
    print("=" * 80)
    print("Model eski maçları test ediyor, hataları buluyor ve kendini evrimleştiriyor...")
    print()
    
    # SelfLearningBrain oluştur
    brain = SelfLearningBrain(model, market_types, config)
    
    # Evrimleşme: Geçmiş maçlardan öğren
    evolution_results = brain.learn_from_past_matches(
        season=train_years[-1],  # Son sezon üzerinde test et
        league_ids=league_ids,
        max_iterations=10,  # 10 iterasyon evrimleşme
        target_accuracy=0.70
    )
    
    print("\n" + "=" * 80)
    print("3. ADIM: EVRİMLEŞMİŞ MODEL KAYDI")
    print("=" * 80)
    
    # Evrimleşmiş modeli kaydet
    evolved_model_path = "model_evolution_v1.0_evolved.pth"
    torch.save(brain.model.state_dict(), evolved_model_path)
    print(f"✅ Evrimleşmiş model kaydedildi: {evolved_model_path}")
    
    # Model versiyonunu güncelle
    session = get_session()
    try:
        # Önceki versiyonları deaktif et
        ModelVersionRepository.deactivate_all(session)
        
        # Yeni versiyon oluştur
        version = "v1.0-evolved"
        description = f"Evrimleşmiş Model - İlk: {train_years[0]}-{train_years[-1]}, " \
                     f"En iyi doğruluk: {evolution_results.get('best_accuracy', 0):.2%}, " \
                     f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        model_version = ModelVersionRepository.create(session, version, description)
        session.commit()
        
        print(f"✅ Model versiyonu kaydedildi: {version}")
        print(f"📊 En iyi doğruluk: {evolution_results.get('best_accuracy', 0):.2%}")
        print(f"🔄 Toplam iterasyon: {evolution_results.get('total_iterations', 0)}")
        
    finally:
        session.close()
    
    print("\n" + "=" * 80)
    print("✅ EVRİMLEŞME TAMAMLANDI!")
    print("=" * 80)
    print("\n📝 Model artık:")
    print("  • Her hata yaptığında otomatik analiz yapıyor")
    print("  • Hatalardan öğrenerek kendini güncelliyor")
    print("  • Feature'ları iyileştiriyor")
    print("  • Parametreleri optimize ediyor")
    print("  • Sürekli evrimleşiyor")
    print("\n💡 Model kullanıma hazır!")
    
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()






