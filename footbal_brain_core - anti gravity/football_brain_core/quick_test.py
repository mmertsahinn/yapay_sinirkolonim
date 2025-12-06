"""
PRD'ye uygun model eğitimi - 3-5 sezon veri ile
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

from football_brain_core.src.config import Config
from football_brain_core.src.models.train_offline import OfflineTrainer
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import LeagueRepository
import torch

print("🚀 PRD'ye uygun model eğitimi başlıyor...")
print("📊 3-5 sezon veri ile çoklu market tahminleri\n")

config = Config()
config.MODEL_CONFIG.epochs = 50  # PRD'ye uygun tam eğitim
config.MODEL_CONFIG.batch_size = 32

# PRD'de belirtilen çoklu marketler
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
print(f"📦 Batch size: {config.MODEL_CONFIG.batch_size}\n")

trainer = OfflineTrainer(market_types, config, model_config={
    "hidden_size": 128,  # PRD'ye uygun tam model
    "num_layers": 2,     # Derin mimari
    "dropout": 0.3
})

session = get_session()
try:
    league_ids = [
        LeagueRepository.get_or_create(session, league.name).id
        for league in config.TARGET_LEAGUES
    ]
    print(f"🏆 Ligler: {len(league_ids)} lig\n")
finally:
    session.close()

print("📚 Veri hazırlanıyor...")

# Veritabanındaki en eski yıldan 2022'ye kadar eğitim yıllarını otomatik bul
session = get_session()
try:
    from sqlalchemy import func, extract
    from football_brain_core.src.db.schema import Match
    
    # Veritabanındaki tüm yılları bul
    years_query = session.query(
        extract('year', Match.match_date).label('year')
    ).distinct().order_by('year').all()
    
    available_years = sorted([int(y[0]) for y in years_query])
    
    # 2022'ye kadar olan yılları al (eğitim için)
    train_years = [y for y in available_years if y <= 2022]
    
    if not train_years:
        print("⚠️  UYARI: 2022'ye kadar veri bulunamadı, mevcut yılları kullanıyoruz")
        train_years = available_years[:3] if len(available_years) >= 3 else available_years
    
    print(f"📅 Eğitim yılları (otomatik): {train_years[0]} - {train_years[-1]}")
    print(f"📅 Validation: {train_years[-1]} (son yıl)\n")
    
finally:
    session.close()

try:
    # En eski yıldan 2022'ye kadar eğitim (ders alsınlar)
    # Validation için son yıl kullan
    model = trainer.train(train_years, [train_years[-1]], league_ids)
    print("\n✅ Model eğitimi tamamlandı!")
    print("💾 Model kaydediliyor...")
    
    # Modeli kaydet
    torch.save(model.state_dict(), "model_prd_v1.0.pth")
    print("✅ Model 'model_prd_v1.0.pth' olarak kaydedildi!")
    print("\n📝 Sonraki adımlar:")
    print("   1. Model versiyonunu kaydet: ModelVersionRepository ile")
    print("   2. Tahmin yap: python predict_with_explanations.py")
    print("   3. Excel raporu: python -m football_brain_core.src.cli.main report-daily")
    
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Çözüm önerileri:")
    print("   1. Veri yüklendiğinden emin ol: python -m football_brain_core.src.cli.main load-historical --seasons 2023")
    print("   2. Veritabanında maç olduğunu kontrol et")
    print("   3. API key'in doğru ayarlandığından emin ol")

