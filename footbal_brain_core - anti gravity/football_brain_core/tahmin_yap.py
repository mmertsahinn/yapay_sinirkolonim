"""
Basit tahmin scripti - Eğitilmiş model ile tahmin yap
"""
import sys
from pathlib import Path
from datetime import date, timedelta
import torch

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.config import Config
from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.features.feature_builder import FeatureBuilder
from football_brain_core.src.features.market_targets import MarketType, MARKET_OUTCOMES
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, LeagueRepository

def load_model(model_path: str, market_types: list, input_size: int):
    """Modeli yükle"""
    model = MultiTaskModel(
        input_size=input_size,
        market_types=market_types,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_matches(days_ahead: int = 7, model_path: str = "model_prd_v1.0.pth"):
    """Gelecek maçlar için tahmin yap"""
    
    print("=" * 80)
    print("TAHMIN YAPILIYOR")
    print("=" * 80)
    
    # Market tipleri
    market_types = [
        MarketType.MATCH_RESULT,
        MarketType.BTTS,
        MarketType.OVER_UNDER_25,
        MarketType.GOAL_RANGE,
        MarketType.CORRECT_SCORE,
        MarketType.DOUBLE_CHANCE,
    ]
    
    # Model yükle
    print(f"\n📦 Model yükleniyor: {model_path}")
    config = Config()
    
    # Feature size'ı almak için bir örnek feature oluştur
    feature_builder = FeatureBuilder()
    session = get_session()
    
    try:
        # Örnek bir maç al
        sample_match = MatchRepository.get_by_id(session, 1)
        if sample_match:
            sample_features = feature_builder.build_match_features(
                sample_match.home_team_id,
                sample_match.away_team_id,
                sample_match.match_date,
                sample_match.league_id,
                session
            )
            input_size = len(sample_features)
        else:
            # Varsayılan feature size
            input_size = 50  # FeatureBuilder'dan gelen feature sayısı
        
        model = load_model(model_path, market_types, input_size)
        print("✅ Model yüklendi!")
        
        # Gelecek maçları bul
        date_from = date.today()
        date_to = date_from + timedelta(days=days_ahead)
        
        print(f"\n📅 Tarih aralığı: {date_from} - {date_to}")
        
        matches = MatchRepository.get_by_date_range(session, date_from, date_to)
        # Sadece skoru olmayan maçlar (henüz oynanmamış)
        matches = [m for m in matches if m.home_score is None or m.away_score is None]
        
        print(f"🎯 {len(matches)} maç bulundu\n")
        
        if not matches:
            print("⚠️  Tahmin edilecek maç bulunamadı!")
            return
        
        # Her maç için tahmin yap
        for i, match in enumerate(matches[:10], 1):  # İlk 10 maç
            try:
                print(f"\n[{i}/{min(10, len(matches))}] {match.home_team.name} vs {match.away_team.name}")
                print(f"   📅 {match.match_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"   🏆 {match.league.name}")
                
                # Feature oluştur
                features = feature_builder.build_match_features(
                    match.home_team_id,
                    match.away_team_id,
                    match.match_date,
                    match.league_id,
                    session
                )
                
                # Tahmin yap
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(features_tensor)
                    probas = model.predict_proba(features_tensor)
                    predictions = model.predict(features_tensor)
                
                # Sonuçları göster
                print("   📊 Tahminler:")
                for market_type in market_types:
                    pred_idx = predictions[market_type.value].cpu().numpy()[0]
                    outcome = MARKET_OUTCOMES[market_type][pred_idx]
                    probability = probas[market_type.value].cpu().numpy()[0][pred_idx]
                    
                    print(f"      • {market_type.value}: {outcome} ({probability*100:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Hata: {e}")
                continue
        
        print("\n" + "=" * 80)
        print("✅ Tahminler tamamlandı!")
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print("💡 Önce model eğitmelisiniz: python quick_test.py")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    # Model yolu
    model_path = "model_prd_v1.0.pth"
    
    # Kaç gün ileriye tahmin yapılacak
    days_ahead = 7
    
    predict_matches(days_ahead=days_ahead, model_path=model_path)






