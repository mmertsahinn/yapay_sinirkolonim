"""
Tarihsel veri yükle - Kolay başlatma scripti
"""
import sys
from pathlib import Path

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.ingestion.historical_loader import HistoricalLoader

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", help="Sezonlar (ornek: 2021 2022 2023)")
    args = parser.parse_args()
    
    print("Tarihsel veri yukleniyor...")
    print("=" * 60)
    loader = HistoricalLoader()
    
    if args.seasons:
        print(f"Sezonlar: {args.seasons}")
        loader.load_all_historical_data(seasons=args.seasons)
    else:
        print("Varsayilan sezonlar yukleniyor...")
        loader.load_all_historical_data()
    
    print("=" * 60)
    print("[OK] Veri yukleme tamamlandi!")

