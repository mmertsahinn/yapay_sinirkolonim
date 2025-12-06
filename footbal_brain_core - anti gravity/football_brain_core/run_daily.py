"""
Günlük güncelleme - Kolay başlatma scripti
"""
import sys
from pathlib import Path

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from football_brain_core.src.ingestion.daily_jobs import DailyJobs

if __name__ == "__main__":
    print("Gunluk guncelleme baslatiliyor...")
    jobs = DailyJobs()
    jobs.run_daily_update()
    print("✅ Gunluk guncelleme tamamlandi!")







