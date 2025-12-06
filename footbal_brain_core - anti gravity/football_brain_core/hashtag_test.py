"""
Hashtag Scraper Test Script
"""
import sys
from pathlib import Path
from datetime import datetime

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.ingestion.hashtag_scraper import HashtagScraper

print("=" * 80)
print("HASHTAG SCRAPER TEST")
print("=" * 80)
print()

try:
    scraper = HashtagScraper()
    
    # Test: Premier League maçı
    print("📱 Test: Premier League - Manchester United vs Liverpool")
    print("-" * 80)
    
    hype = scraper.get_match_hype(
        league_name="Premier League",
        home_team="Manchester United",
        away_team="Liverpool",
        match_date=datetime(2024, 12, 1),
        days_before=1
    )
    
    print(f"\n📊 Hype Analizi Sonuçları:")
    print(f"   Home Support: {hype['home_support']:.2%}")
    print(f"   Away Support: {hype['away_support']:.2%}")
    print(f"   Sentiment Score: {hype['sentiment_score']:.2f}")
    print(f"   Total Tweets: {hype['total_tweets']}")
    print(f"   Home Mentions: {hype['home_mentions']}")
    print(f"   Away Mentions: {hype['away_mentions']}")
    
    print("\n✅ Test tamamlandı!")
    
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()






