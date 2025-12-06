"""
Alternatif Twitter Scraper - Gerçek tweet çekmek için
snscrape Python 3.13 ile uyumlu olmadığı için alternatif yöntemler
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import requests
import json
import time

logger = logging.getLogger(__name__)


class TwitterScraperAlternative:
    """
    Alternatif Twitter scraper - Gerçek tweet çekmek için
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_hashtag_via_nitter(
        self,
        hashtag: str,
        date_from: datetime,
        date_to: datetime,
        max_results: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Nitter instance kullanarak tweet çek (snscrape alternatifi)
        Nitter: Twitter'ın açık kaynak alternatifi, API key gerektirmez
        """
        try:
            # Nitter instance'ları (public)
            nitter_instances = [
                "https://nitter.net",
                "https://nitter.it",
                "https://nitter.pussthecat.org",
            ]
            
            hashtag_clean = hashtag.replace("#", "")
            tweets = []
            
            for instance in nitter_instances:
                try:
                    # Nitter RSS feed kullan
                    url = f"{instance}/search/rss?f=tweets&q={hashtag_clean}"
                    
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        # RSS parse et (basit)
                        # Not: Nitter RSS formatı değişebilir
                        content = response.text
                        # Basit parsing (gerçek implementasyon daha karmaşık olmalı)
                        # Şimdilik placeholder
                        logger.info(f"Nitter'dan tweet çekiliyor: {hashtag}")
                        break
                except Exception as e:
                    logger.debug(f"Nitter instance {instance} hatası: {e}")
                    continue
            
            return tweets
            
        except Exception as e:
            logger.error(f"Nitter scraping hatası: {e}")
            return []
    
    def scrape_hashtag_via_twitter_api_v2(
        self,
        hashtag: str,
        date_from: datetime,
        date_to: datetime,
        max_results: int = 300,
        bearer_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Twitter API v2 kullanarak tweet çek (API key gerektirir)
        """
        if not bearer_token:
            logger.warning("Twitter API v2 bearer token yok, atlanıyor")
            return []
        
        try:
            hashtag_clean = hashtag.replace("#", "")
            
            # Twitter API v2 endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            params = {
                "query": f"#{hashtag_clean} -is:retweet",
                "max_results": min(max_results, 100),  # API limit
                "start_time": date_from.isoformat() + "Z",
                "end_time": date_to.isoformat() + "Z",
                "tweet.fields": "created_at,public_metrics,text"
            }
            
            headers = {
                "Authorization": f"Bearer {bearer_token}"
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tweets = []
                
                for tweet in data.get("data", []):
                    tweets.append({
                        'id': tweet.get('id'),
                        'date': tweet.get('created_at'),
                        'rawContent': tweet.get('text', ''),
                        'likeCount': tweet.get('public_metrics', {}).get('like_count', 0),
                        'retweetCount': tweet.get('public_metrics', {}).get('retweet_count', 0),
                    })
                
                logger.info(f"✅ Twitter API v2'den {len(tweets)} tweet çekildi: {hashtag}")
                return tweets
            else:
                logger.error(f"Twitter API v2 hatası: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Twitter API v2 scraping hatası: {e}")
            return []


def get_twitter_scraper() -> TwitterScraperAlternative:
    """Twitter scraper instance döndür"""
    return TwitterScraperAlternative()






