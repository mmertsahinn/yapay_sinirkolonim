"""
Hashtag Scraper - Twitter'dan lig hashtag'lerini çeker ve hype analizi yapar
snscrape kullanarak: https://github.com/JustAnotherArchivist/snscrape
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import logging
import subprocess
import json
import re
from collections import defaultdict

from src.db.connection import get_session
from src.db.repositories import LeagueRepository, TeamRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HashtagScraper:
    """
    PRD: Hashtag çekici ve hype analizi
    - Lig hashtag'lerini çeker
    - Maç tarihinde hype ölçer
    - Hangi takım daha çok destekleniyor analiz eder
    """
    
    # Lig-Hashtag Mapping
    LEAGUE_HASHTAGS = {
        "Premier League": ["#PremierLeague", "#EPL", "#PL"],
        "La Liga": ["#LaLiga", "#LaLigaSantander"],
        "Serie A": ["#SerieA", "#SerieATIM"],
        "Bundesliga": ["#Bundesliga"],
        "Ligue 1": ["#Ligue1", "#Ligue1UberEats"],
        "Liga Portugal": ["#LigaPortugal", "#PrimeiraLiga"],
        "Süper Lig": ["#SüperLig", "#SuperLig", "#TSL"],
    }
    
    # Takım isimleri için alternatif hashtag'ler (önemli takımlar)
    TEAM_HASHTAGS = {
        # Premier League
        "Manchester United": ["#MUFC", "#ManUnited"],
        "Manchester City": ["#MCFC", "#ManCity"],
        "Liverpool": ["#LFC", "#YNWA"],
        "Chelsea": ["#CFC", "#ChelseaFC"],
        "Arsenal": ["#AFC", "#Arsenal"],
        "Tottenham": ["#THFC", "#Spurs"],
        # La Liga
        "Real Madrid": ["#RealMadrid", "#HalaMadrid"],
        "Barcelona": ["#FCBarcelona", "#Barca"],
        "Atletico Madrid": ["#Atleti", "#Atletico"],
        # Serie A
        "Juventus": ["#Juve", "#ForzaJuve"],
        "AC Milan": ["#ACMilan", "#ForzaMilan"],
        "Inter": ["#Inter", "#ForzaInter"],
        # Bundesliga
        "Bayern Munich": ["#FCBayern", "#MiaSanMia"],
        "Borussia Dortmund": ["#BVB", "#HejaBVB"],
        # Süper Lig
        "Galatasaray": ["#Galatasaray", "#GS"],
        "Fenerbahçe": ["#Fenerbahce", "#FB"],
        "Beşiktaş": ["#Besiktas", "#BJK"],
    }
    
    def __init__(self):
        self.session = None
    
    def get_league_hashtags(self, league_name: str) -> List[str]:
        """Lig için hashtag listesi döndür"""
        return self.LEAGUE_HASHTAGS.get(league_name, [f"#{league_name.replace(' ', '')}"])
    
    def get_team_hashtags(self, team_name: str) -> List[str]:
        """Takım için hashtag listesi döndür"""
        return self.TEAM_HASHTAGS.get(team_name, [f"#{team_name.replace(' ', '')}"])
    
    def scrape_hashtag(
        self,
        hashtag: str,
        date_from: datetime,
        date_to: datetime,
        max_results: int = 300  # 300 tweet çek
    ) -> List[Dict[str, Any]]:
        """
        snscrape kullanarak hashtag'den tweet'leri çeker (library olarak)
        
        Args:
            hashtag: Hashtag (örn: #PremierLeague)
            date_from: Başlangıç tarihi
            date_to: Bitiş tarihi
            max_results: Maksimum sonuç sayısı
            
        Returns:
            Tweet listesi (JSON formatında)
        """
        try:
            # Önce twscrape dene (snscrape'in modern alternatifi)
            try:
                import asyncio
                from twscrape import API, gather
                
                logger.info(f"📱 twscrape ile hashtag çekiliyor: {hashtag} (300 tweet)")
                
                async def scrape_with_twscrape():
                    api = API()
                    await api.pool.add_account("dummy", "dummy", "dummy", "dummy")  # Dummy account
                    await api.pool.login_all()
                    
                    hashtag_clean = hashtag.replace("#", "")
                    query = f"#{hashtag_clean} since:{date_from.strftime('%Y-%m-%d')} until:{date_to.strftime('%Y-%m-%d')}"
                    
                    tweets = []
                    async for tweet in api.search_tweet(query, limit=max_results):
                        try:
                            tweet_dict = {
                                'id': tweet.id,
                                'date': tweet.date.isoformat() if tweet.date else None,
                                'rawContent': tweet.rawContent if hasattr(tweet, 'rawContent') else (tweet.content if hasattr(tweet, 'content') else ''),
                                'user': tweet.user.username if tweet.user else None,
                                'likeCount': tweet.likeCount if hasattr(tweet, 'likeCount') else 0,
                                'retweetCount': tweet.retweetCount if hasattr(tweet, 'retweetCount') else 0,
                            }
                            
                            if tweet.date:
                                tweet_datetime = tweet.date.replace(tzinfo=None) if tweet.date.tzinfo else tweet.date
                                if date_from <= tweet_datetime <= date_to:
                                    tweets.append(tweet_dict)
                        except:
                            continue
                    return tweets
                
                # Async çalıştır
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tweets = loop.run_until_complete(scrape_with_twscrape())
                loop.close()
                
                if tweets:
                    logger.info(f"✅ twscrape ile {len(tweets)} tweet çekildi: {hashtag}")
                    return tweets
            except Exception as twscrape_error:
                logger.debug(f"twscrape hatası: {twscrape_error}, snscrape deneniyor...")
            
            # Fallback: snscrape'i library olarak kullan
            import snscrape.modules.twitter as sntwitter
            
            logger.info(f"📱 snscrape ile hashtag çekiliyor: {hashtag} (300 tweet)")
            
            # Hashtag query oluştur
            hashtag_clean = hashtag.replace("#", "")
            query = f"#{hashtag_clean} since:{date_from.strftime('%Y-%m-%d')} until:{date_to.strftime('%Y-%m-%d')}"
            
            tweets = []
            tweet_count = 0
            
            # Tweet'leri çek (300 tweet)
            logger.info(f"🔄 Tweet'ler çekiliyor: {hashtag}...")
            for i, tweet in enumerate(sntwitter.TwitterHashtagScraper(query).get_items()):
                if tweet_count >= max_results:
                    break
                
                try:
                    # Tweet'i dict formatına çevir
                    tweet_dict = {
                        'id': tweet.id,
                        'date': tweet.date.isoformat() if tweet.date else None,
                        'rawContent': tweet.rawContent if hasattr(tweet, 'rawContent') else (tweet.content if hasattr(tweet, 'content') else ''),
                        'user': tweet.user.username if tweet.user else None,
                        'likeCount': tweet.likeCount if hasattr(tweet, 'likeCount') else 0,
                        'retweetCount': tweet.retweetCount if hasattr(tweet, 'retweetCount') else 0,
                    }
                    
                    # Tarih filtresi kontrolü
                    if tweet.date:
                        tweet_datetime = tweet.date.replace(tzinfo=None) if tweet.date.tzinfo else tweet.date
                        if date_from <= tweet_datetime <= date_to:
                            tweets.append(tweet_dict)
                            tweet_count += 1
                    else:
                        tweets.append(tweet_dict)
                        tweet_count += 1
                    
                    # Her 50 tweet'te bir progress göster
                    if tweet_count % 50 == 0:
                        logger.info(f"   📊 {tweet_count}/{max_results} tweet çekildi...")
                        
                except Exception as e:
                    logger.debug(f"Tweet parse hatası: {e}")
                    continue
            
            logger.info(f"✅ {len(tweets)} tweet çekildi: {hashtag}")
            return tweets
            
        except (ImportError, AttributeError) as e:
            # Python 3.13 uyumluluk sorunu - Alternatif yöntem dene
            logger.warning(f"snscrape/twscrape import hatası: {e}")
            logger.warning(f"Alternatif yöntem deneniyor: {hashtag}")
            return self._scrape_with_alternative(hashtag, date_from, date_to, max_results)
        except Exception as e:
            logger.error(f"Hashtag scraping hatası: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Fallback: Alternatif yöntem
            return self._scrape_with_alternative(hashtag, date_from, date_to, max_results)
    
    def _scrape_with_alternative(
        self,
        hashtag: str,
        date_from: datetime,
        date_to: datetime,
        max_results: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Alternatif yöntemlerle gerçek tweet çek
        """
        try:
            # Yöntem 1: Twitter API v2 (eğer bearer token varsa)
            import os
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
            
            if bearer_token:
                from src.ingestion.twitter_scraper_alt import TwitterScraperAlternative
                alt_scraper = TwitterScraperAlternative()
                tweets = alt_scraper.scrape_hashtag_via_twitter_api_v2(
                    hashtag, date_from, date_to, max_results, bearer_token
                )
                if tweets:
                    logger.info(f"✅ Twitter API v2 ile {len(tweets)} tweet çekildi: {hashtag}")
                    return tweets
            
            # Yöntem 2: Nitter (açık kaynak Twitter alternatifi)
            try:
                from src.ingestion.twitter_scraper_alt import TwitterScraperAlternative
                alt_scraper = TwitterScraperAlternative()
                tweets = alt_scraper.scrape_hashtag_via_nitter(
                    hashtag, date_from, date_to, max_results
                )
                if tweets:
                    logger.info(f"✅ Nitter ile {len(tweets)} tweet çekildi: {hashtag}")
                    return tweets
            except Exception as e:
                logger.debug(f"Nitter hatası: {e}")
            
            # Yöntem 3: Python 3.12'ye geçiş önerisi
            logger.error(f"❌ Gerçek tweet çekilemedi!")
            logger.error(f"📥 Python 3.12 kur ve snscrape'i yükle:")
            logger.error(f"   1. py -3.12 -m venv venv312")
            logger.error(f"   2. .\\venv312\\Scripts\\activate")
            logger.error(f"   3. pip install snscrape")
            logger.error(f"   4. python tum_maclar_hype_cek.py")
            logger.warning(f"⚠️  Şimdilik boş liste döndürülüyor (default hype değerleri kullanılacak)")
            return []
            
        except Exception as e:
            logger.error(f"Alternatif scraping hatası: {e}")
            return []
    
    def analyze_hype(
        self,
        tweets: List[Dict[str, Any]],
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        Tweet'lerden hype analizi yapar
        Hangi takım daha çok destekleniyor?
        
        Returns:
            {
                "home_support": float,  # 0-1 arası
                "away_support": float,  # 0-1 arası
                "total_tweets": int,
                "home_mentions": int,
                "away_mentions": int,
                "sentiment_score": float  # -1 (away) to +1 (home)
            }
        """
        if not tweets:
            return {
                "home_support": 0.5,
                "away_support": 0.5,
                "total_tweets": 0,
                "home_mentions": 0,
                "away_mentions": 0,
                "sentiment_score": 0.0
            }
        
        home_mentions = 0
        away_mentions = 0
        
        # Takım isimlerini normalize et (küçük harf, boşlukları kaldır)
        home_normalized = self._normalize_team_name(home_team)
        away_normalized = self._normalize_team_name(away_team)
        
        # Takım hashtag'leri
        home_hashtags = [h.lower().replace("#", "") for h in self.get_team_hashtags(home_team)]
        away_hashtags = [h.lower().replace("#", "") for h in self.get_team_hashtags(away_team)]
        
        for tweet in tweets:
            content = tweet.get('rawContent', '').lower()
            
            # Takım isimleri ve hashtag'leri kontrol et
            if any(name in content for name in [home_normalized] + home_hashtags):
                home_mentions += 1
            
            if any(name in content for name in [away_normalized] + away_hashtags):
                away_mentions += 1
        
        total_mentions = home_mentions + away_mentions
        
        if total_mentions == 0:
            return {
                "home_support": 0.5,
                "away_support": 0.5,
                "total_tweets": len(tweets),
                "home_mentions": 0,
                "away_mentions": 0,
                "sentiment_score": 0.0
            }
        
        home_support = home_mentions / total_mentions
        away_support = away_mentions / total_mentions
        
        # Sentiment score: -1 (tam away) to +1 (tam home)
        sentiment_score = home_support - away_support
        
        return {
            "home_support": home_support,
            "away_support": away_support,
            "total_tweets": len(tweets),
            "home_mentions": home_mentions,
            "away_mentions": away_mentions,
            "sentiment_score": sentiment_score
        }
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Takım ismini normalize et (arama için)"""
        # Küçük harf, boşlukları kaldır, özel karakterleri temizle
        normalized = team_name.lower().replace(" ", "").replace("fc", "").replace("cf", "")
        return normalized
    
    def get_match_hype(
        self,
        league_name: str,
        home_team: str,
        away_team: str,
        match_date: datetime,
        days_before: int = 1
    ) -> Dict[str, Any]:
        """
        Maç için hype analizi yapar
        
        Args:
            league_name: Lig adı
            home_team: Ev sahibi takım
            away_team: Deplasman takımı
            match_date: Maç tarihi
            days_before: Maçtan kaç gün öncesine bakılacak
            
        Returns:
            Hype analizi sonuçları
        """
        # Tarih aralığı
        date_from = match_date - timedelta(days=days_before)
        date_to = match_date
        
        # Lig hashtag'lerini al
        league_hashtags = self.get_league_hashtags(league_name)
        
        # Tüm tweet'leri topla
        all_tweets = []
        for hashtag in league_hashtags:
            tweets = self.scrape_hashtag(hashtag, date_from, date_to, max_results=300)  # 300 tweet
            all_tweets.extend(tweets)
        
        # Takım hashtag'lerini de ekle
        home_hashtags = self.get_team_hashtags(home_team)
        away_hashtags = self.get_team_hashtags(away_team)
        
        for hashtag in home_hashtags + away_hashtags:
            tweets = self.scrape_hashtag(hashtag, date_from, date_to, max_results=300)  # 300 tweet
            all_tweets.extend(tweets)
        
        # Duplicate'leri kaldır (tweet ID'ye göre)
        seen_ids = set()
        unique_tweets = []
        for tweet in all_tweets:
            tweet_id = tweet.get('id')
            if tweet_id and tweet_id not in seen_ids:
                seen_ids.add(tweet_id)
                unique_tweets.append(tweet)
        
        # Hype analizi
        hype_result = self.analyze_hype(unique_tweets, home_team, away_team)
        
        logger.info(f"📊 Hype analizi: {home_team} vs {away_team}")
        logger.info(f"   Home support: {hype_result['home_support']:.2%}")
        logger.info(f"   Away support: {hype_result['away_support']:.2%}")
        logger.info(f"   Total tweets: {hype_result['total_tweets']}")
        
        return hype_result
    
    def get_match_hype_cached(
        self,
        match_id: Optional[int],
        league_name: str,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Dict[str, Any]:
        """
        Cache'li hype analizi (veritabanından önce kontrol eder)
        Twitter çalışmazsa alternatif yöntemleri kullanır
        """
        session = get_session()
        try:
            # Önce Twitter'ı dene
            try:
                twitter_result = self.get_match_hype(
                    league_name,
                    home_team,
                    away_team,
                    match_date,
                    days_before=1
                )
                
                # Eğer tweet çekildiyse Twitter sonucunu kullan
                if twitter_result.get("total_tweets", 0) > 0:
                    logger.info(f"✅ Twitter'dan {twitter_result['total_tweets']} tweet çekildi")
                    return twitter_result
            except Exception as e:
                logger.debug(f"Twitter hype hatası: {e}")
            
            # Twitter çalışmadıysa alternatif yöntemleri kullan
            logger.info("🔄 Twitter çalışmadı, alternatif yöntemler deneniyor...")
            try:
                from src.ingestion.alternative_hype_scraper import AlternativeHypeScraper
                alt_scraper = AlternativeHypeScraper()
                alt_result = alt_scraper.get_match_hype(
                    league_name,
                    home_team,
                    away_team,
                    match_date
                )
                
                # Alternatif scraper formatını Twitter formatına çevir
                return {
                    "home_support": alt_result.get("home_support", 0.5),
                    "away_support": alt_result.get("away_support", 0.5),
                    "total_tweets": alt_result.get("total_mentions", 0),
                    "home_mentions": int(alt_result.get("total_mentions", 0) * alt_result.get("home_support", 0.5)),
                    "away_mentions": int(alt_result.get("total_mentions", 0) * alt_result.get("away_support", 0.5)),
                    "sentiment_score": alt_result.get("sentiment_score", 0.0),
                    "sources": alt_result.get("sources", [])
                }
            except Exception as e:
                logger.warning(f"Alternatif hype hatası: {e}")
            
            # Hiçbiri çalışmadıysa default değerler
            logger.warning("⚠️  Hiçbir hype kaynağı çalışmadı, default değerler kullanılıyor")
            return {
                "home_support": 0.5,
                "away_support": 0.5,
                "total_tweets": 0,
                "home_mentions": 0,
                "away_mentions": 0,
                "sentiment_score": 0.0
            }
        finally:
            session.close()

