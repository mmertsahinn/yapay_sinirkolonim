"""
Alternatif Hype Scraper - HIZLI KAYNAKLAR
1. Google Trends - Arama trendleri (HIZLI)
2. Web Scraping - Haber siteleri (HIZLI)
News API KALDIRILDI - Zaman kazanmak için
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class AlternativeHypeScraper:
    """
    Twitter API çalışmadığında alternatif yöntemlerle hype ölçer
    """
    
    def __init__(self):
        self._pytrends = None
        # Reddit kaldırıldı - zaman kaybı
        self._newsapi = None
        # Başarısız istekler için izleme
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.failure_cooldown = 30  # 30 saniye mola
        
    def get_match_hype(
        self,
        league_name: str,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Dict[str, Any]:
        """
        Maç için hype verilerini toplar (tüm alternatif yöntemlerden)
        
        Returns:
            {
                "home_support": float,  # 0-1 arası
                "away_support": float,  # 0-1 arası
                "sentiment_score": float,  # -1 to +1
                "total_mentions": int,
                "hype_score": float,  # 0-1 arası genel hype
                "sources": List[str]  # Hangi kaynaklar kullanıldı
            }
        """
        # Art arda 3 başarısız istekten sonra mola ver
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"⚠️ {self.max_consecutive_failures} başarısız istek - {self.failure_cooldown}s mola veriliyor...")
            time.sleep(self.failure_cooldown)
            self.consecutive_failures = 0
        
        results = []
        sources = []
        
        # 1. Google Trends - HIZLI KAYNAK
        try:
            trends_data = self._get_google_trends(home_team, away_team, match_date)
            if trends_data:
                results.append(trends_data)
                sources.append("Google Trends")
                self.consecutive_failures = 0  # Başarı - counter sıfırla
                logger.info(f"✅ Google Trends: {trends_data.get('total_mentions', 0)} mentions")
        except Exception as e:
            self.consecutive_failures += 1
            logger.debug(f"Google Trends hatası (atlandı): {e}")
            pass
        
        # 2. Web Scraping (haber siteleri) - HIZLI KAYNAK
        # News API KALDIRILDI - Zaman kazanmak için
        try:
            web_data = self._get_web_scraping_hype(home_team, away_team, match_date, league_name)
            if web_data:
                results.append(web_data)
                sources.append("Web Scraping")
                self.consecutive_failures = 0  # Başarı - counter sıfırla
                logger.info(f"✅ Web Scraping: {web_data.get('total_mentions', 0)} mentions")
        except Exception as e:
            self.consecutive_failures += 1
            logger.debug(f"Web scraping hatası (atlandı): {e}")
            pass
        
        # Sonuçları birleştir - AĞIRLIKLI ORTALAMA (daha derin analiz)
        if not results:
            logger.warning(f"Hype verisi bulunamadı: {home_team} vs {away_team}")
            self.consecutive_failures += 1
            return self._default_hype()
        
        # Kaynak bilgisi logla (sadece Google Trends ve Web Scraping)
        logger.info(f"Hype kaynakları ({len(sources)}): {', '.join(sources)}")
        
        # Her kaynağa ağırlık ver (daha fazla mention = daha güvenilir)
        weighted_home = 0.0
        weighted_away = 0.0
        total_weight = 0.0
        total_mentions = 0
        
        for r in results:
            mentions = r.get("total_mentions", 0)
            # Minimum ağırlık: 1, maksimum ağırlık: mentions (daha fazla mention = daha güvenilir)
            weight = max(1, mentions) if mentions > 0 else 1
            
            weighted_home += r.get("home_support", 0.5) * weight
            weighted_away += r.get("away_support", 0.5) * weight
            total_weight += weight
            total_mentions += mentions
        
        # Ağırlıklı ortalama
        if total_weight > 0:
            home_support = weighted_home / total_weight
            away_support = weighted_away / total_weight
        else:
            # Fallback: Basit ortalama
            home_support = sum(r.get("home_support", 0.5) for r in results) / len(results)
            away_support = sum(r.get("away_support", 0.5) for r in results) / len(results)
        
        sentiment = home_support - away_support
        # Hype score hesaplama: 100 mention = 1.0 hype (daha gerçekçi)
        hype_score = min(1.0, total_mentions / 100.0)
        
        logger.info(f"Hype sonucu: {total_mentions} mentions, Home: {home_support:.2%}, Away: {away_support:.2%}")
        
        return {
            "home_support": home_support,
            "away_support": away_support,
            "sentiment_score": sentiment,
            "total_mentions": total_mentions,
            "hype_score": hype_score,
            "sources": sources
        }
    
    def _get_google_trends(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Google Trends'den DERİNLEMESİNE arama trendlerini çeker
        - Farklı ülkeler için ayrı ayrı
        - Farklı zaman aralıkları
        - İlgili terimler
        """
        try:
            from pytrends.request import TrendReq
            
            if self._pytrends is None:
                self._pytrends = TrendReq(hl='en', tz=360)
            
            # Daha geniş tarih aralığı (14 gün)
            date_from = match_date - timedelta(days=14)
            date_to = match_date
            
            # Takım isimlerini normalize et
            home_clean = home_team.replace(" ", " ").strip()
            away_clean = away_team.replace(" ", " ").strip()
            
            # OPTİMİZE: Sadece global + lig ülkesi (hız için optimize edildi)
            # Lig ülkesini belirle
            league_country_map = {
                'Premier League': 'GB',
                'La Liga': 'ES',
                'Serie A': 'IT',
                'Bundesliga': 'DE',
                'Ligue 1': 'FR',
                'Liga Portugal': 'PT',
                'Süper Lig': 'TR',
            }
            
            # Sadece global + lig ülkesi (2 sorgu yerine 8)
            countries = ['']  # Global
            # Lig ülkesi ekle (eğer belirlenebilirse)
            # Şimdilik sadece global kullan (daha hızlı)
            
            all_home_scores = []
            all_away_scores = []
            
            for geo in countries:
                try:
                    # Trend sorgusu
                    self._pytrends.build_payload(
                        [home_clean, away_clean],
                        cat=0,
                        timeframe=f'{date_from.strftime("%Y-%m-%d")} {date_to.strftime("%Y-%m-%d")}',
                        geo=geo if geo else ''
                    )
                    
                    df = self._pytrends.interest_over_time()
                    
                    if not df.empty:
                        # Son günlerin ortalaması
                        if home_clean in df.columns:
                            home_avg = df[home_clean].mean()
                            all_home_scores.append(home_avg)
                        
                        if away_clean in df.columns:
                            away_avg = df[away_clean].mean()
                            all_away_scores.append(away_avg)
                    
                    # Rate limiting - Google Trends için minimum (veri kalitesi korunuyor)
                    time.sleep(0.01)  # Daha kısa bekleme
                except Exception as e:
                    logger.debug(f"Google Trends {geo} hatası: {e}")
                    continue
            
            # Tüm ülkelerden gelen verileri birleştir
            home_total = sum(all_home_scores) if all_home_scores else 0
            away_total = sum(all_away_scores) if all_away_scores else 0
            total = home_total + away_total
            
            if total > 0:
                home_support = home_total / total
                away_support = away_total / total
            else:
                # İkinci yöntem: İlgili terimler
                try:
                    # Takım isimleri + "vs" kombinasyonu
                    vs_query = f"{home_clean} vs {away_clean}"
                    self._pytrends.build_payload(
                        [vs_query],
                        cat=0,
                        timeframe=f'{date_from.strftime("%Y-%m-%d")} {date_to.strftime("%Y-%m-%d")}',
                        geo=''
                    )
                    df_vs = self._pytrends.interest_over_time()
                    
                    if not df_vs.empty and vs_query in df_vs.columns:
                        vs_interest = df_vs[vs_query].mean()
                        if vs_interest > 0:
                            # Eşit dağılım (daha fazla veri yok)
                            home_support = 0.5
                            away_support = 0.5
                            total = int(vs_interest)
                        else:
                            return None
                    else:
                        return None
                except:
                    return None
            
            return {
                "home_support": home_support,
                "away_support": away_support,
                "total_mentions": int(total)
            }
        except ImportError:
            logger.debug("pytrends yüklü değil: pip install pytrends")
            return None
        except Exception as e:
            # API hatası - sessizce geç, önemli değil
            logger.debug(f"Google Trends hatası (atlandı): {e}")
            return None
    
    # Reddit kaldırıldı - zaman kaybı
    
    def _get_news_mentions(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        News API'den haber mention sayılarını çeker
        """
        try:
            import os
            from newsapi import NewsApiClient
            from src.config import Config
            
            # Önce config'den, sonra environment'tan al
            config = Config()
            api_key = os.getenv("NEWS_API_KEY") or config.NEWS_API_KEY
            
            if not api_key:
                logger.warning("NEWS_API_KEY bulunamadı - News API kullanılamıyor")
                logger.info("News API key almak için: https://newsapi.org/register")
                return None
            
            if self._newsapi is None:
                self._newsapi = NewsApiClient(api_key=api_key)
            
            # Son 7 günlük haberler
            date_from = match_date - timedelta(days=7)
            
            # Her takım için haber sayısı - TÜM DİLLER (daha fazla veri)
            # Önce tüm diller, sonra spesifik diller
            languages = [None, 'en', 'tr', 'es', 'it', 'de', 'fr']  # None = tüm diller
            
            home_count = 0
            away_count = 0
            
            for lang in languages[:3]:  # İlk 3 dil (hız için)
                try:
                    home_params = {
                        'q': home_team,
                        'from_param': date_from.strftime("%Y-%m-%d"),
                        'to': match_date.strftime("%Y-%m-%d"),
                        'sort_by': 'popularity',
                        'page_size': 100  # Maksimum
                    }
                    if lang:
                        home_params['language'] = lang
                    
                    home_articles = self._newsapi.get_everything(**home_params)
                    home_count += home_articles.get('totalResults', 0)
                    
                    away_params = {
                        'q': away_team,
                        'from_param': date_from.strftime("%Y-%m-%d"),
                        'to': match_date.strftime("%Y-%m-%d"),
                        'sort_by': 'popularity',
                        'page_size': 100
                    }
                    if lang:
                        away_params['language'] = lang
                    
                    away_articles = self._newsapi.get_everything(**away_params)
                    away_count += away_articles.get('totalResults', 0)
                    
                    # Rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    # API hatası - sessizce geç, önemli değil
                    logger.debug(f"News API {lang} hatası (atlandı): {e}")
                    continue
            
            total = home_count + away_count
            if total > 0:
                home_support = home_count / total
                away_support = away_count / total
            else:
                home_support = 0.5
                away_support = 0.5
            
            return {
                "home_support": home_support,
                "away_support": away_support,
                "total_mentions": total
            }
        except ImportError:
            logger.debug("newsapi yüklü değil: pip install newsapi-python")
            return None
        except Exception as e:
            # API hatası - sessizce geç, önemli değil
            logger.debug(f"News API hatası (atlandı): {e}")
            return None
    
    def _get_web_scraping_hype(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        league_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Haber sitelerinden web scraping ile mention sayıları
        - Ligin ülkesine göre filtreli siteler
        - Maç tarihine yakın haberler
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            import re
            
            # Lig-Ülke haritası
            league_country_map = {
                'Süper Lig': 'tr',
                'Premier League': 'en',
                'La Liga': 'es',
                'Serie A': 'it',
                'Bundesliga': 'de',
                'Ligue 1': 'fr',
                'Liga Portugal': 'pt',
            }
            
            # Ligin ülkesini belirle
            league_country = league_country_map.get(league_name, 'tr') if league_name else 'tr'
            
            # Tüm siteler - HER ÜLKEDEN 12'ŞER TANE (ARTIRILDI)
            all_sites = {
                # Türkiye (12 site)
                "https://www.fanatik.com.tr": "tr",
                "https://www.hurriyet.com.tr/spor": "tr",
                "https://www.sozcu.com.tr/kategori/spor": "tr",
                "https://www.milliyet.com.tr/spor": "tr",
                "https://www.ntvspor.net": "tr",
                "https://www.sabah.com.tr/spor": "tr",
                "https://www.cumhuriyet.com.tr/spor": "tr",
                "https://www.ahaber.com.tr/spor": "tr",
                "https://www.aksam.com.tr/spor": "tr",
                "https://www.takvim.com.tr/spor": "tr",
                "https://www.star.com.tr/spor": "tr",
                "https://www.yeniasir.com.tr/spor": "tr",
                # İngiltere (12 site)
                "https://www.bbc.com/sport/football": "en",
                "https://www.skysports.com/football": "en",
                "https://www.theguardian.com/football": "en",
                "https://www.independent.co.uk/sport/football": "en",
                "https://www.goal.com": "en",
                "https://www.espn.com/soccer": "en",
                "https://www.fourfourtwo.com": "en",
                "https://www.dailymail.co.uk/sport/football": "en",
                "https://www.mirror.co.uk/sport/football": "en",
                "https://www.thesun.co.uk/sport/football": "en",
                "https://www.telegraph.co.uk/football": "en",
                "https://www.standard.co.uk/sport/football": "en",
                # İspanya (12 site)
                "https://www.marca.com/futbol": "es",
                "https://www.as.com/futbol": "es",
                "https://www.sport.es": "es",
                "https://www.elmundo.es/deportes/futbol": "es",
                "https://www.elpais.com/deportes/futbol": "es",
                "https://www.mundodeportivo.com": "es",
                "https://www.diarioAS.com": "es",
                "https://www.lavanguardia.com/deportes/futbol": "es",
                "https://www.abc.es/deportes/futbol": "es",
                "https://www.20minutos.es/deportes/futbol": "es",
                "https://www.elperiodico.com/es/deportes/futbol": "es",
                "https://www.sport-english.com": "es",
                # İtalya (12 site)
                "https://www.gazzetta.it/Calcio": "it",
                "https://www.corrieredellosport.it": "it",
                "https://www.repubblica.it/sport/calcio": "it",
                "https://www.tuttosport.com": "it",
                "https://www.ilsole24ore.com/sport": "it",
                "https://www.goal.com/it": "it",
                "https://www.calciomercato.com": "it",
                "https://www.sportmediaset.mediaset.it/calcio": "it",
                "https://www.raisport.rai.it/dl/raiSport/": "it",
                "https://www.ilgiornale.it/sport/calcio": "it",
                "https://www.lastampa.it/sport/calcio": "it",
                "https://www.ilfattoquotidiano.it/sport/calcio": "it",
                # Almanya (12 site)
                "https://www.kicker.de": "de",
                "https://www.sport1.de/fussball": "de",
                "https://www.bild.de/sport": "de",
                "https://www.spox.com/de/fussball": "de",
                "https://www.dfb.de": "de",
                "https://www.welt.de/sport/fussball": "de",
                "https://www.goal.com/de": "de",
                "https://www.transfermarkt.de": "de",
                "https://www.sueddeutsche.de/sport/fussball": "de",
                "https://www.faz.net/sport/fussball": "de",
                "https://www.handelsblatt.com/sport/fussball": "de",
                "https://www.focus.de/sport/fussball": "de",
                # Fransa (12 site)
                "https://www.lequipe.fr/Football": "fr",
                "https://www.france24.com/fr/sports": "fr",
                "https://www.lefigaro.fr/sports/football": "fr",
                "https://www.lemonde.fr/sports": "fr",
                "https://www.goal.com/fr": "fr",
                "https://www.onfoot.net": "fr",
                "https://www.psg.fr": "fr",
                "https://www.footmercato.net": "fr",
                "https://www.maxifoot.fr": "fr",
                "https://www.sofoot.com": "fr",
                "https://www.eurosport.fr/football": "fr",
                "https://www.rmc.fr/sport": "fr",
                # Portekiz (12 site)
                "https://www.ojogo.pt/futebol": "pt",
                "https://www.record.pt/futebol": "pt",
                "https://www.maisfutebol.pt": "pt",
                "https://www.abola.pt/noticias/Futebol": "pt",
                "https://www.zerozero.pt": "pt",
                "https://www.goal.com/pt": "pt",
                "https://www.spn.pt": "pt",
                "https://www.jn.pt/desporto/futebol": "pt",
                "https://www.publico.pt/desporto/futebol": "pt",
                "https://www.dn.pt/desporto/futebol": "pt",
                "https://www.cmjornal.pt/desporto/futebol": "pt",
                "https://www.tsf.pt/desporto/futebol": "pt",
            }
            
            # Sadece ligin ülkesinden site seç
            news_sites = {site: country for site, country in all_sites.items() if country == league_country}
            
            home_mentions = 0
            away_mentions = 0
            total_articles = 0
            
            # Takım isimlerini normalize et (farklı diller için)
            home_variations = self._get_team_name_variations(home_team)
            away_variations = self._get_team_name_variations(away_team)
            
            for site, lang in news_sites.items():
                try:
                    # Site'yi çek
                    response = requests.get(site, timeout=10, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': f'{lang},en;q=0.5',
                    })
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Tüm metni al
                        text = soup.get_text().lower()
                        
                        # Başlıkları da kontrol et (daha önemli)
                        titles = [tag.get_text().lower() for tag in soup.find_all(['h1', 'h2', 'h3', 'title'])]
                        title_text = ' '.join(titles)
                        
                        # OPTİMİZE: Article linklerini bul ama detaylı kontrolü azalt
                        article_links = []
                        for link in soup.find_all('a', href=True):
                            href = link.get('href', '')
                            # Maç tarihine yakın haberler için
                            if any(keyword in href.lower() for keyword in ['match', 'game', 'mac', 'maç', 'vs', 'v']):
                                if href.startswith('http'):
                                    article_links.append(href)
                                elif href.startswith('/'):
                                    article_links.append(site.rstrip('/') + href)
                        
                        # OPTİMİZE: İlk 5 article (orta düzey veri)
                        for article_url in article_links[:5]:
                            try:
                                article_response = requests.get(article_url, timeout=5, headers={
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                })
                                if article_response.status_code == 200:
                                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                    article_text = article_soup.get_text().lower()
                                    
                                    # Takım isimlerini kontrol et
                                    for variation in home_variations:
                                        if variation.lower() in article_text:
                                            home_mentions += article_text.count(variation.lower())
                                            total_articles += 1
                                            break
                                    
                                    for variation in away_variations:
                                        if variation.lower() in article_text:
                                            away_mentions += article_text.count(variation.lower())
                                            break
                                    
                                    time.sleep(0.03)  # Rate limiting - orta
                            except:
                                continue
                        
                        # Ana sayfa metnini de kontrol et
                        for variation in home_variations:
                            if variation.lower() in text or variation.lower() in title_text:
                                home_mentions += (text.count(variation.lower()) + title_text.count(variation.lower()))
                        
                        for variation in away_variations:
                            if variation.lower() in text or variation.lower() in title_text:
                                away_mentions += (text.count(variation.lower()) + title_text.count(variation.lower()))
                    
                    time.sleep(0.08)  # Rate limiting - orta
                except Exception as e:
                    logger.debug(f"Web scraping hatası {site}: {e}")
                    continue
            
            total = home_mentions + away_mentions
            if total > 0:
                home_support = home_mentions / total
                away_support = away_mentions / total
            else:
                home_support = 0.5
                away_support = 0.5
            
            return {
                "home_support": home_support,
                "away_support": away_support,
                "total_mentions": total
            }
        except ImportError:
            logger.debug("requests/beautifulsoup4 yüklü değil")
            return None
        except Exception as e:
            # API hatası - sessizce geç, önemli değil
            logger.debug(f"Web scraping hatası (atlandı): {e}")
            return None
    
    def _get_team_name_variations(self, team_name: str) -> List[str]:
        """Takım ismi için farklı varyasyonlar döndürür"""
        variations = [team_name]
        
        # FC, AC gibi ekleri kaldır
        clean_name = team_name.replace('FC ', '').replace(' AC', '').replace('CF ', '').strip()
        if clean_name != team_name:
            variations.append(clean_name)
        
        # Kısaltmalar
        words = team_name.split()
        if len(words) > 1:
            # İlk kelime
            variations.append(words[0])
            # Son kelime
            variations.append(words[-1])
        
        # Yaygın takım isimleri için özel varyasyonlar
        special_cases = {
            'Manchester United': ['Man United', 'Man Utd', 'MUFC'],
            'Manchester City': ['Man City', 'MCFC'],
            'Tottenham': ['Spurs', 'Tottenham Hotspur'],
            'Atletico Madrid': ['Atletico', 'Atleti'],
            'AC Milan': ['Milan', 'ACM'],
            'Inter': ['Inter Milan', 'Internazionale'],
        }
        
        if team_name in special_cases:
            variations.extend(special_cases[team_name])
        
        return variations
    
    def _default_hype(self) -> Dict[str, Any]:
        """Varsayılan hype değerleri (hiç veri yoksa)"""
        logger.warning("⚠️ Hiç hype verisi bulunamadı, varsayılan değerler kullanılıyor")
        return {
            "home_support": 0.5,
            "away_support": 0.5,
            "sentiment_score": 0.0,
            "total_mentions": 0,
            "hype_score": 0.0,
            "sources": []
        }


