import os
import time
import sys
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Callable

import requests


class APIFootballClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://v3.football.api-sports.io", 
                 on_limit_warning: Optional[Callable[[int, int], None]] = None):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided via API_FOOTBALL_KEY environment variable or constructor")
        self.base_url = base_url
        self.headers = {
            "x-apisports-key": self.api_key,
            "Content-Type": "application/json"
        }
        # Rate limiting - plan tipine göre ayarlanabilir
        # Free: 100/day, Basic: 300/day, Pro: 1000/day
        self.rate_limit_delay = 0.1  # Varsayılan (Free tier için güvenli)
        self.requests_today = 0
        self.daily_limit = 100  # Varsayılan Free tier limit
        self.on_limit_warning = on_limit_warning  # Callback: (remaining, limit)
        self.last_warning_remaining = None  # Tekrar uyarı vermemek için
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Rate limit kontrolü
        if self.requests_today >= self.daily_limit:
            self._notify_limit_exceeded()
            raise ValueError(f"Günlük API limiti aşıldı: {self.daily_limit} requests/day")
        
        # Rate limit delay (plan tipine göre optimize edilebilir)
        time.sleep(self.rate_limit_delay)
        
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        
        # Rate limit bilgilerini header'lardan al
        if "x-ratelimit-requests-limit" in response.headers:
            self.daily_limit = int(response.headers.get("x-ratelimit-requests-limit", 100))
        if "x-ratelimit-requests-remaining" in response.headers:
            remaining = int(response.headers.get("x-ratelimit-requests-remaining", 0))
            self.requests_today = self.daily_limit - remaining
            
            # Limit kontrolü ve bildirim
            self._check_and_notify_limit(remaining)
        
        # Plan tipine göre delay ayarla
        if self.daily_limit >= 1000:  # Pro plan
            self.rate_limit_delay = 0.05  # Daha hızlı
        elif self.daily_limit >= 300:  # Basic plan
            self.rate_limit_delay = 0.08
        else:  # Free plan
            self.rate_limit_delay = 0.1
        
        data = response.json()
        if "response" in data:
            return data["response"]
        return data
    
    def _check_and_notify_limit(self, remaining: int):
        """Limit durumunu kontrol et ve gerekirse bildirim gönder"""
        # Limit doldu mu?
        if remaining == 0:
            self._notify_limit_exceeded()
        # Limit azaldı mı? (10'dan az kaldıysa veya %20'nin altındaysa)
        elif remaining <= 10 or (remaining / self.daily_limit) < 0.2:
            # Sadece değiştiyse uyar (tekrar uyarı vermemek için)
            if self.last_warning_remaining != remaining:
                self._notify_limit_low(remaining)
                self.last_warning_remaining = remaining
    
    def _notify_limit_exceeded(self):
        """Limit dolduğunda bildirim gönder"""
        message = f"\n{'='*70}\n"
        message += f"!!! API LIMITI DOLDU! !!!\n"
        message += f"{'='*70}\n"
        message += f"Gunluk limit: {self.daily_limit} requests/day\n"
        message += f"Kullanilan: {self.requests_today}\n"
        message += f"Kalan: 0\n"
        message += f"{'='*70}\n"
        message += f"YENI API KEY GEREKLI!\n"
        message += f"Yeni key'i ayarlamak icin:\n"
        message += f'  $env:API_FOOTBALL_KEY="YENI_KEY_BURAYA"\n'
        message += f"{'='*70}\n"
        
        # Konsola yazdır (kırmızı renk için)
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleTextAttribute(kernel32.GetStdHandle(-11), 0x0C)  # Kırmızı
            except:
                pass
        
        try:
            print(message, flush=True)
        except (ValueError, OSError):
            # stdout kapalıysa sessizce devam et
            pass
        
        # Windows bildirimi gönder
        self._send_windows_notification(
            "API Limit Doldu!",
            f"API limiti doldu: {self.requests_today}/{self.daily_limit} kullanıldı"
        )
        
        # Callback çağır
        if self.on_limit_warning:
            try:
                self.on_limit_warning(0, self.daily_limit)
            except:
                pass
    
    def _notify_limit_low(self, remaining: int):
        """Limit azaldığında uyarı gönder"""
        percentage = (remaining / self.daily_limit) * 100
        message = f"\n{'='*70}\n"
        message += f"!!! API LIMITI AZALIYOR! !!!\n"
        message += f"{'='*70}\n"
        message += f"Kalan: {remaining}/{self.daily_limit} ({percentage:.1f}%)\n"
        message += f"Kullanilan: {self.requests_today}/{self.daily_limit}\n"
        message += f"{'='*70}\n"
        
        try:
            print(message, flush=True)
        except (ValueError, OSError):
            # stdout kapalıysa sessizce devam et
            pass
        
        # Windows bildirimi gönder
        self._send_windows_notification(
            "API Limit Uyarısı",
            f"API limiti azalıyor: {remaining} kaldı ({percentage:.1f}%)"
        )
        
        # Callback çağır
        if self.on_limit_warning:
            try:
                self.on_limit_warning(remaining, self.daily_limit)
            except:
                pass
    
    def _send_windows_notification(self, title: str, message: str):
        """Windows toast bildirimi gönder"""
        if sys.platform == "win32":
            try:
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast(title, message, duration=10, threaded=True)
            except ImportError:
                # win10toast yoksa, alternatif yöntem dene
                try:
                    import subprocess
                    # PowerShell ile bildirim gönder
                    ps_script = f'''
                    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                    $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                    $textNodes = $template.GetElementsByTagName("text")
                    $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) | Out-Null
                    $textNodes.Item(1).AppendChild($template.CreateTextNode("{message}")) | Out-Null
                    $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
                    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Football Brain Core").Show($toast)
                    '''
                    subprocess.Popen(["powershell", "-Command", ps_script], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
                except:
                    pass  # Bildirim gönderilemezse sessizce devam et
    
    def get_leagues(
        self,
        country: Optional[str] = None,
        season: Optional[int] = None,
        league_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        params = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season
        if league_id:
            params["id"] = league_id
        
        return self._make_request("/leagues", params)
    
    def get_teams(
        self,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        team_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        params = {}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if team_id:
            params["id"] = team_id
        
        return self._make_request("/teams", params)
    
    def get_fixtures(
        self,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        team_id: Optional[int] = None,
        fixture_id: Optional[int] = None,
        last: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        params = {}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if date_from:
            params["from"] = date_from.strftime("%Y-%m-%d")
        if date_to:
            params["to"] = date_to.strftime("%Y-%m-%d")
        if team_id:
            params["team"] = team_id
        if fixture_id:
            params["id"] = fixture_id
        if last:
            params["last"] = last
        
        return self._make_request("/fixtures", params)
    
    def get_fixture_events(self, fixture_id: int) -> List[Dict[str, Any]]:
        params = {"fixture": fixture_id}
        return self._make_request("/fixtures/events", params)
    
    def get_fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        params = {"fixture": fixture_id}
        return self._make_request("/fixtures/statistics", params)
    
    def get_standings(
        self,
        league_id: int,
        season: int
    ) -> List[Dict[str, Any]]:
        params = {
            "league": league_id,
            "season": season
        }
        return self._make_request("/standings", params)
    
    def get_odds(
        self,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        fixture_id: Optional[int] = None,
        bookmaker: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        params = {}
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if date_from:
            params["from"] = date_from.strftime("%Y-%m-%d")
        if date_to:
            params["to"] = date_to.strftime("%Y-%m-%d")
        if fixture_id:
            params["fixture"] = fixture_id
        if bookmaker:
            params["bookmaker"] = bookmaker
        
        return self._make_request("/odds", params)
    
    def get_league_ids_for_season(self, season: int) -> Dict[str, int]:
        target_leagues = {
            "Premier League": "England",
            "La Liga": "Spain",
            "Serie A": "Italy",
            "Bundesliga": "Germany",
            "Ligue 1": "France",
            "Liga Portugal": "Portugal",
            "Süper Lig": "Turkey"
        }
        
        league_ids = {}
        for league_name, country in target_leagues.items():
            leagues = self.get_leagues(country=country, season=season)
            for league in leagues:
                if league.get("league", {}).get("name") == league_name:
                    league_ids[league_name] = league["league"]["id"]
                    break
        
        return league_ids

