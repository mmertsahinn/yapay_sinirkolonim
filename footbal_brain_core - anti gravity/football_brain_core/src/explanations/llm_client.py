import os
from typing import Dict, Any, Optional
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.provider = provider
        
        if provider == "openrouter":
            from football_brain_core.src.config import Config
            config = Config()
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or config.OPENROUTER_API_KEY or os.getenv("OPENAI_API_KEY")
            self.base_url = base_url or "https://openrouter.ai/api/v1"
            self.model = "openai/gpt-oss-20b:free"  # PRD'de belirtilen model
        elif provider == "openrouter-grok":
            # OpenRouter üzerinden Grok kullanımı
            from football_brain_core.src.config import Config
            config = Config()
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or config.OPENROUTER_API_KEY or os.getenv("GROK_API_KEY")
            self.base_url = base_url or "https://openrouter.ai/api/v1"
            self.model = "x-ai/grok-4.1-fast:free"  # PRD'de belirtilen Grok modeli
        elif provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = "gpt-4o-mini"
        elif provider == "grok":
            # Direkt Grok API (eski yöntem)
            self.api_key = api_key or os.getenv("GROK_API_KEY")
            self.base_url = base_url or "https://api.x.ai/v1"
            self.model = "grok-beta"
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openrouter', 'openai', or 'grok'")
        
        if not self.api_key:
            logger.warning("API key not found. LLM explanations will use fallback.")
    
    def generate_explanation(
        self,
        match_context: Dict[str, Any],
        predicted_outcomes: Dict[str, str],
        summary_stats: Dict[str, Any]
    ) -> str:
        prompt = self._build_prompt(match_context, predicted_outcomes, summary_stats)
        
        try:
            if self.provider == "openrouter" or self.provider == "openrouter-grok":
                response = self._call_openrouter(prompt)
            elif self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "grok":
                response = self._call_grok(prompt)
            else:
                response = "Explanation not available"
            
            return response
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._fallback_explanation(match_context, predicted_outcomes)
    
    def _build_prompt(
        self,
        match_context: Dict[str, Any],
        predicted_outcomes: Dict[str, str],
        summary_stats: Dict[str, Any]
    ) -> str:
        home_team = match_context.get("home_team", "Home Team")
        away_team = match_context.get("away_team", "Away Team")
        
        prompt = f"""Write a brief 2-3 sentence explanation for a football match prediction.

Match: {home_team} vs {away_team}

Predicted outcomes:
"""
        for market, outcome in predicted_outcomes.items():
            prompt += f"- {market}: {outcome}\n"
        
        prompt += f"""
Recent statistics:
- Home team form: {summary_stats.get('home_form', 'N/A')}
- Away team form: {summary_stats.get('away_form', 'N/A')}
- Home team goals (avg): {summary_stats.get('home_avg_goals', 'N/A')}
- Away team goals (avg): {summary_stats.get('away_avg_goals', 'N/A')}

Write a concise explanation (2-3 sentences) explaining why these predictions were made based on the team patterns and statistics. Be specific and data-driven."""
        
        return prompt
    
    def _call_openrouter(self, prompt: str) -> str:
        """OpenRouter API çağrısı - PRD'de belirtilen openai/gpt-oss-20b:free modeli"""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/football-brain-core",  # Opsiyonel: app tracking
            "X-Title": "Football Brain Core"  # Opsiyonel: app name
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a football analyst providing concise, data-driven match predictions. Write 2-3 sentences explaining team patterns and statistical reasoning."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _call_openai(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a football analyst providing concise match predictions."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_grok(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a football analyst."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _fallback_explanation(
        self,
        match_context: Dict[str, Any],
        predicted_outcomes: Dict[str, str]
    ) -> str:
        home_team = match_context.get("home_team", "Home Team")
        away_team = match_context.get("away_team", "Away Team")
        
        return f"Prediction for {home_team} vs {away_team} based on recent team performance patterns and statistical analysis."

