import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class LeagueConfig:
    name: str
    country: str
    api_league_id: int


@dataclass
class ModelConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    sequence_length: int = 10


@dataclass
class LLMConfig:
    openai_api_key: Optional[str] = None
    grok_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    default_model: str = "openai/gpt-oss-20b:free"  # PRD: openai/gpt-oss-20b:free
    default_grok_model: str = "x-ai/grok-4.1-fast:free"  # PRD: x-ai/grok-4.1-fast:free
    use_grok_for_long: bool = False  # Uzun açıklamalar için Grok kullan
    max_tokens: int = 200


class Config:
    TARGET_LEAGUES: List[LeagueConfig] = [
        LeagueConfig("Premier League", "England", 39),
        LeagueConfig("La Liga", "Spain", 140),
        LeagueConfig("Serie A", "Italy", 135),
        LeagueConfig("Bundesliga", "Germany", 78),
        LeagueConfig("Ligue 1", "France", 61),
        LeagueConfig("Liga Portugal", "Portugal", 94),
        LeagueConfig("Süper Lig", "Turkey", 203),
    ]
    
    API_FOOTBALL_KEY: str = os.getenv("API_FOOTBALL_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "b2da97d4752c48119233564ff59b0f14")  # News API için hype verisi
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./football_brain.db")
    
    # OpenRouter API Key (GPT ve Grok için)
    # Varsayılan key config'de, ama ortam değişkeni öncelikli
    OPENROUTER_API_KEY: str = os.getenv(
        "OPENROUTER_API_KEY",
        "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"
    )
    
    HISTORICAL_SEASONS: int = 5
    SEQUENCE_LENGTH: int = 10
    
    MODEL_CONFIG: ModelConfig = ModelConfig()
    
    LLM_CONFIG: LLMConfig = LLMConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        grok_api_key=os.getenv("GROK_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    EXCEL_OUTPUT_PATH: str = "./reports/predictions.xlsx"
    REPORTS_DIR: str = "./reports"
    
    @classmethod
    def load_from_yaml(cls, path: str = "config.yaml") -> "Config":
        if os.path.exists(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                config = cls()
                if "leagues" in data:
                    config.TARGET_LEAGUES = [
                        LeagueConfig(**league) for league in data["leagues"]
                    ]
                if "model" in data:
                    config.MODEL_CONFIG = ModelConfig(**data["model"])
                return config
        return cls()

