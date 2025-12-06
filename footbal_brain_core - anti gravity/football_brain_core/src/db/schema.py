from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    code: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    teams: Mapped[list["Team"]] = relationship("Team", back_populates="league")
    matches: Mapped[list["Match"]] = relationship("Match", back_populates="league")


class Team(Base):
    __tablename__ = "teams"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    league_id: Mapped[int] = mapped_column(Integer, ForeignKey("leagues.id"), nullable=False)
    code: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    league: Mapped["League"] = relationship("League", back_populates="teams")
    home_matches: Mapped[list["Match"]] = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches: Mapped[list["Match"]] = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")


class Match(Base):
    __tablename__ = "matches"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True, index=True)
    league_id: Mapped[int] = mapped_column(Integer, ForeignKey("leagues.id"), nullable=False)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    match_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Hashtag Hype Features
    home_support: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.5)
    away_support: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.5)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    total_tweets: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=0)
    hype_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    league: Mapped["League"] = relationship("League", back_populates="matches")
    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    stats: Mapped[list["Stat"]] = relationship("Stat", back_populates="match")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="match")
    results: Mapped[list["Result"]] = relationship("Result", back_populates="match")
    explanations: Mapped[list["Explanation"]] = relationship("Explanation", back_populates="match")
    odds: Mapped[Optional["MatchOdds"]] = relationship("MatchOdds", back_populates="match", uselist=False)


class MatchOdds(Base):
    """Bookmaker odds for matches"""
    __tablename__ = "match_odds"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False, unique=True, index=True)
    
    # Bet365 odds
    b365_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b365_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b365_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Betfair odds
    bf_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bf_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Betfred odds
    bfd_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bfd_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bfd_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # BetMGM odds
    bmgm_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bmgm_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bmgm_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Betvictor odds
    bv_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bv_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bv_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Coral odds
    cl_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cl_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cl_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Ladbrokes odds
    lb_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lb_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lb_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Pinnacle odds
    p_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # William Hill odds
    wh_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wh_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wh_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Market averages and maximums
    max_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Over/Under 2.5 goals
    b365_over_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b365_under_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_over_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_under_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_over_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_under_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_over_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_under_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Asian Handicap
    ah_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b365_ah_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b365_ah_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_ah_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_ah_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Closing odds (if available)
    b365_ch: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b365_cd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b365_ca: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Store all odds as JSON for flexibility
    all_odds: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    match: Mapped["Match"] = relationship("Match", back_populates="odds")


class Stat(Base):
    __tablename__ = "stats"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    stat_type: Mapped[str] = mapped_column(String(100), nullable=False)
    stat_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    match: Mapped["Match"] = relationship("Match", back_populates="stats")
    team: Mapped["Team"] = relationship("Team")


class Market(Base):
    __tablename__ = "markets"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="market")
    results: Mapped[list["Result"]] = relationship("Result", back_populates="market")
    explanations: Mapped[list["Explanation"]] = relationship("Explanation", back_populates="market")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    market_id: Mapped[int] = mapped_column(Integer, ForeignKey("markets.id"), nullable=False)
    predicted_outcome: Mapped[str] = mapped_column(String(255), nullable=False)
    p_hat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    match: Mapped["Match"] = relationship("Match", back_populates="predictions")
    market: Mapped["Market"] = relationship("Market", back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="predictions")


class Result(Base):
    __tablename__ = "results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    market_id: Mapped[int] = mapped_column(Integer, ForeignKey("markets.id"), nullable=False)
    actual_outcome: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    match: Mapped["Match"] = relationship("Match", back_populates="results")
    market: Mapped["Market"] = relationship("Market", back_populates="results")


class Experiment(Base):
    __tablename__ = "experiments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Explanation(Base):
    __tablename__ = "explanations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    market_id: Mapped[int] = mapped_column(Integer, ForeignKey("markets.id"), nullable=False)
    explanation_text: Mapped[str] = mapped_column(Text, nullable=False)
    llm_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    match: Mapped["Match"] = relationship("Match", back_populates="explanations")
    market: Mapped["Market"] = relationship("Market", back_populates="explanations")


class ErrorCase(Base):
    __tablename__ = "error_cases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    market_id: Mapped[int] = mapped_column(Integer, ForeignKey("markets.id"), nullable=False)
    predicted_outcome: Mapped[str] = mapped_column(String(255), nullable=False)
    actual_outcome: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=False)
    llm_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_cluster_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("error_clusters.id"), nullable=True)
    resolution_level: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    # Relationships
    match: Mapped["Match"] = relationship("Match")
    market: Mapped["Market"] = relationship("Market")
    model_version: Mapped["ModelVersion"] = relationship("ModelVersion")
    error_cluster: Mapped[Optional["ErrorCluster"]] = relationship("ErrorCluster", back_populates="error_cases")


class ErrorCluster(Base):
    __tablename__ = "error_clusters"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_name: Mapped[str] = mapped_column(String(255), nullable=False)
    league_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("leagues.id"), nullable=True)
    market_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("markets.id"), nullable=True)
    error_summary: Mapped[str] = mapped_column(Text, nullable=False)
    example_count: Mapped[int] = mapped_column(Integer, default=0)
    resolution_level: Mapped[str] = mapped_column(String(50), default="unresolved")
    root_cause: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feature_vector: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    error_cases: Mapped[list["ErrorCase"]] = relationship("ErrorCase", back_populates="error_cluster")
    league: Mapped[Optional["League"]] = relationship("League")
    market: Mapped[Optional["Market"]] = relationship("Market")


class HumanFeedback(Base):
    __tablename__ = "human_feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    error_cluster_id: Mapped[int] = mapped_column(Integer, ForeignKey("error_clusters.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    user_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_features: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    action_taken: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    answered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Relationships
    error_cluster: Mapped["ErrorCluster"] = relationship("ErrorCluster")


class EvolutionPlan(Base):
    __tablename__ = "evolution_plans"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    error_cluster_id: Mapped[int] = mapped_column(Integer, ForeignKey("error_clusters.id"), nullable=False)
    plan_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_changes: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # Relationships
    error_cluster: Mapped["ErrorCluster"] = relationship("ErrorCluster")
