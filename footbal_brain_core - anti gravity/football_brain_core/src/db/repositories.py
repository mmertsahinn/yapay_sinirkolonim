from datetime import datetime, date
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from src.db.connection import get_session
from src.db.schema import (
    League, Team, Match, Stat, Market, Prediction, Result,
    Experiment, ModelVersion, Explanation, ErrorCase,
    ErrorCluster, HumanFeedback, EvolutionPlan
)


class LeagueRepository:
    @staticmethod
    def get_or_create(session: Session, name: str, country: Optional[str] = None, code: Optional[str] = None) -> League:
        league = session.query(League).filter(League.name == name).first()
        if not league:
            league = League(name=name, country=country, code=code)
            session.add(league)
            session.flush()
        return league
    
    @staticmethod
    def get_by_id(session: Session, league_id: int) -> Optional[League]:
        return session.query(League).filter(League.id == league_id).first()
    
    @staticmethod
    def get_by_name(session: Session, name: str) -> Optional[League]:
        return session.query(League).filter(League.name == name).first()
    
    @staticmethod
    def get_all(session: Session) -> List[League]:
        return session.query(League).all()


class TeamRepository:
    @staticmethod
    def get_or_create(session: Session, name: str, league_id: int, code: Optional[str] = None) -> Team:
        team = session.query(Team).filter(
            and_(Team.name == name, Team.league_id == league_id)
        ).first()
        if not team:
            team = Team(name=name, league_id=league_id, code=code)
            session.add(team)
            session.flush()
        return team
    
    @staticmethod
    def get_by_id(session: Session, team_id: int) -> Optional[Team]:
        return session.query(Team).filter(Team.id == team_id).first()
    
    @staticmethod
    def get_by_league(session: Session, league_id: int) -> List[Team]:
        return session.query(Team).filter(Team.league_id == league_id).all()
    
    @staticmethod
    def get_all(session: Session) -> List[Team]:
        return session.query(Team).all()
    
    @staticmethod
    def get_by_name(session: Session, name: str) -> Optional[Team]:
        return session.query(Team).filter(Team.name.ilike(f"%{name}%")).first()
    
    @staticmethod
    def get_by_name_and_league(session: Session, name: str, league_id: int) -> Optional[Team]:
        """Takım ismini ve lig ID'sini kullanarak takımı bulur (isim eşleştirmesi esnek)"""
        # Önce tam eşleşme dene
        team = session.query(Team).filter(
            and_(Team.name == name, Team.league_id == league_id)
        ).first()
        
        if team:
            return team
        
        # Sonra kısmi eşleşme dene (case-insensitive)
        team = session.query(Team).filter(
            and_(Team.name.ilike(f"%{name}%"), Team.league_id == league_id)
        ).first()
        
        return team


class MatchRepository:
    @staticmethod
    def get_or_create(
        session: Session,
        match_id: str,
        league_id: int,
        home_team_id: int,
        away_team_id: int,
        match_date: datetime,
        home_score: Optional[int] = None,
        away_score: Optional[int] = None,
        status: Optional[str] = None
    ) -> Match:
        match = session.query(Match).filter(Match.match_id == match_id).first()
        if not match:
            match = Match(
                match_id=match_id,
                league_id=league_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                match_date=match_date,
                home_score=home_score,
                away_score=away_score,
                status=status
            )
            session.add(match)
            session.flush()
        else:
            if home_score is not None:
                match.home_score = home_score
            if away_score is not None:
                match.away_score = away_score
            if status:
                match.status = status
            match.updated_at = datetime.utcnow()
        return match
    
    @staticmethod
    def get_by_id(session: Session, match_id: int) -> Optional[Match]:
        return session.query(Match).filter(Match.id == match_id).first()
    
    @staticmethod
    def get_by_date_range(session: Session, date_from: date, date_to: date) -> List[Match]:
        return session.query(Match).filter(
            and_(
                Match.match_date >= date_from,
                Match.match_date <= date_to
            )
        ).all()
    
    @staticmethod
    def get_by_league_and_season(session: Session, league_id: int, season: int) -> List[Match]:
        season_start = datetime(season, 8, 1)
        season_end = datetime(season + 1, 7, 31)
        return session.query(Match).filter(
            and_(
                Match.league_id == league_id,
                Match.match_date >= season_start,
                Match.match_date <= season_end
            )
        ).order_by(Match.match_date).all()
    
    @staticmethod
    def get_team_matches(session: Session, team_id: int, limit: int = 10) -> List[Match]:
        return session.query(Match).filter(
            or_(
                Match.home_team_id == team_id,
                Match.away_team_id == team_id
            )
        ).order_by(Match.match_date.desc()).limit(limit).all()


class StatRepository:
    @staticmethod
    def create(session: Session, match_id: int, team_id: int, stat_type: str, stat_value: float) -> Stat:
        stat = Stat(
            match_id=match_id,
            team_id=team_id,
            stat_type=stat_type,
            stat_value=stat_value
        )
        session.add(stat)
        return stat
    
    @staticmethod
    def get_by_match(session: Session, match_id: int) -> List[Stat]:
        return session.query(Stat).filter(Stat.match_id == match_id).all()


class MarketRepository:
    @staticmethod
    def get_or_create(session: Session, name: str, description: Optional[str] = None) -> Market:
        market = session.query(Market).filter(Market.name == name).first()
        if not market:
            market = Market(name=name, description=description)
            session.add(market)
            session.flush()
        return market
    
    @staticmethod
    def get_all(session: Session) -> List[Market]:
        return session.query(Market).all()


class PredictionRepository:
    @staticmethod
    def create(
        session: Session,
        match_id: int,
        market_id: int,
        predicted_outcome: str,
        model_version_id: int,
        p_hat: Optional[float] = None
    ) -> Prediction:
        prediction = Prediction(
            match_id=match_id,
            market_id=market_id,
            predicted_outcome=predicted_outcome,
            model_version_id=model_version_id,
            p_hat=p_hat
        )
        session.add(prediction)
        return prediction
    
    @staticmethod
    def get_by_match(session: Session, match_id: int) -> List[Prediction]:
        return session.query(Prediction).filter(Prediction.match_id == match_id).all()
    
    @staticmethod
    def get_by_model_version(session: Session, model_version_id: int) -> List[Prediction]:
        return session.query(Prediction).filter(Prediction.model_version_id == model_version_id).all()


class ResultRepository:
    @staticmethod
    def get_or_create(
        session: Session,
        match_id: int,
        market_id: int,
        actual_outcome: str
    ) -> Result:
        result = session.query(Result).filter(
            and_(
                Result.match_id == match_id,
                Result.market_id == market_id
            )
        ).first()
        if not result:
            result = Result(
                match_id=match_id,
                market_id=market_id,
                actual_outcome=actual_outcome
            )
            session.add(result)
        else:
            result.actual_outcome = actual_outcome
            result.updated_at = datetime.utcnow()
        return result
    
    @staticmethod
    def get_by_match(session: Session, match_id: int) -> List[Result]:
        return session.query(Result).filter(Result.match_id == match_id).all()


class ModelVersionRepository:
    @staticmethod
    def create(session: Session, version: str, description: Optional[str] = None) -> ModelVersion:
        model_version = ModelVersion(version=version, description=description)
        session.add(model_version)
        session.flush()
        return model_version
    
    @staticmethod
    def get_active(session: Session) -> Optional[ModelVersion]:
        return session.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    
    @staticmethod
    def get_by_version(session: Session, version: str) -> Optional[ModelVersion]:
        return session.query(ModelVersion).filter(ModelVersion.version == version).first()
    
    @staticmethod
    def deactivate_all(session: Session):
        session.query(ModelVersion).update({"is_active": False})


class ExperimentRepository:
    @staticmethod
    def create(
        session: Session,
        experiment_id: str,
        config: Dict[str, Any],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        experiment = Experiment(
            experiment_id=experiment_id,
            config=config,
            period_start=period_start,
            period_end=period_end,
            metrics=metrics
        )
        session.add(experiment)
        session.flush()
        return experiment
    
    @staticmethod
    def get_by_id(session: Session, experiment_id: str) -> Optional[Experiment]:
        return session.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    
    @staticmethod
    def get_all(session: Session) -> List[Experiment]:
        return session.query(Experiment).order_by(Experiment.created_at.desc()).all()


class ExplanationRepository:
    @staticmethod
    def create(
        session: Session,
        match_id: int,
        market_id: int,
        llm_output: str,
        summary_stats: Optional[Dict[str, Any]] = None
    ) -> Explanation:
        explanation = Explanation(
            match_id=match_id,
            market_id=market_id,
            llm_output=llm_output,
            summary_stats=summary_stats
        )
        session.add(explanation)
        return explanation
    
    @staticmethod
    def get_by_match(session: Session, match_id: int) -> List[Explanation]:
        return session.query(Explanation).filter(Explanation.match_id == match_id).all()


class ErrorCaseRepository:
    """PRD: Error Inbox - Hatalı tahminlerin saklandığı repository"""
    
    @staticmethod
    def create(
        session: Session,
        match_id: int,
        market_id: int,
        predicted_outcome: str,
        actual_outcome: str,
        model_version_id: int,
        llm_comment: Optional[str] = None,
        user_note: Optional[str] = None
    ) -> ErrorCase:
        error_case = ErrorCase(
            match_id=match_id,
            market_id=market_id,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            model_version_id=model_version_id,
            llm_comment=llm_comment,
            user_note=user_note
        )
        session.add(error_case)
        session.flush()
        return error_case
    
    @staticmethod
    def get_unresolved(session: Session) -> List[ErrorCase]:
        return session.query(ErrorCase).filter(
            ErrorCase.error_cluster_id == None
        ).all()
    
    @staticmethod
    def get_by_cluster(session: Session, cluster_id: int) -> List[ErrorCase]:
        return session.query(ErrorCase).filter(
            ErrorCase.error_cluster_id == cluster_id
        ).all()


class ErrorClusterRepository:
    """PRD: Hata Cluster'ları - Benzer hataların gruplandığı repository"""
    
    @staticmethod
    def create(
        session: Session,
        cluster_name: str,
        error_summary: str,
        league_id: Optional[int] = None,
        market_id: Optional[int] = None,
        feature_vector: Optional[Dict[str, Any]] = None
    ) -> ErrorCluster:
        cluster = ErrorCluster(
            cluster_name=cluster_name,
            league_id=league_id,
            market_id=market_id,
            error_summary=error_summary,
            example_count=0,
            resolution_level="unresolved",
            feature_vector=feature_vector
        )
        session.add(cluster)
        session.flush()
        return cluster
    
    @staticmethod
    def get_unresolved(session: Session) -> List[ErrorCluster]:
        return session.query(ErrorCluster).filter(
            ErrorCluster.resolution_level == "unresolved"
        ).all()
    
    @staticmethod
    def update_resolution(
        session: Session,
        cluster_id: int,
        resolution_level: str,
        root_cause: Optional[str] = None
    ):
        cluster = session.query(ErrorCluster).filter(ErrorCluster.id == cluster_id).first()
        if cluster:
            cluster.resolution_level = resolution_level
            if root_cause:
                cluster.root_cause = root_cause
            session.flush()


class HumanFeedbackRepository:
    """PRD: Kullanıcı geri bildirimleri repository"""
    
    @staticmethod
    def create(
        session: Session,
        error_cluster_id: int,
        question: str
    ) -> HumanFeedback:
        feedback = HumanFeedback(
            error_cluster_id=error_cluster_id,
            question=question
        )
        session.add(feedback)
        session.flush()
        return feedback
    
    @staticmethod
    def answer(
        session: Session,
        feedback_id: int,
        user_answer: str,
        suggested_features: Optional[Dict[str, Any]] = None,
        action_taken: Optional[str] = None
    ):
        from datetime import datetime
        feedback = session.query(HumanFeedback).filter(HumanFeedback.id == feedback_id).first()
        if feedback:
            feedback.user_answer = user_answer
            feedback.suggested_features = suggested_features
            feedback.action_taken = action_taken
            feedback.answered_at = datetime.utcnow()
            session.flush()


class EvolutionPlanRepository:
    """PRD: Evrim planları repository"""
    
    @staticmethod
    def create(
        session: Session,
        error_cluster_id: int,
        plan_type: str,
        description: str,
        suggested_changes: Dict[str, Any]
    ) -> EvolutionPlan:
        plan = EvolutionPlan(
            error_cluster_id=error_cluster_id,
            plan_type=plan_type,
            description=description,
            suggested_changes=suggested_changes,
            status="pending"
        )
        session.add(plan)
        session.flush()
        return plan
    
    @staticmethod
    def get_pending(session: Session) -> List[EvolutionPlan]:
        return session.query(EvolutionPlan).filter(
            EvolutionPlan.status == "pending"
        ).all()


