import argparse
import sys
from datetime import datetime, date
from typing import List
import logging

from football_brain_core.src.config import Config
from football_brain_core.src.ingestion.historical_loader import HistoricalLoader
from football_brain_core.src.ingestion.daily_jobs import DailyJobs
from football_brain_core.src.models.train_offline import OfflineTrainer
from football_brain_core.src.experiments.runner import ExperimentRunner
from football_brain_core.src.experiments.tracker import ExperimentTracker
from football_brain_core.src.reporting.daily_report import DailyReporter
from football_brain_core.src.reporting.weekly_report import WeeklyReporter
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_engine
from football_brain_core.src.db.schema import Base

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_database():
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database initialized")


def load_historical_data(seasons: List[int] = None):
    loader = HistoricalLoader()
    if seasons:
        loader.load_all_historical_data(seasons=seasons)
    else:
        loader.load_all_historical_data()
    logger.info("Historical data loading completed")


def run_daily_update():
    jobs = DailyJobs()
    jobs.run_daily_update()
    logger.info("Daily update completed")


def train_model(
    train_seasons: List[int],
    val_seasons: List[int],
    market_types: List[MarketType]
):
    config = Config()
    trainer = OfflineTrainer(market_types, config)
    
    from football_brain_core.src.db.repositories import LeagueRepository
    from football_brain_core.src.db.connection import get_session
    session = get_session()
    try:
        league_ids = [
            LeagueRepository.get_or_create(session, league.name).id
            for league in config.TARGET_LEAGUES
        ]
    finally:
        session.close()
    
    model = trainer.train(train_seasons, val_seasons, league_ids)
    logger.info("Model training completed")
    return model


def run_experiment(
    train_seasons: List[int],
    val_seasons: List[int],
    market_types: List[MarketType]
):
    runner = ExperimentRunner()
    result = runner.run_experiment(
        experiment_config={"model_config": {}},
        train_seasons=train_seasons,
        val_seasons=val_seasons,
        market_types=market_types
    )
    logger.info(f"Experiment completed: {result['experiment_id']}")
    return result


def generate_daily_report():
    reporter = DailyReporter()
    output_path = reporter.generate_daily_report()
    logger.info(f"Daily report generated: {output_path}")


def generate_weekly_report():
    reporter = WeeklyReporter()
    results = reporter.generate_weekly_report()
    logger.info(f"Weekly report generated: {results.get('excel_path', 'N/A')}")


def list_experiments():
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments()
    for exp in experiments:
        print(f"Experiment ID: {exp['experiment_id']}")
        print(f"Created: {exp['created_at']}")
        print(f"Metrics: {exp.get('metrics', {})}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Football Brain Core CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    init_parser = subparsers.add_parser("init-db", help="Initialize database")
    
    load_parser = subparsers.add_parser("load-historical", help="Load historical data")
    load_parser.add_argument("--seasons", type=int, nargs="+", help="Seasons to load")
    
    daily_parser = subparsers.add_parser("daily-update", help="Run daily update")
    
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--train-seasons", type=int, nargs="+", required=True)
    train_parser.add_argument("--val-seasons", type=int, nargs="+", required=True)
    
    experiment_parser = subparsers.add_parser("experiment", help="Run experiment")
    experiment_parser.add_argument("--train-seasons", type=int, nargs="+", required=True)
    experiment_parser.add_argument("--val-seasons", type=int, nargs="+", required=True)
    
    report_daily_parser = subparsers.add_parser("report-daily", help="Generate daily report")
    
    report_weekly_parser = subparsers.add_parser("report-weekly", help="Generate weekly report")
    
    list_exp_parser = subparsers.add_parser("list-experiments", help="List experiments")
    
    learn_parser = subparsers.add_parser("self-learn", help="Beyin kendini test edip öğrenir")
    learn_parser.add_argument("--season", type=int, required=True, help="Hangi sezon üzerinde öğrenilecek")
    learn_parser.add_argument("--max-iterations", type=int, default=10, help="Maksimum öğrenme iterasyonu")
    learn_parser.add_argument("--target-accuracy", type=float, default=0.70, help="Hedef doğruluk oranı")
    
    continuous_learn_parser = subparsers.add_parser("continuous-learn", help="Sürekli öğrenme döngüsü")
    continuous_learn_parser.add_argument("--seasons", type=int, nargs="+", required=True, help="Hangi sezonlar")
    continuous_learn_parser.add_argument("--max-iterations", type=int, default=10, help="Sezon başına maksimum iterasyon")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "init-db":
            init_database()
        
        elif args.command == "load-historical":
            load_historical_data(args.seasons)
        
        elif args.command == "daily-update":
            run_daily_update()
        
        elif args.command == "train":
            market_types = [
                MarketType.MATCH_RESULT,
                MarketType.BTTS,
                MarketType.OVER_UNDER_25,
            ]
            train_model(args.train_seasons, args.val_seasons, market_types)
        
        elif args.command == "experiment":
            market_types = [
                MarketType.MATCH_RESULT,
                MarketType.BTTS,
                MarketType.OVER_UNDER_25,
            ]
            run_experiment(args.train_seasons, args.val_seasons, market_types)
        
        elif args.command == "report-daily":
            generate_daily_report()
        
        elif args.command == "report-weekly":
            generate_weekly_report()
        
        elif args.command == "list-experiments":
            list_experiments()
        
        elif args.command == "self-learn":
            from football_brain_core.src.models.self_learning import SelfLearningBrain
            from football_brain_core.src.models.multi_task_model import MultiTaskModel
            from football_brain_core.src.db.repositories import ModelVersionRepository
            
            session = get_session()
            try:
                active_model = ModelVersionRepository.get_active(session)
                if not active_model:
                    logger.error("Aktif model bulunamadı! Önce model eğitmelisin.")
                    sys.exit(1)
                
                # Model yükleme kodu buraya eklenecek
                # Şimdilik placeholder
                logger.info("Model yükleniyor...")
                
                market_types = [
                    MarketType.MATCH_RESULT,
                    MarketType.BTTS,
                    MarketType.OVER_UNDER_25,
                ]
                
                # SelfLearningBrain oluştur ve öğren
                brain = SelfLearningBrain(model, market_types)
                results = brain.learn_from_past_matches(
                    season=args.season,
                    max_iterations=args.max_iterations,
                    target_accuracy=args.target_accuracy
                )
                
                logger.info(f"✅ Öğrenme tamamlandı! En iyi doğruluk: {results['best_accuracy']:.2%}")
            finally:
                session.close()
        
        elif args.command == "continuous-learn":
            from football_brain_core.src.models.self_learning import SelfLearningBrain
            from football_brain_core.src.models.multi_task_model import MultiTaskModel
            from football_brain_core.src.db.repositories import ModelVersionRepository
            
            session = get_session()
            try:
                active_model = ModelVersionRepository.get_active(session)
                if not active_model:
                    logger.error("Aktif model bulunamadı! Önce model eğitmelisin.")
                    sys.exit(1)
                
                market_types = [
                    MarketType.MATCH_RESULT,
                    MarketType.BTTS,
                    MarketType.OVER_UNDER_25,
                ]
                
                brain = SelfLearningBrain(model, market_types)
                results = brain.continuous_learning_loop(
                    seasons=args.seasons,
                    max_iterations_per_season=args.max_iterations
                )
                
                logger.info(f"✅ Sürekli öğrenme tamamlandı!")
                logger.info(f"🏆 Genel en iyi doğruluk: {results['overall_best_accuracy']:.2%}")
            finally:
                session.close()
    
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

