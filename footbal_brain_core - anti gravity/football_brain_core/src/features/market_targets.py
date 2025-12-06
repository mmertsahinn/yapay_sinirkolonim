from enum import Enum
from typing import Dict, List, Tuple, Optional


class MarketType(Enum):
    MATCH_RESULT = "match_result"
    BTTS = "btts"
    OVER_UNDER_25 = "over_under_25"
    MATCH_RESULT_BTTS = "match_result_btts"
    MATCH_RESULT_OU = "match_result_ou"
    DOUBLE_CHANCE = "double_chance"
    GOAL_RANGE = "goal_range"
    FIRST_GOAL = "first_goal"
    HOME_AWAY_OU = "home_away_ou"
    CLEAN_SHEET = "clean_sheet"
    WIN_MARGIN = "win_margin"
    ODD_EVEN = "odd_even"
    CORRECT_SCORE = "correct_score"


MARKET_OUTCOMES: Dict[MarketType, List[str]] = {
    MarketType.MATCH_RESULT: ["1", "X", "2"],
    MarketType.BTTS: ["Yes", "No"],
    MarketType.OVER_UNDER_25: ["Over 2.5", "Under 2.5"],
    MarketType.MATCH_RESULT_BTTS: [
        "1-Yes", "1-No", "X-Yes", "X-No", "2-Yes", "2-No"
    ],
    MarketType.MATCH_RESULT_OU: [
        "1-Over", "1-Under", "X-Over", "X-Under", "2-Over", "2-Under"
    ],
    MarketType.DOUBLE_CHANCE: ["1X", "12", "X2"],
    MarketType.GOAL_RANGE: ["0-1", "2-3", "4-5", "6+"],
    MarketType.FIRST_GOAL: ["Home", "Away", "No Goal"],
    MarketType.HOME_AWAY_OU: [
        "Home Over 1.5", "Home Under 1.5",
        "Away Over 1.5", "Away Under 1.5"
    ],
    MarketType.CLEAN_SHEET: [
        "Home Clean Sheet", "Away Clean Sheet",
        "Both Clean Sheets", "Neither Clean Sheet"
    ],
    MarketType.WIN_MARGIN: [
        "Home 1", "Home 2", "Home 3+",
        "Away 1", "Away 2", "Away 3+",
        "Draw"
    ],
    MarketType.ODD_EVEN: ["Odd", "Even"],
    MarketType.CORRECT_SCORE: [
        "0-0", "1-0", "0-1", "1-1", "2-0", "0-2",
        "2-1", "1-2", "2-2", "3-0", "0-3", "3-1",
        "1-3", "3-2", "2-3", "3-3", "4-0", "0-4",
        "4-1", "1-4", "4-2", "2-4", "Other"
    ],
}


def calculate_match_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "1"
    elif home_score < away_score:
        return "2"
    else:
        return "X"


def calculate_btts(home_score: int, away_score: int) -> str:
    return "Yes" if home_score > 0 and away_score > 0 else "No"


def calculate_over_under_25(home_score: int, away_score: int) -> str:
    total = home_score + away_score
    return "Over 2.5" if total > 2.5 else "Under 2.5"


def calculate_goal_range(home_score: int, away_score: int) -> str:
    total = home_score + away_score
    if total <= 1:
        return "0-1"
    elif total <= 3:
        return "2-3"
    elif total <= 5:
        return "4-5"
    else:
        return "6+"


def calculate_correct_score(home_score: int, away_score: int) -> str:
    score_str = f"{home_score}-{away_score}"
    if score_str in MARKET_OUTCOMES[MarketType.CORRECT_SCORE]:
        return score_str
    return "Other"


def calculate_odd_even(home_score: int, away_score: int) -> str:
    total = home_score + away_score
    return "Odd" if total % 2 == 1 else "Even"


def calculate_all_market_outcomes(home_score: int, away_score: int) -> Dict[MarketType, str]:
    return {
        MarketType.MATCH_RESULT: calculate_match_result(home_score, away_score),
        MarketType.BTTS: calculate_btts(home_score, away_score),
        MarketType.OVER_UNDER_25: calculate_over_under_25(home_score, away_score),
        MarketType.GOAL_RANGE: calculate_goal_range(home_score, away_score),
        MarketType.CORRECT_SCORE: calculate_correct_score(home_score, away_score),
        MarketType.ODD_EVEN: calculate_odd_even(home_score, away_score),
    }


def get_market_names() -> List[str]:
    return [market.value for market in MarketType]


def get_num_classes_for_market(market_type: MarketType) -> int:
    return len(MARKET_OUTCOMES[market_type])







