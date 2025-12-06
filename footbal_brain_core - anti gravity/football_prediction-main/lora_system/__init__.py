"""
🧬 KAOTIK EVRİMSEL LoRA SİSTEMİ
================================

Tamamen doğal seleksiyon ile çalışan, kaotik evrimsel öğrenme sistemi.

Modüller:
- lora_adapter: LoRA temel sınıfı (rank=16, alpha=16)
- chaos_evolution: Kaotik evrim motoru (doğum/ölüm/çiftleşme)
- meta_lora: Meta-LoRA attention sistemi
- replay_buffer: Hafıza/replay buffer
"""

__version__ = "1.0.0"
__author__ = "Football Brain Core"

from .lora_adapter import LoRAAdapter, LoRALinear
from .chaos_evolution import ChaosEvolutionManager
from .meta_lora import MetaLoRA, SimpleMetaLoRA
from .replay_buffer import ReplayBuffer
from .specialization_tracker import SpecializationTracker
from .score_predictor import ScorePredictor, score_predictor
from .collective_memory import CollectiveMemory, collective_memory
from .miracle_system import MiracleSystem, miracle_system
from .resurrection_system_v2 import ResurrectionSystemV2
from .top_score_calculator import TopScoreCalculator, top_score_calculator
from .advanced_score_calculator import AdvancedScoreCalculator, advanced_score_calculator
from .lora_archetypes import LoRAArchetypes, lora_archetypes
from .advanced_mechanics import (
    AdvancedMechanicsManager,
    EliteResistance,
    AntiInbreeding,
    Hibernation,
    PositiveFeedbackBrake
)
from .temperament_learning import TemperamentBasedLearning, temperament_learning
from .psychological_responses import PsychologicalResponseSystem, psychological_responses
from .adaptive_nature import AdaptiveNatureSystem, adaptive_nature
from .historical_learning import HistoricalLearningSystem, historical_learning
from .reputation_system import ReputationSystem, reputation_system
from .experience_based_resistance import ExperienceBasedResistance, experience_resistance
from .ultra_chaotic_mating import UltraChaoticMating, ultra_chaotic_mating
from .dynamic_specialization import DynamicSpecialization, dynamic_specialization
from .meta_adaptive_learning import MetaAdaptiveLearning, meta_adaptive_learning
from .kfac_fisher import KFACFisher, kfac_fisher
from .life_energy_system import LifeEnergySystem, life_energy_system
from .master_flux_equation import MasterFluxEquation, master_flux
from .fluid_temperament import FluidTemperament, fluid_temperament
from .ghost_fields import GhostFields, ghost_fields
from .nature_thermostat import NatureThermostat, nature_thermostat
from .tes_scoreboard import TESScoreboard, tes_scoreboard
from .physics_based_archetypes import PhysicsBasedArchetypes, physics_archetypes
from .tes_triple_scoreboard import TESTripleScoreboard, tes_triple_scoreboard
from .living_loras_reporter import LivingLoRAsReporter, living_reporter
from .langevin_dynamics import LangevinDynamics, langevin_dynamics
from .lazarus_potential import LazarusPotential, lazarus_potential
from .onsager_machlup import OnsagerMachlup, onsager_machlup
from .particle_archetypes import ParticleArchetypes, particle_archetypes
from .team_specialization_manager import TeamSpecializationManager, team_specialization_manager
from .global_specialization_manager import GlobalSpecializationManager, global_specialization_manager
from .death_immunity_system import (
    calculate_death_immunity,
    apply_death_immunity_to_energy_loss,
    check_specialization_loss_warning
)
try:
    from .specialization_system import SpecializationSystem
except ImportError:
    SpecializationSystem = SpecializationTracker  # Fallback

__all__ = [
    'LoRAAdapter',
    'LoRALinear',
    'ChaosEvolutionManager',
    'MetaLoRA',
    'SimpleMetaLoRA',
    'ReplayBuffer',
    'SpecializationTracker',
    'SpecializationSystem',
    'ScorePredictor',
    'score_predictor',
    'AdvancedMechanicsManager',
    'TemperamentBasedLearning',
    'temperament_learning'
]

