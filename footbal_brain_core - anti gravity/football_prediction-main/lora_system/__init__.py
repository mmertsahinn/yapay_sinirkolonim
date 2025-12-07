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
from .social_network import SocialNetwork
from .social_network_visualizer import SocialNetworkVisualizer
from .mentorship_inheritance import MentorshipInheritance, mentorship_inheritance
from .collective_intelligence import CollectiveIntelligence, collective_intelligence
from .temperament_encoder import TemperamentEncoder, get_temperament_encoder
from .social_attention_layer import SocialAttentionLayer, get_social_attention
from .knowledge_distillation import DiscoveryDistillation, get_distillation
from .meta_learning import CumulativeMetaLearner, get_meta_learner
from .deep_collective_intelligence import DeepLoRACollectiveIntelligence, get_deep_collective_intelligence
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
from .adaptive_nature import AdaptiveNature
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

# NEW 3 MAIN SYSTEMS
from .advanced_categorization import AdvancedCategorization
from .social_network import SocialNetwork

# NEUROEVOLUTION DEEP COLLECTIVE INTELLIGENCE SYSTEMS
from .neuroevolution_engine import (
    NeuroevolutionEngine,
    EvolutionStrategy,
    get_neuroevolution_engine
)
from .thinking_patterns import (
    ThinkingPattern,
    EvolvableThinkingSystem,
    get_thinking_system
)
from .evolvable_lora_adapter import (
    EvolvableLoRAAdapter,
    DynamicLoRALinear,
    create_evolvable_from_base
)
from .deep_knowledge_transfer import (
    DeepKnowledgeTransfer,
    Domain,
    get_deep_knowledge_transfer
)
from .enhanced_temperament_processing import (
    EnhancedTemperamentProcessing,
    get_enhanced_temperament_processing
)
from .advanced_social_network import (
    AdvancedSocialNetwork,
    get_advanced_social_network
)
from .cumulative_evolution import (
    CumulativeEvolutionSystem,
    get_cumulative_evolution_system
)
from .collective_intelligence_brain import (
    CollectiveIntelligenceBrain,
    get_collective_intelligence_brain
)
from .integration_wrapper import (
    upgrade_lora_to_evolvable,
    upgrade_social_network_to_advanced,
    gradual_migration_wrapper
)
from .performance_optimizer import (
    GPUOptimizer,
    BatchProcessor,
    MemoryOptimizer,
    PerformanceMonitor,
    get_gpu_optimizer,
    get_batch_processor,
    get_memory_optimizer,
    get_performance_monitor
)

__all__ = [
    'LoRAAdapter',
    'LoRALinear',
    'ChaosEvolutionManager',
    'MetaLoRA',
    'SimpleMetaLoRA',
    'ReplayBuffer',
    'SpecializationTracker',
    'AdvancedCategorization', # Replaces SpecializationSystem
    'SocialNetwork',
    'AdaptiveNature',
    'ScorePredictor',
    'score_predictor',
    'AdvancedMechanicsManager',
    'TemperamentBasedLearning',
    'temperament_learning',
    # Neuroevolution systems
    'NeuroevolutionEngine',
    'EvolutionStrategy',
    'get_neuroevolution_engine',
    'ThinkingPattern',
    'EvolvableThinkingSystem',
    'get_thinking_system',
    'EvolvableLoRAAdapter',
    'DynamicLoRALinear',
    'create_evolvable_from_base',
    'DeepKnowledgeTransfer',
    'Domain',
    'get_deep_knowledge_transfer',
    'EnhancedTemperamentProcessing',
    'get_enhanced_temperament_processing',
    'AdvancedSocialNetwork',
    'get_advanced_social_network',
    'CumulativeEvolutionSystem',
    'get_cumulative_evolution_system',
    'CollectiveIntelligenceBrain',
    'get_collective_intelligence_brain',
    # Integration
    'upgrade_lora_to_evolvable',
    'upgrade_social_network_to_advanced',
    'gradual_migration_wrapper',
    # Performance
    'GPUOptimizer',
    'BatchProcessor',
    'MemoryOptimizer',
    'PerformanceMonitor',
    'get_gpu_optimizer',
    'get_batch_processor',
    'get_memory_optimizer',
    'get_performance_monitor'
]

