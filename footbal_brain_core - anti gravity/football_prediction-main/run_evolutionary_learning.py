"""
🌍 EVRİMSEL ÖĞRENME - ANA LOOP
===============================

Tüm sistemi birleştiren ana execution script.

Sistem bileşenleri:
1. Base Ensemble (Sklearn modeller)
2. LoRA Ecosystem (20+ LoRA'lar)
3. Chaos Evolution (Doğum/ölüm/çiftleşme)
4. Meta-LoRA (Attention)
5. Nature + Entropy (Doğa tepkileri + Soğuma)
6. Natural Triggers (Eşik bazlı olaylar)
7. Chaotic Global Learner
8. Advanced Incremental Learner
9. Replay Buffer
10. Evolution Logger
"""

import os
import sys
# Windows konsolunda emoji desteği için
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Optional

# LoRA sistem modülleri
from lora_system import (
    LoRAAdapter, 
    ChaosEvolutionManager,
    MetaLoRA, 
    SimpleMetaLoRA,
    ReplayBuffer,
    SpecializationTracker,
    AdvancedMechanicsManager
)

# 🧬 DEEP LEARNING & SIEVE (NEW!)
from lora_system.deep_learning_optimization import DeepKnowledgeDistiller, CollectiveDeepLearner
from lora_system.background_sieve import BackgroundSieve
from lora_system.butterfly_effect import ButterflyEffect
from lora_system.tribe_trainer import TribeTrainer

# 🎯 ADVANCED CATEGORIZATION (NEW!)
from lora_system.advanced_categorization import AdvancedCategorization
from lora_system.social_network_visualizer import SocialNetworkVisualizer
from lora_system.nature_entropy_system import (
    NatureEntropySystem, 
    GoallessDriftSystem,
    Goal,
    TraumaEvent
)

from lora_system.evolution_logger import EvolutionLogger
from lora_system.lora_wallet import WalletManager
from lora_system.match_results_logger import MatchResultsLogger
from lora_system.collective_memory import CollectiveMemory

# 🌊 PARÇACIK FİZİĞİ SİSTEMLERİ!
from lora_system.langevin_dynamics import langevin_dynamics
from lora_system.lazarus_potential import lazarus_potential
from lora_system.onsager_machlup import onsager_machlup
from lora_system.particle_archetypes import particle_archetypes
# Not: AdaptiveNatureSystem zaten aşağıda başlatılıyor!

# Mevcut sistemler
try:
    from chaotic_global_learner import ChaoticGlobalLearner
    CHAOTIC_AVAILABLE = True
except ImportError:
    CHAOTIC_AVAILABLE = False
    print("⚠️ ChaoticGlobalLearner bulunamadı, kullanılmayacak")

try:
    from advanced_incremental_system import AdvancedIncrementalLearner
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False
    print("⚠️ AdvancedIncrementalLearner bulunamadı, kullanılmayacak")


class EvolutionaryLearningSystem:
    """
    Tüm sistemi yöneten ana sınıf
    """
    
    def __init__(self, config_path: str = "evolutionary_config.yaml"):
        print("\n" + "="*80)
        print("🌍 EVRİMSEL ÖĞRENME SİSTEMİ BAŞLATILIYOR")
        print("="*80 + "\n")
        
        # Config yükle
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Device
        self.device = self.config.get('device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA kullanılamıyor, CPU'ya geçiliyor")
            self.device = 'cpu'
        
        print(f"💻 Device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Paths
        self.paths = self.config['paths']
        
        # Sonuç dosyası (gerçek sonuçlar için)
        self.results_df = None
        
        # Sistemleri başlat
        self._initialize_systems()
        
        print("\n✅ Tüm sistemler hazır!\n")
    
    def _initialize_systems(self):
        """Tüm alt sistemleri başlat"""
        
        # 1) Base Ensemble yükle
        print("📦 Base Ensemble yükleniyor...")
        self.ensemble = joblib.load(self.paths['base_model'])
        self.label_encoder = joblib.load(self.paths['label_encoder'])
        print(f"   ✅ {len(self.label_encoder.classes_)} sınıf: {self.label_encoder.classes_}")
        
        # 2) LoRA Evolution Manager
        print("\n🧬 Evrim Motoru başlatılıyor...")
        self.evolution_manager = ChaosEvolutionManager(self.config, device=self.device)
        
        # İlk popülasyonu oluştur
        start_pop = self.config['population']['start_population']
        # 🌊 INPUT DIM: 60 (base) + 3 (base_proba) + 15 (tarihsel) = 78
        self.evolution_manager.initialize_population(start_pop, input_dim=78, hidden_dim=128, device=self.device)
        
        # 🔬 Her LoRA'ya fizik özelliklerini ver (Life Energy + Fluid Temperament zaten sonra eklenir)
        
        # 3) Meta-LoRA
        print("\n🧠 Meta-LoRA başlatılıyor...")
        if self.config['meta_lora']['enabled']:
            self.meta_lora = MetaLoRA(input_dim=78, hidden_dim=64).to(self.device)
            print("   ✅ Attention-based Meta-LoRA")
        else:
            self.meta_lora = SimpleMetaLoRA()
            print("   ✅ Simple Fitness-based Meta-LoRA")
        
        # 4) Nature + Entropy
        print("\n🌍 Doğa + Entropi Sistemi başlatılıyor...")
        self.nature_system = NatureEntropySystem()
        
        # 5) Natural Triggers (REMOVED - INTEGRATED INTO ADAPTIVE NATURE)
        # self.trigger_system = NaturalTriggerSystem()
        
        # 6) Replay Buffer
        print("\n💾 Replay Buffer başlatılıyor...")
        buffer_size = self.config['buffer']['max_size']
        self.buffer = ReplayBuffer(max_size=buffer_size)
        
        # Buffer'ı yüklemeyi dene
        if os.path.exists(self.paths['buffer']):
            self.buffer.load(self.paths['buffer'])
        
        # 7) Evolution Logger
        print("\n📝 Evolution Logger başlatılıyor...")
        self.logger = EvolutionLogger(log_dir="evolution_logs")
        
        # 7.5) Match Results Logger (evolution_logs içinde!)
        print("\n📊 Match Results Logger başlatılıyor...")
        match_log_file = os.path.join("evolution_logs", "match_results.log")
        self.match_logger = MatchResultsLogger(log_file=match_log_file)
        
        # 7.6) ⚡ Living LoRAs Reporter (CANLI RAPOR!)
        print("\n⚡ Living LoRAs Reporter başlatılıyor...")
        from lora_system.living_loras_reporter import LivingLoRAsReporter
        self.living_reporter = LivingLoRAsReporter()
        
        # 8) Chaotic Global (opsiyonel)
        if CHAOTIC_AVAILABLE:
            print("\n🌪️ Chaotic Global Learner başlatılıyor...")
            self.chaotic_global = ChaoticGlobalLearner()
        else:
            self.chaotic_global = None
        
        # 9) Advanced Incremental (opsiyonel)
        if INCREMENTAL_AVAILABLE:
            print("\n📈 Advanced Incremental Learner başlatılıyor...")
            self.incremental_learner = AdvancedIncrementalLearner(n_features=60)
        else:
            self.incremental_learner = None
        
        # 10) Goalless Drift System
        self.goalless_system = GoallessDriftSystem()
        
        # 11) Specialization Tracker (Legacy support)
        print("\n🎯 Specialization Tracker başlatılıyor...")
        self.spec_tracker = SpecializationTracker()
        
        # 11.2) 🕸️ Arka Plan Elek Sistemi
        print("\n🕸️ Arka Plan Elek Sistemi (Sieve) başlatılıyor...")
        self.background_sieve = BackgroundSieve(buffer_size=50)

        # 11.3) 🧬 Deep Learning Optimization
        print("\n🧬 Deep Learning Optimization (Distillation) başlatılıyor...")
        self.distiller = DeepKnowledgeDistiller(device=self.device)
        self.collective_learner = CollectiveDeepLearner(device=self.device)
        self.tribe_trainer = TribeTrainer(self.distiller, device=self.device)

        # 11.5) 🎯 ADVANCED CATEGORIZATION
        print("\n🧠 Advanced Categorization System kısmi entegrasyon...")
        self.advanced_categorization = AdvancedCategorization()

        # 12) Parçacık Fiziği Motorları (Global instances)
        print("\n🌊 Parçacık Fiziği Motorları atanıyor...")
        self.langevin = langevin_dynamics
        self.lazarus = lazarus_potential
        self.onsager = onsager_machlup
        self.social_visualizer = SocialNetworkVisualizer()
        
        # 🌐 SOSYAL AĞ VE MENTÖRLÜK (KRİTİK!)
        print("\n🌐 Sosyal Öğrenme Ağı başlatılıyor...")
        from lora_system.social_network import SocialNetwork
        from lora_system.mentorship_inheritance import MentorshipInheritance
        from lora_system.collective_intelligence import CollectiveIntelligence
        self.social_network = SocialNetwork()
        self.mentorship_system = MentorshipInheritance()
        self.collective_intelligence = CollectiveIntelligence()
        
        # 11.4) 🦋 Kelebek Etkisi
        print("\n🦋 Kelebek Etkisi Modülü başlatılıyor...")
        self.butterfly_effect = ButterflyEffect(self.social_network)

        # 12) Wallet Manager
        print("\n💼 LoRA Wallet Manager başlatılıyor...")
        self.wallet_manager = WalletManager(wallet_dir="lora_wallets")
        
        # 12.5) Collective Memory (Ortak Hafıza - MODEL İÇİNDE!)
        print("\n🌐 Ortak Hafıza başlatılıyor...")
        self.collective_memory = CollectiveMemory()
        
        # 13) Advanced Mechanics
        print("\n🎯 Gelişmiş Mekanikler başlatılıyor...")
        adv_config = self.config.get('advanced_mechanics', {})
        self.advanced_mechanics = AdvancedMechanicsManager(adv_config)
        
        # 14) 🏆 Mucize Sistemi (Hall of Fame)
        print("\n🏆 Mucize Sistemi başlatılıyor...")
        from lora_system.miracle_system import MiracleSystem
        self.miracle_system = MiracleSystem(miracle_dir="mucizeler")
        
        # 🏆 TAKIM UZMANLIK YÖNETİCİSİ (Yeni sistem!)
        from lora_system.team_specialization_manager import team_specialization_manager
        self.team_spec_manager = team_specialization_manager
        
        # 🌍 GENEL UZMANLIK YÖNETİCİSİ (Takıma bağlı olmayan!)
        from lora_system.global_specialization_manager import global_specialization_manager
        self.global_spec_manager = global_specialization_manager
        
        # 🔄 SPECIALIZATION SYNC MANAGER (PT kopyalama/güncelleme!)
        print("\n🔄 Specialization Sync Manager başlatılıyor...")
        from lora_system.specialization_sync_manager import specialization_sync_manager
        self.sync_manager = specialization_sync_manager
        
        # 15) 📚 TÜM ZAMANLAR LoRA KAYDI (Ölüler dahil!)
        self.all_loras_ever = {}  # {lora_id: {'lora': lora_obj, 'final_fitness': ..., 'alive': True/False}}
        
        # 16) 🌍 EVRİMLEŞEN DOĞA SİSTEMİ (Adaptive Nature!)
        print("\n🌍 Evrimleşen Doğa Sistemi başlatılıyor...")
        from lora_system.adaptive_nature import AdaptiveNature
        self.adaptive_nature = AdaptiveNature()
        
        # 17) 📚 TARİHSEL ÖĞRENME SİSTEMİ
        print("\n📚 Tarihsel Öğrenme Sistemi başlatılıyor...")
        from lora_system.historical_learning import HistoricalLearningSystem
        self.historical_learning = HistoricalLearningSystem()
        
        # 18) 🛡️ DENEYİM BAZLI DİRENÇ SİSTEMİ
        print("\n🛡️ Deneyim Bazlı Direnç Sistemi başlatılıyor...")
        from lora_system.experience_based_resistance import ExperienceBasedResistance
        self.experience_resistance = ExperienceBasedResistance()
        # Legacy Hall Checker REMOVED
        
        # 19) 💕 ULTRA KAOTİK ÇİFTLEŞME
        print("\n💕 Ultra Kaotik Çiftleşme Sistemi başlatılıyor...")
        from lora_system.ultra_chaotic_mating import UltraChaoticMating
        self.ultra_mating = UltraChaoticMating()
        
        # 20) 🔍 DİNAMİK UZMANLIK KEŞFİ
        print("\n🔍 Dinamik Uzmanlık Keşif Sistemi başlatılıyor...")
        from lora_system.dynamic_specialization import DynamicSpecialization
        self.dynamic_spec = DynamicSpecialization()
        
        # 21) 🧠 META-ADAPTIF ÖĞRENME HIZI
        print("\n🧠 Meta-Adaptif Öğrenme Hızı Sistemi başlatılıyor...")
        from lora_system.meta_adaptive_learning import MetaAdaptiveLearning
        self.meta_learning = MetaAdaptiveLearning()
        
        # ============================================
        # 🔬 FİZİK MOTORU (TES!)
        # ============================================
        
        print("\n" + "🔬"*40)
        print("FİZİK MOTORU! (Termodinamik Evrimsel Skor)")
        print("🔬"*40)
        
        print("\n🌊 Master Flux Equation...")
        from lora_system.master_flux_equation import MasterFluxEquation
        self.master_flux = MasterFluxEquation()
        
        print("🔬 K-FAC Fisher...")
        from lora_system.kfac_fisher import KFACFisher
        self.kfac_fisher = KFACFisher()
        
        print("🧟 Lazarus Potential...")
        from lora_system.lazarus_potential import LazarusPotential
        self.lazarus = LazarusPotential()
        
        print("⚡ Life Energy...")
        from lora_system.life_energy_system import LifeEnergySystem
        self.life_energy = LifeEnergySystem()
        
        print("🌊 Fluid Temperament...")
        from lora_system.fluid_temperament import FluidTemperament
        self.fluid_temperament = FluidTemperament()
        
        print("👻 Ghost Fields...")
        from lora_system.ghost_fields import GhostFields
        self.ghost_fields = GhostFields(γ=0.1)
        
        print("👻 Ghost Field Logger...")
        from lora_system.ghost_field_logger import GhostFieldLogger
        self.ghost_logger = GhostFieldLogger()
        
        print("🔍 Log Validation System...")
        from lora_system.log_validation_system import LogValidationSystem
        self.log_validator = LogValidationSystem()
        
        print("📊 Log Dashboard...")
        from lora_system.log_dashboard import LogDashboard
        self.log_dashboard = LogDashboard()
        
        print("🔬 Hall & Specialization Auditor...")
        from lora_system.hall_specialization_auditor import HallSpecializationAuditor
        self.hall_auditor = HallSpecializationAuditor()
        
        print("🔄 Dynamic Relocation Engine...")
        from lora_system.dynamic_relocation_engine import DynamicRelocationEngine
        self.relocation_engine = DynamicRelocationEngine()
        
        print("🧟 Resurrection Debugger...")
        from lora_system.resurrection_debugger import ResurrectionDebugger
        self.resurrection_debugger = ResurrectionDebugger()
        
        # Legacy Hall Checker REMOVED
        
        print("📚 Comprehensive Population History...")
        from lora_system.comprehensive_population_history import ComprehensivePopulationHistory
        self.population_history = ComprehensivePopulationHistory()
        
        print("🔍 Team Specialization Auditor...")
        from lora_system.team_specialization_auditor import TeamSpecializationAuditor
        self.team_spec_auditor = TeamSpecializationAuditor()
        
        print("🔄 LoRA Sync Coordinator...")
        from lora_system.lora_sync_coordinator import LoRASyncCoordinator
        self.lora_sync = LoRASyncCoordinator()
        
        # 10) Doğa Termostatı
        from lora_system.nature_thermostat import NatureThermostat
        self.nature_thermostat = NatureThermostat()
        
        # 11) LoRA Panel Generator (YENİ!)
        from lora_system.lora_panel_generator import LoRAPanelGenerator
        self.panel_generator = LoRAPanelGenerator()
        
        # 12) Particle Archetypes (YENİ!)
        from lora_system.particle_archetypes import ParticleArchetypes
        self.particle_arch = ParticleArchetypes()
        
        # 13) TES Triple Scoreboard (YENİ!)
        from lora_system.tes_triple_scoreboard import TESTripleScoreboard
        self.tes_triple_scoreboard = TESTripleScoreboard()
        
        # 🔗 Dependency Injection: Thermostat'ı Evolution Manager'a ver
        self.evolution_manager.nature_thermostat = self.nature_thermostat
        
    def _calculate_expert_consensus(self, features: np.ndarray, base_proba: np.ndarray) -> np.ndarray:
        """
        🧠 UZMAN KONSENSÜSÜ (Collective Wisdom)
        
        En iyi 5 LoRA'nın (veya Hype Uzmanlarının) ortak fikrini hesapla.
        Bu, "Toplumun Sesi"dir.
        """
        population = self.evolution_manager.population
        if len(population) < 5:
            return base_proba
            
        # En iyi 5'i seç (Fitness'a göre)
        experts = sorted(population, key=lambda x: x.get_recent_fitness(), reverse=True)[:5]
        
        expert_probas = []
        for expert in experts:
            # Uzman tahmini
            try:
                p = expert.predict(features, base_proba, self.device)
                expert_probas.append(p)
            except:
                pass
        
        if not expert_probas:
            return base_proba
            
        # Ortalama al
        consensus = np.mean(expert_probas, axis=0)
        return consensus

    def _get_socially_adjusted_proba(self, lora, base_proba: np.ndarray, expert_consensus: np.ndarray) -> np.ndarray:
        """
        🎭 SOSYAL ADAPTASYON (Mizaca Göre!)
        
        LoRA'nın mizacına göre "Toplumun Sesi"ni ne kadar dinleyeceği.
        
        - Yüksek Social Intelligence: Uzmanları dinler (%70'e kadar)
        - Yüksek Contrarian (Karşıt): Uzmanların tersine gider!
        - Yüksek Independence: Sadece kendi bildiğini (base) okur
        """
        temp = lora.temperament
        
        social_score = temp.get('social_intelligence', 0.5)
        contrarian_score = temp.get('contrarian_score', 0.5)
        independence_score = temp.get('independence', 0.5)
        
        # 1. Sosyal Etki (Social Intelligence)
        # 0.5 -> %0 etki, 1.0 -> %50 etki
        social_weight = max(0.0, (social_score - 0.5) * 1.0)
        
        # 2. Karşıtlık Etkisi (Contrarian)
        # Uzmanlar A diyorsa, B'ye kayar (basitçe consensus'u ters çevirip normalize et)
        if contrarian_score > 0.7:
            # Ters consensus (basit yaklaşım: 1 - p, sonra normalize)
            inv_consensus = 1.0 - expert_consensus
            inv_consensus /= inv_consensus.sum()
            
            # Karşıtlık ağırlığı
            contrarian_weight = (contrarian_score - 0.7) * 1.0  # Max %30
            
            # Karşıt görüşü karıştır
            target_signal = (expert_consensus * (1 - contrarian_weight)) + (inv_consensus * contrarian_weight)
        else:
            target_signal = expert_consensus
            
        # 3. Bağımsızlık (Independence)
        # Yüksekse, sosyal etkiyi azaltır
        if independence_score > 0.6:
            social_weight *= (1.0 - (independence_score - 0.6) * 2.0)
        
        # Final Karışım
        # Base Proba (Ensemble) + Social Context
        final_input_proba = (base_proba * (1 - social_weight)) + (target_signal * social_weight)
        
        return final_input_proba

    def load_state(self):
        """💾 SİSTEM DURUMUNU YÜKLE (RESUME)"""
        import os
        print("\n" + "💾"*40)
        print("SİSTEM DURUMU YÜKLENİYOR (RESUME)...")
        print("💾"*40)
        
        try:
            # 1. Popülasyonu yükle
            if os.path.exists(self.paths['lora_population']):
                population = joblib.load(self.paths['lora_population'])
                self.evolution_manager.population = population
                print(f"   ✅ Popülasyon yüklendi: {len(population)} LoRA")
            else:
                print("   ⚠️ Popülasyon dosyası bulunamadı!")

            # 2. Match Count yükle (State pkl içinde olabilir veya ayrı)
            state_path = "evolution_state.pkl"
            if os.path.exists(state_path):
                state = joblib.load(state_path)
                self.evolution_manager.match_count = state.get('match_count', 0)
                # Diğer state'leri de yükle (gerekirse)
                print(f"   ✅ Match Count: {self.evolution_manager.match_count}")
            else:
                print("   ℹ️ State dosyası yok, maç sayısı 0'dan başlayabilir.")
                
            # 3. Collective Memory
            if os.path.exists(self.paths['collective_memory']):
                self.collective_memory.memory = joblib.load(self.paths['collective_memory'])
                print(f"   ✅ Ortak Hafıza: {len(self.collective_memory.memory)} kayıt")

            # 4. Hall of Fame / Mucizeler (Opsiyonel, zaten ayrı modülde ama burada da refresh edilebilir)
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ YÜKLEME HATASI: {e}")
            print("   ⚠️ Sıfırdan başlanıyor...")

    def load_state(self):
        """💾 SİSTEM DURUMUNU YÜKLE (RESUME)"""
        import os
        print("\n" + "💾"*40)
        print("SİSTEM DURUMU YÜKLENİYOR (RESUME)...")
        print("💾"*40)
        
        try:
            # 1. Popülasyonu yükle
            if os.path.exists(self.paths['lora_population']):
                population = joblib.load(self.paths['lora_population'])
                self.evolution_manager.population = population
                print(f"   ✅ Popülasyon yüklendi: {len(population)} LoRA")
            else:
                print("   ⚠️ Popülasyon dosyası bulunamadı!")

            # 2. Match Count yükle (State pkl içinde olabilir veya ayrı)
            state_path = "evolution_state.pkl"
            if os.path.exists(state_path):
                state = joblib.load(state_path)
                self.evolution_manager.match_count = state.get('match_count', 0)
                # Diğer state'leri de yükle (gerekirse)
                print(f"   ✅ Match Count: {self.evolution_manager.match_count}")
            else:
                print("   ℹ️ State dosyası yok, maç sayısı 0'dan başlayabilir.")
                
            # 3. Collective Memory
            if os.path.exists(self.paths['collective_memory']):
                self.collective_memory.memory = joblib.load(self.paths['collective_memory'])
                print(f"   ✅ Ortak Hafıza: {len(self.collective_memory.memory)} kayıt")

            # 4. Hall of Fame / Mucizeler (Opsiyonel, zaten ayrı modülde ama burada da refresh edilebilir)
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ YÜKLEME HATASI: {e}")
            print("   ⚠️ Sıfırdan başlanıyor...")

    def _get_physics_snapshot(self, lora):
        """O anki fizik durumunu yakala (Loglama için)"""
        # Langevin
        langevin_temp = getattr(lora, '_langevin_temp', self.langevin.T_base)
        nose_hoover_xi = self.langevin.xi.get(lora.id, 0.0)
        
        # Lazarus
        lazarus_lambda = getattr(lora, '_lazarus_lambda', 0.5)
        
        # Onsager
        om_action = getattr(lora, '_om_action', 0.0)
        
        # Ghost
        ghost_potential = getattr(lora, '_ghost_potential', 0.0)
        
        return {
             'langevin_temp': langevin_temp,
             'nose_hoover_xi': nose_hoover_xi,
             'kinetic_energy': langevin_temp * 0.5, # Basit yaklasim
             'om_action': om_action,
             'lazarus_lambda': lazarus_lambda,
             'ghost_potential': ghost_potential
        }

    def run(self, csv_path: str, start_match: int = 0, max_matches: int = None, results_csv: str = None):
        print("🌡️ Nature's Thermostat...")
        from lora_system.nature_thermostat import NatureThermostat
        self.nature_thermostat = NatureThermostat()
        
        print("\n" + "🌊"*40)
        print("PARÇACIK FİZİĞİ MOTORU!")
        print("🌊"*40)
        
        print("\n🌊 Langevin Dynamics (Stokastik SDE!)")
        self.langevin = langevin_dynamics
        
        print("🧟 Lazarus Potential (Fisher Info!)")
        self.lazarus = lazarus_potential
        
        print("🌀 Onsager-Machlup (Yörünge İntegrali!)")
        self.onsager = onsager_machlup
        
        print("🎭 Particle Archetypes!")
        self.particle_arch = particle_archetypes
        
        print("\n✅ PARÇACIK FİZİĞİ HAZIR!")
        print("🌊"*40 + "\n")
        
        print("\n✅ FİZİK MOTORU HAZIR!")
        print("🔬"*40 + "\n")
        
        # ============================================
        # SİSTEMLERİ BİRBİRİNE BAĞLA!
        # ============================================
        
        self.evolution_manager.experience_resistance = self.experience_resistance
        self.evolution_manager.ultra_mating = self.ultra_mating
        self.evolution_manager.nature_thermostat = self.nature_thermostat  # 🌡️ AKIŞKAN EVRİM İÇİN!
        self.trigger_system.adaptive_nature = self.adaptive_nature
        self.trigger_system.nature_thermostat = self.nature_thermostat  # 🌊 AKIŞKAN EŞİK İÇİN!
        
        print("\n✅ Tüm sistemler birbirine bağlandı! (Akışkan entegrasyon)")
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Veriyi yükle"""
        print(f"\n📂 Veri yükleniyor: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"   ✅ {len(df)} maç yüklendi")
        return df
    
    def _prepare_historical_features(self, home_history: Dict, away_history: Dict,
                                     h2h_history: Dict) -> np.ndarray:
        """
        🌊 TARİHSEL VERİLERİ FEATURE'LARA ÇEVİR!
        
        LoRA'lar bunu kullanarak öğrenecek!
        
        Returns:
            numpy array: 20+ tarihsel feature
        """
        features = []
        
        # HOME TAKIM TARİHİ (5 feature)
        features.append(home_history.get('avg_scored', 0.0))
        features.append(home_history.get('avg_conceded', 0.0))
        features.append(home_history.get('form', 0) / 5.0)  # Normalize (-1 ile +1 arası)
        features.append(home_history.get('avg_hype', 0.5))
        features.append(1.0 if home_history.get('hype_trend') == 'increasing' else 
                       (-1.0 if home_history.get('hype_trend') == 'decreasing' else 0.0))
        
        # AWAY TAKIM TARİHİ (5 feature)
        features.append(away_history.get('avg_scored', 0.0))
        features.append(away_history.get('avg_conceded', 0.0))
        features.append(away_history.get('form', 0) / 5.0)
        features.append(away_history.get('avg_hype', 0.5))
        features.append(1.0 if away_history.get('hype_trend') == 'increasing' else 
                       (-1.0 if away_history.get('hype_trend') == 'decreasing' else 0.0))
        
        # H2H (HEAD TO HEAD) TARİHİ (4 feature)
        features.append(h2h_history.get('team1_avg_goals', 0.0))  # Home (team1) ortalama gol
        features.append(h2h_history.get('team2_avg_goals', 0.0))  # Away (team2) ortalama gol
        features.append(h2h_history.get('team1_wins', 0) / max(1, h2h_history.get('matches_found', 1)))  # Home kazanma oranı
        features.append(h2h_history.get('draws', 0) / max(1, h2h_history.get('matches_found', 1)))  # Beraberlik oranı
        
        # DATA QUALITY (1 feature) - Ne kadar veri var?
        data_quality = (
            home_history.get('matches_found', 0) +
            away_history.get('matches_found', 0) +
            h2h_history.get('matches_found', 0)
        ) / 15.0  # Max 5+5+5=15, normalize
        features.append(min(1.0, data_quality))
        
        return np.array(features, dtype=np.float32)
    
    def prepare_features(self, row: pd.Series) -> np.ndarray:
        """
        Bir maçtan feature'ları çıkar
        (train_enhance_v2.py'deki aynı feature listesi)
        """
        feature_names = ['home_team_strength', 'away_team_strength', 'home_team_defense', 'away_team_defense',
            'home_xG', 'away_xG', 'xG_difference', 'home_form', 'away_form',
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
            'goal_ratio', 'xG_ratio', 'day_of_week', 'month',
            'home_support', 'away_support', 'support_difference', 'support_ratio',
            'sentiment_score', 'sentiment_positive', 'sentiment_negative',
            'total_tweets', 'log_total_tweets', 'high_hype', 'hype_score',
            'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
            'hype_favored_team', 'odds_favored_team', 'hype_odds_alignment',
            'home_hype_inflation', 'away_hype_inflation', 'hype_inflation_score',
            'high_engagement_match', 'hype_odds_discrepancy', 'tweets_odds_ratio',
            'odds_market_efficiency',
            'odds_entropy',
            'hype_odds_consistency_home', 'hype_odds_consistency_away', 'hype_odds_consistency_draw',
            'hype_odds_diff_home', 'hype_odds_diff_away', 'hype_odds_diff_draw',
            'home_hype_inflation_score', 'away_hype_inflation_score', 'total_hype_inflation',
            'tweets_odds_variance', 'tweets_per_odds_prob',
            'implied_prob_over_25', 'implied_prob_under_25', 'tweets_over_odds',
            'ah_home_implied_prob', 'ah_away_implied_prob',
            'hype_ah_consistency_home', 'hype_ah_consistency_away',
            'market_consensus']
        
        features = []
        for feat in feature_names:
            val = row.get(feat, 0.0)
            # NaN kontrolü
            if pd.isna(val):
                val = 0.0
            features.append(val)
        
        return np.array(features, dtype=np.float32)
    
    def run_match(self, match_data: pd.Series, match_idx: int) -> Dict:
        """
        Tek bir maç için tahmin + öğrenme döngüsü
        """
        home_team = match_data.get('home_team', 'Unknown')
        away_team = match_data.get('away_team', 'Unknown')
        match_date = match_data.get('date', 'Unknown')
        
        print(f"\n🔍 DEBUG: Maç #{match_idx} başlıyor: {home_team} vs {away_team}")
        
        self.logger.log_match_start(match_idx, home_team, away_team)
        
        # 🌊 ORTAK HAFIZADAN TARİHSEL VERİ ÇEK! (Akışkan!)
        print(f"📚 Ortak hafızadan tarihsel veri çekiliyor...")
        
        home_history = self.collective_memory.get_team_recent_history(
            home_team, last_n=5, current_match_idx=match_idx
        )
        away_history = self.collective_memory.get_team_recent_history(
            away_team, last_n=5, current_match_idx=match_idx
        )
        h2h_history = self.collective_memory.get_h2h_history(
            home_team, away_team, last_n=5, current_match_idx=match_idx
        )
        
        print(f"   📊 {home_team}: {home_history['matches_found']} maç bulundu")
        print(f"      Son 5 gol: {home_history['scored']} (avg: {home_history['avg_scored']:.1f})")
        print(f"      Form: {home_history['form']:+d}")
        print(f"      Hype trend: {home_history['hype_trend']}")
        
        print(f"   📊 {away_team}: {away_history['matches_found']} maç bulundu")
        print(f"      Son 5 gol: {away_history['scored']} (avg: {away_history['avg_scored']:.1f})")
        print(f"      Form: {away_history['form']:+d}")
        
        print(f"   🆚 H2H: {h2h_history['matches_found']} karşılaşma")
        print(f"      Skor geçmişi: {h2h_history['last_5_scores']}")
        
        # 🌊 GHOST FIELD ETKİSİ (Mizaç Dalgalanması!)
        # Maçtan önce hayaletler bazı LoRA'ların aklını çeler!
        if self.ghost_fields and len(self.ghost_fields.ghost_influence) > 0:
            # %10 ihtimalle hayalet fırtınası (daha çok etki)
            intensity = 0.3 if np.random.random() < 0.1 else 0.05
            self.ghost_fields.apply_temperament_perturbation(
                self.evolution_manager.population, 
                intensity=intensity
            )

        # 1) Özellikleri hazırla
        print(f"🔍 DEBUG: Özellikler hazırlanıyor...")
        base_features = self.prepare_features(match_data)  # 60 feature
        
        # 🌊 TARİHSEL VERİYİ HAZIRLA! (LoRA'lar için!)
        historical_features = self._prepare_historical_features(
            home_history, away_history, h2h_history
        )  # 15 feature
        
        print(f"🔍 DEBUG: Base features: {base_features.shape}, Historical: {historical_features.shape}")
        
        # 2) Base Ensemble tahmini (SADECE 60 feature!)
        print(f"🔍 DEBUG: Ensemble tahmin yapılıyor (60 feature)...")
        try:
            base_proba = self.ensemble.predict_proba(base_features.reshape(1, -1))[0]
            print(f"🔍 DEBUG: Ensemble başarılı! Proba: {base_proba}")
        except Exception as e:
            print(f"❌ ENSEMBLE HATASI: {e}")
            raise
        
        # 🌊 LoRA'lar için tam feature set (60 + 15 = 75)
        lora_features = np.concatenate([base_features, historical_features])
        print(f"🔍 DEBUG: base_features.shape = {base_features.shape}")
        print(f"🔍 DEBUG: historical_features.shape = {historical_features.shape}")
        print(f"🔍 DEBUG: lora_features.shape = {lora_features.shape} (base + historical)")
        print(f"🔍 DEBUG: base_proba.shape = {base_proba.shape}")
        
        base_pred_idx = np.argmax(base_proba)
        base_prediction = self.label_encoder.classes_[base_pred_idx]
        
        # 3) LoRA Ecosystem tahmini
        population = self.evolution_manager.population
        individual_predictions = []  # Her LoRA'nın tahmini
        
        print(f"🔍 DEBUG: LoRA tahmini başlıyor... Popülasyon: {len(population)}")
        
        if len(population) > 0:
            # LoRA features + base_proba birleştir (75 + 3 = 78 boyut)
            # lora_features = 60 base + 15 historical = 75
            # base_proba = 3
            # TOPLAM = 78 ✅
            combined_features = np.concatenate([lora_features, base_proba])
            print(f"🔍 DEBUG: Combined features shape: {combined_features.shape} (75 + 3 = 78)")
            
            # Her LoRA'dan tahmin al (detaylı log için)
            print(f"🔍 DEBUG: Bireysel LoRA tahminleri alınıyor...")
            
            # 🧠 UZMAN KONSENSÜSÜ HESAPLA (Bir kere)
            expert_consensus = self._calculate_expert_consensus(lora_features, base_proba)
            
            try:
                for i, lora in enumerate(population):
                    # 🎭 SOSYAL ADAPTASYON: Mizaca göre input'u değiştir!
                    # Bazıları "Toplumun Sesi"ni duyar, bazıları duymaz!
                    adjusted_base_proba = self._get_socially_adjusted_proba(lora, base_proba, expert_consensus)
                    
                    lora_pred = lora.predict(lora_features, adjusted_base_proba, self.device)  # 🌊 Sosyal context dahil!
                    individual_predictions.append((lora, lora_pred))
                    if i == 0:
                        print(f"🔍 DEBUG: İlk LoRA başarılı! Pred: {lora_pred}")
            except Exception as e:
                print(f"❌ LoRA.predict HATASI: {e}")
                raise
            
            print(f"🔍 DEBUG: Meta-LoRA aggregate ediliyor...")
            try:
                lora_proba, lora_info = self.meta_lora.aggregate_predictions(
                    combined_features, base_proba, population, self.device
                )
                print(f"🔍 DEBUG: Meta-LoRA başarılı! Proba: {lora_proba}")
            except Exception as e:
                print(f"❌ META-LoRA HATASI: {e}")
                raise
        else:
            lora_proba = base_proba
            lora_info = {}
        
        # 4) Chaotic Global
        print(f"🔍 DEBUG: Chaotic Global çalışıyor...")
        if self.chaotic_global and hasattr(self.chaotic_global, 'predict_with_global_context'):
            try:
                # CSV'den tüm maçları yükle (geçici)
                df_all = pd.read_csv('football_match_data.csv', low_memory=False)
                global_proba, context = self.chaotic_global.predict_with_global_context(
                    features, lora_proba, df_all, match_date
                )
                print(f"🔍 DEBUG: Chaotic başarılı! Proba: {global_proba}")
            except Exception as e:
                print(f"⚠️ Chaotic hatası: {e}, atlanıyor...")
                global_proba = lora_proba
        else:
            global_proba = lora_proba
        
        # 5) Incremental
        print(f"🔍 DEBUG: Incremental çalışıyor...")
        if self.incremental_learner and hasattr(self.incremental_learner, 'adjust_prediction'):
            try:
                final_proba = self.incremental_learner.adjust_prediction(features, global_proba)
                print(f"🔍 DEBUG: Incremental başarılı! Proba: {final_proba}")
            except Exception as e:
                print(f"⚠️ Incremental hatası: {e}, atlanıyor...")
                final_proba = global_proba
        else:
            final_proba = global_proba
        
        # 6) Final tahmin
        final_pred_idx = np.argmax(final_proba)
        final_prediction = self.label_encoder.classes_[final_pred_idx]
        confidence = float(final_proba[final_pred_idx])
        
        # Log prediction (skor bilgilerini sonra ekleyeceğiz)
        # Şimdilik sadece kazanan tahmini
        
        # Gerçek sonucu al (varsa)
        actual_result = self._get_actual_result(match_data)
        
        # 7) SKOR TAHMİNİ
        from lora_system.score_predictor import score_predictor
        
        # xG verisi var mı kontrol et (varsa kullan, yoksa LoRA tahminlerini kullan)
        home_xg = match_data.get('home_xG')
        away_xg = match_data.get('away_xG')
        
        # xG verisi yoksa veya NaN ise, None olarak işaretle
        if pd.isna(home_xg) or pd.isna(away_xg):
            home_xg = None
            away_xg = None
        
        # xG'den skor tahmini (varsa)
        if home_xg is not None and away_xg is not None:
            predicted_score = score_predictor.predict_score_from_xg(home_xg, away_xg)
        else:
            # xG yok, placeholder (LoRA tahminleri kullanılacak)
            predicted_score = None
        
        # Gerçek skoru al
        actual_score = self._get_actual_score(match_data)
        
        # Skor fitness hesapla (eğer gerçek skor varsa)
        score_fitness = None
        if actual_score is not None:
            score_fitness = score_predictor.calculate_score_fitness(predicted_score, actual_score)
            
        # Doğruluk kontrolü (correct)
        correct = False
        if actual_result:
            correct = (final_prediction == actual_result)
        
        result = {
            'match_idx': match_idx,
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date,
            'base_prediction': base_prediction,
            'base_proba': base_proba,
            'lora_proba': lora_proba,
            'lora_info': lora_info,  # 🧠 Meta-LoRA bilgisi (attention weights)
            'final_prediction': final_prediction,
            'predicted_winner': final_prediction,  # ✅ LOGGING İÇİN ALIAS
            'final_proba': final_proba,
            'confidence': confidence,
            'actual_result': actual_result,
            'actual_winner': actual_result,  # ✅ LOGGING İÇİN ALIAS
            'correct': correct,              # ✅ LOGGING İÇİN EKLENDİ!
            'population_size': len(population),
            # Skor bilgileri
            'home_xg': home_xg,
            'away_xg': away_xg,
            'predicted_score': predicted_score,
            'actual_score': actual_score,
            'score_fitness': score_fitness
        }
        
        # Eğer gerçek sonuç varsa öğrenme yap
        if actual_result is not None:
            self._learn_from_match(result, lora_features, match_data, individual_predictions)
        
        return result
    
    def _get_actual_result(self, match_data: pd.Series) -> Optional[str]:
        """
        Gerçek sonucu al (varsa)
        
        LABEL ENCODER FORMAT: 'away_win', 'draw', 'home_win'
        
        Öncelik:
        1. _actual_result (sonuç dosyasından eklenen)
        2. result sütunu
        3. goal sütunlarından hesapla
        """
        # 1) Sonuç dosyasından gelen gerçek sonuç (öncelikli!)
        if '_actual_result' in match_data and pd.notna(match_data['_actual_result']):
            result = str(match_data['_actual_result']).upper()
            # Label encoder formatına çevir
            if result == 'HOME':
                return 'home_win'
            elif result == 'AWAY':
                return 'away_win'
            elif result == 'DRAW':
                return 'draw'
            return result.lower().replace('_', '_')
        
        # 2) result sütunu varsa
        if 'result' in match_data and pd.notna(match_data['result']):
            result_val = str(match_data['result']).lower()
            # Eski format dönüşümü
            if 'home' in result_val:
                return 'home_win'
            elif 'away' in result_val:
                return 'away_win'
            elif 'draw' in result_val:
                return 'draw'
            return result_val
        
        # 3) Sonuç dosyasındaki skordan hesapla
        if '_actual_home_goals' in match_data and '_actual_away_goals' in match_data:
            home_goals = match_data['_actual_home_goals']
            away_goals = match_data['_actual_away_goals']
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                if home_goals > away_goals:
                    return 'home_win'
                elif away_goals > home_goals:
                    return 'away_win'
                else:
                    return 'draw'
        
        # 4) Skordan hesapla (home_goals ve away_goals)
        if 'home_goals' in match_data and 'away_goals' in match_data:
            home_goals = match_data['home_goals']
            away_goals = match_data['away_goals']
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                if home_goals > away_goals:
                    return 'home_win'
                elif away_goals > home_goals:
                    return 'away_win'
                else:
                    return 'draw'
        
        # 5) Eski format (home_scored, away_scored)
        if 'home_scored' in match_data and 'away_scored' in match_data:
            home_goals = match_data['home_scored']
            away_goals = match_data['away_scored']
            
            if pd.notna(home_goals) and pd.notna(away_goals):
                if home_goals > away_goals:
                    return 'home_win'
                elif away_goals > home_goals:
                    return 'away_win'
                else:
                    return 'draw'
        
        return None
    
    def _get_actual_score(self, match_data: pd.Series) -> Optional[tuple]:
        """
        Gerçek skoru al (home_goals, away_goals)
        
        Returns:
            (home_goals, away_goals) veya None
        """
        # 1) Sonuç dosyasından (_actual_home_goals, _actual_away_goals)
        if '_actual_home_goals' in match_data and '_actual_away_goals' in match_data:
            h = match_data['_actual_home_goals']
            a = match_data['_actual_away_goals']
            if pd.notna(h) and pd.notna(a):
                return (int(h), int(a))
        
        # 2) Normal home_goals, away_goals
        if 'home_goals' in match_data and 'away_goals' in match_data:
            h = match_data['home_goals']
            a = match_data['away_goals']
            if pd.notna(h) and pd.notna(a):
                return (int(h), int(a))
        
        # 3) Eski format (home_scored, away_scored)
        if 'home_scored' in match_data and 'away_scored' in match_data:
            h = match_data['home_scored']
            a = match_data['away_scored']
            if pd.notna(h) and pd.notna(a):
                return (int(h), int(a))
        
        return None
    
    def _learn_from_match(self, result: Dict, features: np.ndarray, match_data: pd.Series, individual_predictions: List):
        """
        Maçtan öğren (evrim + entropi + buffer)
        """
        # 🆕 LoRA skor tahminlerini baştan tanımla! (scope sorunu)
        lora_score_predictions = []
        score_fit = None  # 🆕 score_fit de baştan tanımla!
        
        actual_result = result['actual_result']
        final_prediction = result['final_prediction']
        base_proba = result['base_proba']
        final_proba = result['final_proba']
        population_size = result['population_size']
        
        # Debug: Format kontrol
        print(f"\n🔍 DEBUG: actual_result = '{actual_result}'")
        print(f"🔍 DEBUG: label_encoder.classes_ = {self.label_encoder.classes_}")
        print(f"🔍 DEBUG: final_prediction = '{final_prediction}'")
        
        # Doğru mu?
        correct = (final_prediction == actual_result)
        
        # actual_idx hesapla - GÜVENLI
        try:
            actual_idx = np.where(self.label_encoder.classes_ == actual_result)[0][0]
        except IndexError:
            print(f"❌ HATA: '{actual_result}' label_encoder'da bulunamadı!")
            print(f"   Beklenen formatlar: {list(self.label_encoder.classes_)}")
            # Fallback: en yakın eşleşmeyi bul
            if 'home' in actual_result.lower():
                actual_idx = list(self.label_encoder.classes_).index('home_win')
            elif 'away' in actual_result.lower():
                actual_idx = list(self.label_encoder.classes_).index('away_win')
            else:
                actual_idx = list(self.label_encoder.classes_).index('draw')
            print(f"   Fallback kullanıldı: {self.label_encoder.classes_[actual_idx]}")
        
        # Sürpriz hesapla
        surprise = 1.0 - final_proba[actual_idx]
        
        # Mistake severity
        if not correct:
            mistake_severity = final_proba[np.argmax(final_proba)]  # Ne kadar emindik?
        else:
            mistake_severity = 0.0
        
        print(f"\n{'='*80}")
        print(f"⚽ MAÇ #{result['match_idx']}")
        print(f"{'='*80}")
        print(f"📅 Tarih: {result['date']}")
        match_time = match_data.get('time', 'Bilinmiyor')
        print(f"⏰ Saat: {match_time}")
        print(f"🏟️  {result['home_team']} vs {result['away_team']}")
        print(f"{'='*80}")
        
        # Ensemble tahmini
        base_pred_idx = np.argmax(base_proba)
        base_pred = self.label_encoder.classes_[base_pred_idx]
        print(f"\n📊 ENSEMBLE TAHMİNİ:")
        for i, cls in enumerate(self.label_encoder.classes_):
            print(f"   {cls}: {base_proba[i]*100:.1f}%")
        print(f"   → Tahmin: {base_pred.upper()}")
        
        # DETAYLI LoRA TAHMİNLERİ
        self.logger.log_detailed_predictions(
            self.evolution_manager.population,
            individual_predictions,
            actual_result,
            self.label_encoder
        )
        
        # Final tahmin - NET FORMAT
        print(f"\n🔮 TAHMİN:")
        winner_text = "EV SAHİBİ" if 'home' in final_prediction.lower() else ("DEPLASMAN" if 'away' in final_prediction.lower() else "BERABERE")
        print(f"   • Kim kazanır? {winner_text}")
        print(f"   • Güven: {result['confidence']*100:.0f}%")
        
        # 🧠 LoRA KONSENSUS (Ortak Fikir)
        if len(individual_predictions) > 0:
            lora_votes = {'HOME': 0, 'DRAW': 0, 'AWAY': 0, 'home_win': 0, 'draw': 0, 'away_win': 0}
            for lora, proba in individual_predictions:
                pred_idx = np.argmax(proba)
                pred = self.label_encoder.classes_[pred_idx]
                
                # Case-insensitive mapping
                pred_upper = pred.upper() if isinstance(pred, str) else pred
                if pred_upper in lora_votes:
                    lora_votes[pred_upper] += 1
                elif pred in ['home_win', 'HOME_WIN', 'HOME']:
                    lora_votes['HOME'] += 1
                elif pred in ['draw', 'DRAW']:
                    lora_votes['DRAW'] += 1
                elif pred in ['away_win', 'AWAY_WIN', 'AWAY']:
                    lora_votes['AWAY'] += 1
                else:
                    # Fallback
                    if 'home' in str(pred).lower():
                        lora_votes['HOME'] += 1
                    elif 'away' in str(pred).lower():
                        lora_votes['AWAY'] += 1
                    else:
                        lora_votes['DRAW'] += 1
            
            # Sadece büyük harfli anahtarları topla
            total_votes = lora_votes['HOME'] + lora_votes['DRAW'] + lora_votes['AWAY']
            print(f"\n🧠 LoRA ORTAK FİKRİ ({len(individual_predictions)} LoRA):")
            for outcome in ['HOME', 'DRAW', 'AWAY']:
                votes = lora_votes[outcome]
                percentage = (votes / total_votes * 100) if total_votes > 0 else 0
                outcome_text = "EV SAHİBİ" if outcome == 'HOME' else ("BERABERE" if outcome == 'DRAW' else "DEPLASMAN")
                bar = "█" * int(percentage / 5)  # Her 5% için bir blok
                print(f"   {outcome_text:12s}: {votes:3d} LoRA ({percentage:5.1f}%) {bar}")
        
        print(f"\n📥 GERÇEK SONUÇ:")
        actual_winner_text = "EV SAHİBİ" if 'home' in actual_result.lower() else ("DEPLASMAN" if 'away' in actual_result.lower() else "BERABERE")
        print(f"   • Kazanan: {actual_winner_text}")
        
        # SKOR TAHMİNİ VE KARŞILAŞTIRMA
        if result.get('predicted_score') and result.get('actual_score'):
            actual_score = result['actual_score']
            score_fit = result.get('score_fitness', {})
            
            # 🆕 LoRA'LARIN SKOR TAHMİNLERİNİ TOPLA!
            # lora_score_predictions, yukarıda (satır 915+) toplanmış olmalı
            if len(lora_score_predictions) > 0:
                # En çok tekrar eden skor tahmini (çoğunluk)
                from collections import Counter
                score_counts = Counter(lora_score_predictions)
                most_common_score = score_counts.most_common(1)[0][0]
                pred_score = most_common_score
                
                # Kaç LoRA bu skoru tahmin etti?
                vote_count = score_counts[most_common_score]
                total_loras = len(lora_score_predictions)
                vote_percentage = (vote_count / total_loras * 100) if total_loras > 0 else 0
                
                print(f"   • LoRA'ların skor tahmini: {pred_score[0]}-{pred_score[1]} ({vote_count}/{total_loras} LoRA, %{vote_percentage:.0f})")
                
                # En çok tahmin edilen ilk 3 skor
                top_3_scores = score_counts.most_common(3)
                if len(top_3_scores) > 1:
                    print(f"   • Diğer tahminler:")
                    for (home, away), count in top_3_scores[1:]:
                        pct = (count / total_loras * 100) if total_loras > 0 else 0
                        print(f"      - {home}-{away}: {count} LoRA (%{pct:.0f})")
            elif result.get('predicted_score'):
                # Fallback: xG'den (eğer xG varsa)
                pred_score = result['predicted_score']
                print(f"   • Skor tahmini (xG): {pred_score[0]}-{pred_score[1]}")
            else:
                # xG de yok, skor tahmini yok
                print(f"   • Skor tahmini: Veri yok (xG eksik)")
            
            print(f"   • Maç sonucu: {actual_score[0]}-{actual_score[1]}")
            
            if score_fit:
                print(f"\n🎯 SKOR FITNESS:")
                if score_fit.get('exact_score', 0) > 0:
                    print(f"   ✅ TAM SKOR! (+5 puan)")
                elif score_fit.get('goal_difference', 0) > 0:
                    print(f"   ✅ GOL FARKI DOĞRU! (+2 puan)")
                elif score_fit.get('close_score', 0) > 0:
                    print(f"   ✅ YAKIN SKOR! (+1 puan)")
                
                if score_fit.get('correct_winner', 0) > 0:
                    print(f"   ✅ KAZANAN DOĞRU! (+1 puan)")
                
                print(f"   📊 Toplam Fitness: {score_fit.get('total_fitness', 0):.1f} puan")
        
        print(f"\n🎯 SONUÇ: {'✅ DOĞRU TAHMİN!' if correct else '❌ YANLIŞ TAHMIN!'}")
        if score_fit and score_fit.get('total_fitness', 0) > 0:
            print(f"   📈 Toplam Puan: {score_fit.get('total_fitness', 0):.0f}")
        print(f"{'='*80}")
        
        # Population'ı al (wallet ve diğer işlemler için gerekli!)
        population = self.evolution_manager.population
        
        # 🌐 HER LoRA'NIN DÜŞÜNCESİNİ KAYDET (ORTAK HAFIZA + LOG)
        # 🆕 LoRA skor tahminlerini ÖNCE topla! (yukarıda kullanılıyor)
        lora_thoughts = []
        lora_score_predictions = []  # 🆕 TÜM LoRA SKOR TAHMİNLERİ!
        
        # İLK PASS: Sadece skor tahminlerini topla
        for lora, proba in individual_predictions:
            try:
                # xG varsa kullan, yoksa None geç
                home_xg = result.get('home_xg')
                away_xg = result.get('away_xg')
                
                if home_xg is not None and away_xg is not None:
                    lora_score = lora.predict_score(home_xg, away_xg)
                    lora_score_predictions.append(lora_score)
            except:
                pass  # Hata varsa atla
        
        # İKİNCİ PASS: Detaylı düşünceleri kaydet
        for lora, proba in individual_predictions:
            pred_idx = proba.argmax()
            pred_class = self.label_encoder.classes_[pred_idx]
            lora_confidence = float(proba[pred_idx])
            lora_correct = (pred_idx == actual_idx)
            
            # Kişilik tipi belirle
            temp = lora.temperament
            if temp['independence'] > 0.7:
                temp_type = 'Bağımsız'
            elif temp['social_intelligence'] > 0.7:
                temp_type = 'Sosyal Zeki'
            elif temp['herd_tendency'] > 0.6:
                temp_type = 'Sürü Psikolojisi'
            elif temp['contrarian_score'] > 0.6:
                temp_type = 'Karşıt Görüş'
            else:
                temp_type = 'Dengeli'
            
            # ⚽ LoRA SKOR TAHMİNİ (zaten yukarıda toplandı)
            try:
                home_xg = result.get('home_xg')
                away_xg = result.get('away_xg')
                
                if home_xg is not None and away_xg is not None:
                    lora_score = lora.predict_score(home_xg, away_xg)
                else:
                    lora_score = None  # xG yok
            except:
                lora_score = None  # Hata
            
            # Eski fitness'a göre skor yorumu
            old_fitness = lora.get_recent_fitness()
            
            # 🧠 REASONING: LoRA neden bu tahmini yaptı?
            reasoning = f"Güven: {lora_confidence:.2f}, Uzmanlık: {getattr(lora, 'specialization', 'Yok')}"
            
            # 🧠 LEARNING: LoRA bu maçtan ne öğrendi?
            if lora_correct:
                learning = f"✅ Doğru tahmin! (Fitness {old_fitness:.3f} artacak)"
            else:
                learning = f"❌ Yanlış tahmin. (Fitness {old_fitness:.3f} düşecek)"
                # Pattern bazlı öğrenme
                match_patterns = self.spec_tracker.detect_match_patterns(match_data)
                if match_patterns:
                    learning += f" Pattern: {', '.join(match_patterns)}"
            
            # 🎯 ADJUSTMENTS: Kendi ayarlamaları
            adjustments = []  # Şimdilik boş, ilerisi için
            
            # 🏆 İTİBAR HESAPLA! (Algısal kimlik!)
            from lora_system.reputation_system import reputation_system
            reputation = reputation_system.calculate_reputation(
                lora,
                population,
                all_loras_ever=self.all_loras_ever,
                match_count=result['match_idx']
            )
            
            # 🎭 DUYGU ARKETİPİ
            emotional_archetype = getattr(lora, 'emotional_archetype', 'Dengeli')
            
            # 🔬 FİZİK ARKETİPİ (Frequency + Amplitude!)
            from lora_system.physics_based_archetypes import physics_archetypes
            physics_archetype = physics_archetypes.determine_archetype_from_physics(lora)
            
            # 🌊 PARÇACIK FİZİĞİ VERİLERİNİ HESAPLA!
            # (Bu veriler her maç sonrası güncellenir!)
            
            # 1) Lazarus Λ hesapla
            try:
                lazarus_data = self.lazarus.calculate_lazarus_lambda(lora)
                lora._lazarus_lambda = lazarus_data['lambda']
            except Exception as e:
                # HATA YAZDIRMA! Neden hesaplanamıyor?
                if result['match_idx'] % 50 == 0:  # Sadece ara sıra yazdır
                    print(f"      ⚠️ Lazarus Lambda hesaplanamadı ({lora.name[:20]}): {e}")
                lora._lazarus_lambda = 0.5  # Default
            
            # 2) Onsager-Machlup eylemi hesapla (yörünge integrali)
            try:
                om_data = self.onsager.calculate_action(lora)
                lora._om_action = om_data['action']
            except:
                lora._om_action = 0.0  # Default
            
            # 3) Langevin sıcaklığı (adaptif!)
            # Fitness bazlı basit yaklaşım (şimdilik!)
            if lora.get_recent_fitness() < 0.5:
                lora._langevin_temp = 0.02  # Yüksek! (Keşif!)
            elif lora.get_recent_fitness() < 0.7:
                lora._langevin_temp = 0.01  # Orta
            else:
                lora._langevin_temp = 0.005  # Düşük! (İstikrar!)
            
            # 4) Nosé-Hoover sürtünme (şimdilik default)
            lora._nose_hoover_xi = getattr(lora, '_nose_hoover_xi', 0.0)
            
            # 5) Kinetik enerji (basit yaklaşım: fitness değişim hızı!)
            if len(lora.fitness_history) >= 2:
                fitness_velocity = abs(lora.fitness_history[-1] - lora.fitness_history[-2])
                lora._kinetic_energy = fitness_velocity
            else:
                lora._kinetic_energy = 0.0
            
            # 6) Ghost potansiyel hesapla
            try:
                # Basit yaklaşım: Parametreleri al
                lora_params = lora.get_all_lora_params()
                if isinstance(lora_params, dict):
                    # Dict ise tensor'e çevir
                    param_list = []
                    for k, v in lora_params.items():
                        param_list.append(v.flatten())
                    lora_params = torch.cat(param_list)
                
                ghost_pot = self.ghost_fields.calculate_ghost_potential(lora_params)
                lora._ghost_potential = ghost_pot
            except:
                lora._ghost_potential = 0.0  # Default
            
            # 7) Parçacık Arketipi belirle
            particle_arch_data = self.particle_arch.get_archetype_from_lora(lora)
            particle_archetype = particle_arch_data['primary_archetype']
            
            # 🔬 TES SKORLARI (Her maç hesaplanacak sonra!)
            tes_scores = {}  # Şimdilik boş, öğrenme sonrası hesaplanacak
            
            # ⚡ LIFE ENERGY
            life_energy = getattr(lora, 'life_energy', 1.0)
            
            lora_thoughts.append({
                'lora_id': lora.id,
                'lora_name': lora.name,
                'prediction': pred_class,
                'confidence': lora_confidence,
                'predicted_score': lora_score,
                'old_fitness': old_fitness,
                'temperament_type': temp_type,
                'temperament_values': temp,
                'emotional_archetype': emotional_archetype,
                'physics_archetype': physics_archetype,  # 🔬 Fizik arketip!
                'particle_archetype': particle_archetype,  # 🌊 YENİ: Parçacık arketip!
                'result': 'CORRECT' if lora_correct else 'WRONG',
                'specialization': getattr(lora, 'specialization', None),
                'reasoning': reasoning,
                'learning': learning,
                'adjustments': adjustments,
                'reputation': reputation,
                'authority_weight': reputation['authority_weight'],
                'tes_scores': tes_scores,  # 🔬 TES!
                'life_energy': life_energy  # ⚡ Enerji!
            })
            
            # 🎒 HER LoRA KENDİ WALLET'INA YAZ! (EN ÖNEMLİ!)
            wallet = self.wallet_manager.get_or_create_wallet(lora, population)
            wallet.log_prediction(
                match_num=result['match_idx'],
                home_team=result['home_team'],
                away_team=result['away_team'],
                prediction=pred_class,
                confidence=lora_confidence,
                predicted_score=lora_score,  # ⚽ SKOR TAHMİNİ!
                actual=actual_result,
                actual_score=result.get('actual_score')  # ⚽ GERÇEK SKOR!
            )
        
        self.collective_memory.record_match(
            match_idx=result['match_idx'],
            home_team=result['home_team'],
            away_team=result['away_team'],
            match_date=result['date'],
            lora_predictions=lora_thoughts,
            actual_result=actual_result,
            actual_score=result.get('actual_score')
        )
        
        # 🔥 HYPE VERİLERİNİ ORTAK HAFIZAYA EKLE!
        total_tweets = match_data.get('total_tweets', 0.0)
        sentiment_score = match_data.get('sentiment_score', 0.0)
        home_support = match_data.get('home_support', 0.5)
        away_support = match_data.get('away_support', 0.5)
        
        self.collective_memory.update_match_hype_data(
            result['match_idx'],
            total_tweets,
            sentiment_score,
            home_support,
            away_support
        )
        
        # 🏆 TAKIM UZMANLIK KAYIT! (Her LoRA için)
        actual_score = result.get('actual_score')
        home_support = match_data.get('home_support', 0.5)
        
        if actual_score is not None:  # Gerçek skor varsa
            actual_home_goals, actual_away_goals = actual_score
            
            for lora, proba in individual_predictions:
                # Tahmin edilen kazanan
                pred_idx = proba.argmax()
                predicted_winner_encoded = self.label_encoder.classes_[pred_idx]
                
                # Kazanan formatını düzelt
                if 'home' in predicted_winner_encoded.lower():
                    predicted_winner = 'HOME'
                elif 'away' in predicted_winner_encoded.lower():
                    predicted_winner = 'AWAY'
                else:
                    predicted_winner = 'DRAW'
                
                # Gerçek kazanan
                if 'home' in actual_result.lower():
                    actual_winner = 'HOME'
                elif 'away' in actual_result.lower():
                    actual_winner = 'AWAY'
                else:
                    actual_winner = 'DRAW'
                
                # Tahmin edilen skor (xG varsa!)
                home_xg = result.get('home_xg')
                away_xg = result.get('away_xg')
                
                if home_xg is not None and away_xg is not None:
                    predicted_home_goals, predicted_away_goals = lora.predict_score(home_xg, away_xg)
                else:
                    # xG yok, ortak hafızadan tahmin et
                    predicted_home_goals, predicted_away_goals = 1, 1  # Placeholder
                
                # Takım uzmanlık kaydet!
                self.team_spec_manager.record_match_prediction(
                    lora=lora,
                    home_team=result['home_team'],
                    away_team=result['away_team'],
                    predicted_winner=predicted_winner,
                    actual_winner=actual_winner,
                    predicted_home_goals=predicted_home_goals,
                    predicted_away_goals=predicted_away_goals,
                    actual_home_goals=actual_home_goals,
                    actual_away_goals=actual_away_goals,
                    home_support=home_support,
                    match_idx=result['match_idx']
                )
                
                # 🌍 GENEL uzmanık kaydet!
                self.global_spec_manager.record_global_prediction(
                    lora=lora,
                    predicted_winner=predicted_winner,
                    actual_winner=actual_winner,
                    predicted_home_goals=predicted_home_goals,
                    predicted_away_goals=predicted_away_goals,
                    actual_home_goals=actual_home_goals,
                    actual_away_goals=actual_away_goals,
                    home_support=home_support,
                    match_idx=result['match_idx']
                )
        
        # 📊 GENEL LOG DOSYASINA YAZ (match_results.log)
        
        # 16) TAHMİN RESULT LOGLA (Result logger)
        # Değişkenleri Result'tan al
        lora_info = result.get('lora_info', {})
        correct = result.get('correct', False)
        
        # Context hazırla
        nature_context = {
            'temperature': self.nature_thermostat.temperature,
            'chaos': self.evolution_manager.adaptive_nature.state['chaos'],
            'active_bonds': len(self.evolution_manager.social_network.network.edges()) if hasattr(self.evolution_manager.social_network, 'network') else 0
        }

        self.match_logger.log_match(
            match_idx=result['match_idx'],
            home_team=match_data['home_team'],
            away_team=match_data['away_team'],
            match_date=match_data['date'],
            match_time=match_data.get('time', '00:00'),
            predicted_winner=result['predicted_winner'],
            predicted_score=result['predicted_score'],
            actual_winner=result['actual_winner'],
            actual_score=result['actual_score'],
            winner_correct=result['correct'],
            score_fitness=result.get('score_fitness', {}),
            confidence=result.get('confidence', 0.0),
            population_size=len(population),
            base_proba=result.get('base_proba', None),
            final_proba=result.get('final_proba', None),
            lora_thoughts=lora_info.get('individual_predictions', []),
            nature_context=nature_context  # ✅ EKLENDİ!
        )
        
        # 1) DOĞAYA ETKİ
        if correct:
            self.nature_system.lora_succeeded(quality=result['confidence'], population_size=population_size)
        else:
            self.nature_system.lora_made_mistake(severity=mistake_severity, population_size=population_size)
        
        # 2) TETİKLEYİCİLERİ GÜNCELLE (🌊 AKIŞKAN PARAMETRELER İLE!)
        # Popülasyon entropisi hesapla (tahmin çeşitliliği)
        if len(individual_predictions) > 1:
            # individual_predictions format: [(lora, proba_array), ...]
            probs_dist = [proba for lora, proba in individual_predictions]
            # Tahminlerin çeşitliliğini ölç
            std_devs = np.std(probs_dist, axis=0)
            population_entropy = float(np.mean(std_devs))  # 0-1 arası normalize
        else:
            population_entropy = 0.5  # Default
        
        # Lazarus Lambda ortalaması hesapla
        if hasattr(self, 'lazarus_potential'):
            lazarus_values = []
            for lora in self.evolution_manager.population[:20]:  # İlk 20 LoRA yeterli
                if hasattr(lora, '_lazarus_lambda'):
                    lazarus_values.append(lora._lazarus_lambda)
            lazarus_avg = float(np.mean(lazarus_values)) if lazarus_values else 0.5
        else:
            lazarus_avg = 0.5
        
        nature_event = None  # Legacy trigger system removed
        
        # 3) DOĞA TEPKİSİ KONTROL
        if nature_event is None:
            nature_event = self.nature_system.check_nature_response(population_size)
        
        # 4) DOĞA OLAYI VARSA UYGULA (FREN KONTROLÜ)
        if nature_event:
            # Fren kontrolü
            can_trigger, brake_reason = self.advanced_mechanics.check_nature_event_allowed(
                result['match_idx'],
                nature_event.get('severity', 0.5)
            )
            
            if can_trigger:
                print(f"\n🌍 {nature_event['message']}")
                self._apply_nature_event(nature_event, result['match_idx'])
                
                # Olayı kaydet (fren için)
                self.advanced_mechanics.register_nature_event(
                    result['match_idx'],
                    nature_event['type'],
                    nature_event.get('severity', 0.5)
                )
            else:
                print(f"\n🛑 DOĞA OLAYI ENGELLENDİ!")
                print(f"   Sebep: {brake_reason}")
                print(f"   Doğa Enerjisi: {self.advanced_mechanics.get_nature_energy(result['match_idx'])*100:.0f}%")
        
        # 🌊 DİNAMİK THRESHOLD: AdaptiveNature tarafından yönetilir (legacy removed)
        # self.nature_system.dynamic_population_threshold = ...
        pass
        
        # 5) ENTROPİ (SOĞUMA)
        entropy_effects = self.nature_system.apply_entropy(self.evolution_manager.population)
        
        # 6) HER LoRA ÖĞRENME
        population = self.evolution_manager.population
        
        print(f"\n📚 ÖĞRENME SÜRECİ:")
        print(f"{'='*80}")
        
        correct_loras = []
        wrong_loras = []
        
        # 📚 POPULATION HISTORY: Tahmin kayıtları için individual_predictions'ı kullan
        # (Eğer individual_predictions boşsa, burada hesapla)
        if not individual_predictions:
            # individual_predictions yoksa, burada hesapla
            individual_predictions = []
            for lora in population:
                lora_proba = lora.predict(features, base_proba, self.device)
                individual_predictions.append((lora, lora_proba))
        
        for lora in population:
            # individual_predictions'dan bul (daha önce hesaplandıysa)
            lora_proba = None
            for pred_lora, pred_proba in individual_predictions:
                if pred_lora.id == lora.id:
                    lora_proba = pred_proba
                    break
            
            # Yoksa hesapla
            if lora_proba is None:
                lora_proba = lora.predict(features, base_proba, self.device)
            
            lora_pred_idx = np.argmax(lora_proba)
            lora_correct = (lora_pred_idx == actual_idx)
            lora_confidence = lora_proba[lora_pred_idx]
            
            old_fitness = lora.get_recent_fitness()
            
            # Fitness güncelle
            lora.update_fitness(lora_correct, lora_confidence)
            
            new_fitness = lora.get_recent_fitness()
            fitness_change = new_fitness - old_fitness
            
            # Kayıt
            if lora_correct:
                correct_loras.append((lora, old_fitness, new_fitness, fitness_change))
            else:
                wrong_loras.append((lora, old_fitness, new_fitness, fitness_change))
            
            # 🎭 MİZAÇ EVRİMİ (Dynamic Temperament)
            # Travma kontrolü (Loss > 2.0 ise travma!)
            # Loss henüz hesaplanmadı ama confidence üzerinden tahmin edebiliriz
            # Yanlış ve yüksek güven = Travma!
            is_trauma = (not lora_correct) and (lora_confidence > 0.8)
            fake_loss = 2.0 if is_trauma else (0.5 if not lora_correct else 0.1)
            
            lora.evolve_temperament(lora_correct, fake_loss, is_trauma)
            
            # Online learning (buffer ile)
            # 🌊 DURUMSAL BUFFER ÖRNEKLEME! (Situational Sampling)
            # Maçın durumuna göre geçmişten benzer anıları hatırla!
            
            # Kriterleri belirle
            criteria = {}
            
            # Hype durumu?
            match_hype = result.get('hype_score', 0)
            if match_hype > 0.7 or (isinstance(match_hype, (int, float)) and match_hype > 50000):
                criteria['high_hype'] = True
            
            # Gol farkı? (Farklı yenilgi/galibiyet)
            goal_diff = abs(result.get('home_score', 0) - result.get('away_score', 0))
            if goal_diff >= 3:
                criteria['high_goal_diff'] = True
            
            # Sürpriz? (Loss yüksekse sürprizdir)
            # (Bunu henüz bilmiyoruz, ama genel loss'a bakabiliriz)
            
            # Örnekle!
            buffer_samples = self.buffer.sample_situational(
                criteria, 
                batch_size=self.config['learning']['buffer_batch_size']
            )
            
            # Yeni maç + buffer
            new_example = {
                'features': features,
                'base_proba': base_proba,
                'actual_class_idx': actual_idx
            }
            
            batch = [new_example] + buffer_samples
            
            # 🧠 META-ADAPTIF LEARNING RATE! (Her LoRA farklı hız!)
            # İlk kez öğreniyorsa başlat (mizaç bazlı!)
            if lora.id not in self.meta_learning.learning_rates:
                lora_lr = self.meta_learning.initialize_learning_rate(
                    lora, base_lr=self.config['learning']['learning_rate']
                )
            else:
                lora_lr = self.meta_learning.get_optimal_lr_for_lora(lora)
            
            # Öğren (LoRA'ya özel learning rate!)
            from lora_system.lora_adapter import OnlineLoRALearner
            
            # 👻 GHOST FIELDS: Ataya saygı terimi ekle!
            ancestor_loss = 0.0
            if len(self.ghost_fields.ghost_parameters) > 0 and hasattr(lora, 'parents'):
                ancestor_loss = self.ghost_fields.calculate_ancestor_respect_loss(lora, lora.parents)
                lora._ancestor_respect_loss = ancestor_loss  # Kaydet (log için!)
            
            learner = OnlineLoRALearner(lora, learning_rate=lora_lr, device=self.device)
            
            # 🧬 KNOWLEDGE DISTILLATION (ÇAĞ ATLAMA!)
            # Eğer LoRA yeni ve başarısızsa, bir "Master"dan ders alsın
            distillation_loss = 0.0
            if lora.get_recent_fitness() < 0.6 and len(lora.match_history) < 50:
                teacher = self.distiller.find_best_teacher(population, lora)
                if teacher:
                    # Distillation step
                    # Not: Bu, learner.learn_batch'den önce veya sonra yapılabilir
                    # Burada direkt optimizer step çağrılıyor, dikkat!
                    distillation_loss = self.distiller.distill_knowledge(
                        lora, teacher,
                        features, base_proba, actual_idx,
                        learner.optimizer
                    )

            # 🔍 DEBUG: Parametre değişimini ölç (Öğrenme Kanıtı!)
            # Önceki parametrelerin kopyasını al
            old_params = {}
            for name, p in lora.named_parameters():
                if p.requires_grad:
                    old_params[name] = p.detach().clone()
            
            # Öğrenme adımı
            loss = learner.learn_batch(batch)
            
            # 🕸️ SIEVE KAYDI (Davranış analizi)
            lora_pred_vector = lora.predict(features, base_proba, self.device)
            self.background_sieve.record_behavior(lora.id, lora_pred_vector, lora_correct, abs(1.0 - lora_confidence))

            # 🔍 DEBUG: Parametre değişimini hesapla
            param_change = 0.0
            count = 0
            for name, p in lora.named_parameters():
                if p.requires_grad and name in old_params:
                    diff = torch.norm(p - old_params[name]).item()
                    param_change += diff
                    count += 1
            
            lora._last_param_change = param_change  # Kaydet
            lora._last_loss = loss
            
            # Total loss (Match + Ancestor respect!)
            total_loss = loss + ancestor_loss
            
            # 🧠 KİŞİSEL HAFIZA (Subjective Memory)
            # Her LoRA kendi günlüğünü tutar
            if not hasattr(lora, 'personal_memory_buffer'):
                from lora_system.lora_adapter import PersonalMemory
                lora.personal_memory_buffer = PersonalMemory()
            
            # Bu maçı hatırlamalı mıyım? (Neural Gate)
            if lora.personal_memory_buffer.should_remember(features, loss):
                lora.personal_memory_buffer.add({
                    'features': features,
                    'base_proba': base_proba,
                    'actual_class_idx': actual_idx,
                    'loss': loss,
                    'match_idx': self.evolution_manager.match_count
                })
            
            # 🎭 MİZAÇ EVRİMİ (Neural Reaction)
            # ReactionNet karar verir: Bu olay beni nasıl değiştirmeli?
            lora.evolve_temperament(
                correct=(pred_class == actual_idx),
                loss=loss,
                is_trauma=(loss > 2.0) # Travma eşiği
            )
            
            # 🦋 KELEBEK ETKİSİ TETİKLEME (Her öğrenme adımında şans eseri veya olay bazlı)
            # Eğer büyük bir kayıp (travma) veya büyük bir değişim varsa tetikle
            if loss > 1.5 or lora._last_param_change > 0.5:
                # Olay büyüklüğü: Loss veya değişim miktarı ile orantılı
                magnitude = min(1.0, (loss / 5.0) + (lora._last_param_change / 2.0))
                self.butterfly_effect.trigger_effect(lora, magnitude, population)

            # 🌊 LANGEVIN DYNAMICS: Stokastik parametre güncellemesi!
            # Öğrenme sonrası parametrelere fiziksel gürültü ekle!
            try:
                # LoRA parametrelerini al (lora_A ve lora_B!)
                lora_params_dict = {}
                for name, module in lora.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        lora_params_dict[f"{name}.lora_A"] = module.lora_A.data
                        lora_params_dict[f"{name}.lora_B"] = module.lora_B.data
                
                # Gradient'ları simüle et (loss'tan türet)
                gradients = {}
                for param_name, param_tensor in lora_params_dict.items():
                    # Basit gradient yaklaşımı: Loss'un parametre büyüklüğüne orantılı
                    grad_magnitude = loss * 0.01  # Küçük gradyan simülasyonu
                    gradients[param_name] = torch.randn_like(param_tensor) * grad_magnitude
                
                # Langevin sıcaklığını al (daha önce hesaplanmış!)
                langevin_temp = getattr(lora, '_langevin_temp', 0.01)
                
                # Langevin Dynamics ile parametre güncellemesi!
                langevin_result = self.langevin.update_parameters(
                    lora,
                    gradients,
                    temperature=langevin_temp
                )
                
                # Parametrelere Langevin gürültüsünü ekle!
                noise_scale = langevin_result['noise_magnitude'] * 0.1  # Kontrollü gürültü
                for name, module in lora.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        # Brownian hareket: √(2T) dW
                        # ✅ Device'ı kontrol et!
                        device = module.lora_A.data.device
                        noise_A = torch.randn_like(module.lora_A.data, device=device) * noise_scale
                        noise_B = torch.randn_like(module.lora_B.data, device=device) * noise_scale
                        
                        module.lora_A.data += noise_A
                        module.lora_B.data += noise_B
                
                # Langevin sonuçlarını kaydet
                lora._langevin_temp_effective = langevin_result['T_eff']
                lora._langevin_noise = langevin_result['noise_magnitude']
                
                # Nosé-Hoover xi değerini kaydet (adaptif sürtünme!)
                if hasattr(self.langevin, 'xi') and lora.id in self.langevin.xi:
                    lora._nose_hoover_xi = self.langevin.xi[lora.id]
            except Exception as e:
                # Hata olursa sessizce devam et
                pass
            
            # 🔬 MASTER FLUX: Life Energy güncelle!
            # Darwin + Einstein + Newton!
            # 🛡️ ÖLÜMSÜZLÜK KORUMASINI UYGULA!
            top_5_cache = getattr(self.team_spec_manager, 'top_5_cache', None)
            
            energy_update = self.master_flux.update_life_energy(
                lora,
                population,
                lora_proba,
                final_proba,
                lora_correct,
                fisher_data=None,  # Lazy hesaplama
                dt=1.0,
                top_5_cache=top_5_cache  # 🛡️ Ölümsüzlük için!
            )
            
            # 🔬 TES SKORLARINI KAYDET! (energy_update içinde var!)
            tes_scores = {
                'darwin': energy_update.get('darwin', 0.0),
                'einstein': energy_update.get('einstein', 0.0),
                'newton': energy_update.get('newton', 0.0),
                'total_tes': energy_update.get('darwin', 0.0) + 
                           self.master_flux.λ_einstein * energy_update.get('einstein', 0.0) +
                           self.master_flux.λ_newton * energy_update.get('newton', 0.0)
            }
            
            # LoRA'ya da kaydet!
            lora._tes_scores = tes_scores
            
            # Her 10 maçta learning rate'i adapte et!
            if result['match_idx'] % 10 == 0 and len(lora.fitness_history) >= 10:
                new_lr, reason = self.meta_learning.adapt_learning_rate(
                    lora,
                    lora.fitness_history[-10:],
                    current_lr=lora_lr
                )
                if abs(new_lr - lora_lr) > 0.00001:  # Değişti mi?
                    pass  # Sessizce adapte et
            
            # 🌊 FLUID TEMPERAMENT: Mizaç evrimi! (Ornstein-Uhlenbeck!)
            try:
                # Bu maçta ne yaşadı? (Event listesi oluştur!)
                events = []
                
                if lora_correct:
                    # Başarı streak kontrolü
                    if len(lora.fitness_history) >= 3:
                        recent_fitness = lora.fitness_history[-3:]
                        if all(f > 0.6 for f in recent_fitness):
                            events.append('success_streak')
                else:
                    # Travma kontrolü
                    if lora_confidence > 0.8:  # Çok emindi ama yanıldı!
                        events.append('trauma')
                
                # Rank değişikliği kontrolü (basit yaklaşım)
                if len(lora.fitness_history) >= 2:
                    old_rank = sorted([l.get_recent_fitness() for l in population], reverse=True).index(lora.get_recent_fitness()) if len(population) > 1 else 0
                    # Yeni rank'ı hesaplamak için geçici olarak eski fitness'ı kullan
                    # (Gerçek rank değişikliği için daha sonra hesaplanacak)
                    if lora_correct and old_rank > 5:
                        events.append('rank_rise')
                    elif not lora_correct and old_rank < len(population) - 5:
                        events.append('rank_drop')
                
                # 📚 POPULATION HISTORY: Her tahmini kaydet!
                try:
                    lora_prediction = individual_predictions[i][1] if i < len(individual_predictions) else None
                    if lora_prediction is not None:
                        pred_label = self.label_encoder.classes_[np.argmax(lora_prediction)] if isinstance(lora_prediction, np.ndarray) else str(lora_prediction)
                        actual_label = result.get('actual_result', 'Unknown')
                        confidence = float(np.max(lora_prediction)) if isinstance(lora_prediction, np.ndarray) else 0.5
                        
                        self.population_history.record_prediction(
                            lora,
                            result['match_idx'],
                            pred_label,
                            actual_label,
                            lora_correct,
                            confidence
                        )
                except Exception as e:
                    # Sessizce devam et (debug için)
                    if result['match_idx'] % 10 == 0:
                        print(f"      ⚠️ Population History prediction kaydı hatası: {e}")
                
                # Doğa olayı var mı?
                if nature_event and nature_event.get('type') in ['kara_veba', 'deprem', 'felaket']:
                    events.append('disaster')
                
                # Mizacı evrimleştir!
                self.fluid_temperament.evolve_temperament(
                    lora,
                    match_count=result['match_idx'],
                    events=events
                )
            except Exception as e:
                # Hata olursa sessizce devam et
                pass
            
            # 🌊 PARÇACIK FİZİĞİ VERİLERİNİ TEKRAR HESAPLA! (Öğrenme sonrası!)
            # Çünkü parametreler değişti, fiziksel özellikler de değişmeli!
            try:
                # 1) Lazarus Λ (güncellenmiş parametrelerle!)
                try:
                    lazarus_data = self.lazarus.calculate_lazarus_lambda(lora)
                    lora._lazarus_lambda = lazarus_data['lambda']
                except Exception as laz_err:
                    print(f"⚠️ {lora.name} - Lazarus Lambda hesaplanamadı: {laz_err}")
                    lora._lazarus_lambda = 0.5  # Default
                
                # 2) Onsager-Machlup eylemi (güncellenmiş yörünge!)
                try:
                    om_data = self.onsager.calculate_action(lora)
                    lora._om_action = om_data['action']
                except Exception as om_err:
                    # print(f"⚠️ {lora.name} - OM Action hesaplanamadı: {om_err}")
                    lora._om_action = 0.0  # Default
                
                # 3) Ghost potansiyel (güncellenmiş parametrelerle!)
                try:
                    lora_params = lora.get_all_lora_params()
                    if isinstance(lora_params, dict):
                        param_list = []
                        for k, v in lora_params.items():
                            param_list.append(v.flatten())
                        lora_params = torch.cat(param_list)
                    
                    ghost_pot = self.ghost_fields.calculate_ghost_potential(lora_params)
                    lora._ghost_potential = ghost_pot
                    
                    # 👻 GHOST FIELD ETKİSİNİ KAYDET!
                    lora._ghost_effect_data = {
                        'ghost_potential': ghost_pot,
                        'closest_ancestor': self.ghost_fields.get_closest_ancestor(lora),
                        'ancestor_respect_loss': getattr(lora, '_ancestor_respect_loss', 0.0)
                    }
                    
                except Exception as ghost_err:
                    # print(f"⚠️ {lora.name} - Ghost Potential hesaplanamadı: {ghost_err}")
                    lora._ghost_potential = 0.0  # Default
                    lora._ghost_effect_data = None
                
                # 4) Kinetik enerji güncelle (fitness değişim hızı!)
                if len(lora.fitness_history) >= 2:
                    fitness_velocity = abs(lora.fitness_history[-1] - lora.fitness_history[-2])
                    lora._kinetic_energy = fitness_velocity
                else:
                    lora._kinetic_energy = 0.0
                
                # 5) Parçacık Arketipi güncelle!
                try:
                    particle_arch_data = self.particle_arch.get_archetype_from_lora(lora)
                    lora._particle_archetype = particle_arch_data['primary_archetype']
                except Exception as arch_err:
                    # print(f"⚠️ {lora.name} - Particle Archetype hesaplanamadı: {arch_err}")
                    lora._particle_archetype = "Unknown"
                
                # 6) 🔄 PT SYNC! (Eğer uzmanlığı varsa tüm kopyalarını güncelle!)
                try:
                    self.sync_manager.sync_all_lora_copies(lora)
                except Exception as sync_err:
                    # Sessizce devam et
                    pass
                
            except Exception as e:
                print(f"⚠️ {lora.name} - Parçacık fiziği hesaplanamadı: {e}")
        
        # ÖĞRENME SONUÇLARI
        if len(correct_loras) > 0:
            print(f"\n✅ DOĞRU TAHMİN EDENLER ({len(correct_loras)}/{len(population)} LoRA):")
            for lora, old_fit, new_fit, change in correct_loras[:5]:  # İlk 5
                print(f"   • {lora.name}: Fitness {old_fit:.3f} → {new_fit:.3f} ({change:+.3f})")
            if len(correct_loras) > 5:
                print(f"   ... ve {len(correct_loras)-5} LoRA daha")
        
        if len(wrong_loras) > 0:
            print(f"\n❌ YANLIŞ TAHMİN EDENLER ({len(wrong_loras)}/{len(population)} LoRA):")
            for lora, old_fit, new_fit, change in wrong_loras[:5]:  # İlk 5
                print(f"   • {lora.name}: Fitness {old_fit:.3f} → {new_fit:.3f} ({change:+.3f})")
            if len(wrong_loras) > 5:
                print(f"   ... ve {len(wrong_loras)-5} LoRA daha")
        
        # 📚 POPULATION HISTORY: Her LoRA'nın tahminini kaydet!
        actual_result = result.get('actual_result', 'Unknown')
        for lora, pred in individual_predictions:
            try:
                # Tahmin bilgisi
                if isinstance(pred, np.ndarray):
                    pred_idx = np.argmax(pred)
                    pred_label = self.label_encoder.classes_[pred_idx]
                    confidence = float(np.max(pred))
                else:
                    pred_label = str(pred)
                    confidence = 0.5
                
                # Doğru mu?
                is_correct = (lora in [l for l, _, _, _ in correct_loras])
                
                # 📚 Kaydet! (match_idx çok önemli!)
                try:
                    self.population_history.record_prediction(
                        lora,
                        result['match_idx'],  # ✅ Doğru match_idx!
                        pred_label,
                        actual_result,
                        is_correct,
                        confidence
                    )
                except Exception as pred_err:
                    if result['match_idx'] % 10 == 0:
                        print(f"         ⚠️ Tahmin kaydı hatası: {pred_err}")
            except Exception as e:
                # Sessizce devam et (sadece her 50 maçta bir debug)
                if result['match_idx'] % 50 == 0:
                    print(f"      ⚠️ Population History tahmin kaydı hatası ({lora.name[:20]}): {e}")
        
        # ✅ Güvenli bölme (popülasyon boş olabilir!)
        if len(population) > 0:
            correct_pct = len(correct_loras) / len(population) * 100
            print(f"\n📊 BİLİNME YÜZDESI: %{correct_pct:.1f} LoRA doğru bildi")
        else:
            print(f"\n⚠️ POPÜLASYON BOŞ! Tüm LoRA'lar öldü!")
        print(f"{'='*80}")
        
        # 👻 GHOST FIELD ETKİLERİNİ LOGLA!
        if len(self.ghost_fields.ghost_parameters) > 0:
            affected_loras = []
            
            for lora in population:
                if hasattr(lora, '_ghost_effect_data') and lora._ghost_effect_data:
                    ghost_data = lora._ghost_effect_data
                    
                    # Etki büyüklüğünü hesapla
                    effect_magnitude = ghost_data['ghost_potential']
                    
                    # Etki yönünü belirle (ataya yakın mı uzak mı?)
                    if ghost_data['closest_ancestor']:
                        ancestor_id, distance = ghost_data['closest_ancestor']
                        # Düşük mesafe = ataya çekilme (pull)
                        # Yüksek mesafe = atadan uzaklaşma (push)
                        effect_direction = 'pull' if distance < 1.0 else 'push'
                    else:
                        effect_direction = 'neutral'
                    
                    affected_loras.append({
                        'lora_name': lora.name,
                        'lora_id': lora.id,
                        'ghost_potential': ghost_data['ghost_potential'],
                        'closest_ancestor': ghost_data['closest_ancestor'],
                        'ancestor_respect_loss': ghost_data['ancestor_respect_loss'],
                        'effect_magnitude': effect_magnitude,
                        'effect_direction': effect_direction
                    })
            
            # Güçlü hayalet bul
            strongest_ghost = None
            if len(self.ghost_fields.ghost_influence) > 0:
                strongest_id = max(self.ghost_fields.ghost_influence.items(), 
                                  key=lambda x: x[1])
                strongest_ghost = strongest_id
            
            # Log et!
            self.ghost_logger.log_ghost_effects(
                match_idx=result['match_idx'],
                affected_loras=affected_loras,
                total_ghosts=len(self.ghost_fields.ghost_parameters),
                strongest_ghost=strongest_ghost
            )
            
            # Kısa özet print
            if len(affected_loras) > 0:
                print(f"\n👻 GHOST FIELD ETKİLERİ:")
                print(f"   • {len(self.ghost_fields.ghost_parameters)} hayalet aktif")
                print(f"   • {len(affected_loras)} LoRA etkilendi")
                top_3 = sorted(affected_loras, key=lambda x: x['effect_magnitude'], reverse=True)[:3]
                for i, lora_data in enumerate(top_3, 1):
                    direction_emoji = "⬅️" if lora_data['effect_direction'] == 'pull' else ("➡️" if lora_data['effect_direction'] == 'push' else "↔️")
                    print(f"   {i}. {lora_data['lora_name']}: {direction_emoji} Etki {lora_data['effect_magnitude']:.4f}")
        
        # 🔬 TES SKORLARINI HESAPLA VE ORTAK HAFIZAYA EKLE!
        print(f"\n🔬 TES SKORLARI HESAPLANIYOR...")
        
        # Her LoRA için TES skorunu al (öğrenme döngüsünde hesaplanmış!)
        for thought in lora_thoughts:
            lora_id = thought['lora_id']
            lora_obj = next((l for l in population if l.id == lora_id), None)
            
            if lora_obj and hasattr(lora_obj, '_tes_scores'):
                # Öğrenme döngüsünde hesaplanmış TES skorunu kullan!
                thought['tes_scores'] = lora_obj._tes_scores
            else:
                # Fallback: tes_scoreboard kullan
                from lora_system.tes_scoreboard import tes_scoreboard
                if lora_obj:
                    tes_data = tes_scoreboard.calculate_tes_score(
                        lora_obj,
                        population,
                        collective_memory=self.collective_memory.memory
                    )
                    thought['tes_scores'] = tes_data
        
        # İlk 3 LoRA'nın TES skorunu göster
        print(f"\n   📊 İLK 3 LoRA TES SKORLARI:")
        for i, thought in enumerate(lora_thoughts[:3]):
            tes = thought.get('tes_scores', {})
            if tes:
                print(f"   • {thought['lora_name']}: TES={tes['total_tes']:.3f} (D:{tes['darwin']:.2f} E:{tes['einstein']:.2f} N:{tes['newton']:.2f})")
        
        if len(lora_thoughts) > 3:
            print(f"   ... ve {len(lora_thoughts)-3} LoRA daha hesaplandı")
        
        # 🔍 DİNAMİK UZMANLIK GÜNCELLEMESİ (AKIŞKAN!)
        # Feature kombinasyonlarını analiz et (Kodlanmış pattern YOK!)
        match_feature_combos = self.dynamic_spec.analyze_match_features(match_data)
        
        for lora in population:
            lora_was_correct = any(l[0].id == lora.id for l in correct_loras)
            
            # Pattern keşif güncelle (LoRA kendi pattern'ini bulur!)
            self.dynamic_spec.update_lora_pattern_discovery(lora, match_feature_combos, lora_was_correct)
            
            # Her 20 maçta uzmanlık kontrol et
            if result['match_idx'] % 20 == 0:
                new_spec = self.dynamic_spec.detect_specialization(lora, min_samples=15)
                
                old_spec = getattr(lora, 'specialization', None)
                
                if new_spec and new_spec != old_spec:
                    # UZMANLIK DEĞİŞTİ! (Dinamik keşif!)
                    lora.specialization = new_spec
                    self.logger.log_specialization_change(lora, old_spec, new_spec, result['match_idx'])
        
        # CÜZDAN GÜNCELLEMELERİ (arka planda)
        predictions_dict = {}
        for lora, old_fit, new_fit, change in correct_loras + wrong_loras:
            lora_pred_idx = np.argmax([p for l, p in individual_predictions if l.id == lora.id][0])
            lora_pred = self.label_encoder.classes_[lora_pred_idx]
            lora_conf = [p[lora_pred_idx] for l, p in individual_predictions if l.id == lora.id][0]
            
            predictions_dict[lora.id] = (lora_pred, lora_conf)
            
            # Cüzdan güncelle
            wallet = self.wallet_manager.get_or_create_wallet(lora, population)
            wallet.log_prediction(
                result['match_idx'],
                result['home_team'],
                result['away_team'],
                lora_pred,
                lora_conf,
                actual_result
            )
            wallet.log_learning(result['match_idx'], old_fit, new_fit)
        
        # Her 10 maçta: Tüm cüzdanları tam güncelle
        if result['match_idx'] % 10 == 0:
            self.wallet_manager.update_all_wallets(population, result['match_idx'])
        
        # Her 20 maçta: Evrim geçiren LoRA'ları özetle
        if result['match_idx'] % 20 == 0 and result['match_idx'] > 0:
            self.logger.log_evolved_loras_summary(population)
        
        # 7) BUFFER'A EKLE
        self.buffer.add({
            'features': features,
            'base_proba': base_proba,
            'lora_proba': result['lora_proba'],
            'actual_class_idx': actual_idx,
            'actual_result': actual_result,
            'loss': mistake_severity,
            'surprise': surprise,
            'hype': match_data.get('total_tweets', 0),
            'goal_diff': abs(match_data.get('home_scored', 0) - match_data.get('away_scored', 0)),
            'match_date': result['date'],
            'home_team': result['home_team'],
            'away_team': result['away_team'],
            'league': match_data.get('league', 'Unknown'),
            'predicted_class': final_prediction,
            'correct': correct
        })
        
        # 🌡️ NATURE THERMOSTAT GÜNCELLEME (AKIŞKAN DOĞA!)
        # =================================================
        if hasattr(self, 'nature_thermostat') and len(individual_predictions) > 0:
            # 1. Popülasyon entropisini hesapla
            # Her LoRA'nın tahmin olasılıklarını al
            all_probas = np.array([p for _, p in individual_predictions])
            pop_entropy = self.nature_thermostat.calculate_population_entropy(all_probas)
            
            # 2. Termostatı güncelle
            thermo_stats = self.nature_thermostat.update_temperature(pop_entropy)
            
            # 3. Logla (Eğitici!)
            temp = thermo_stats['temperature']
            status = thermo_stats['status']
            
            print(f"\n🌡️ DOĞA TERMOSTATI (Maç #{result['match_idx']}):")
            print(f"   • Popülasyon Entropisi: {pop_entropy:.3f} (Çeşitlilik)")
            
            # Sıcaklık Barı
            bar_len = 20
            filled = int(temp * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"   • Sıcaklık: {temp:.3f} [{bar}] {status}")
            
            if temp > 0.75:
                print(f"   🔥 UYARI: Doğa çok ısındı! LoRA'lar çok başarılı/benzer.")
                print(f"      → Zorluk artırılıyor (Kaotik olaylar tetiklenecek!)")
            elif temp < 0.25:
                print(f"   ❄️ BİLGİ: Doğa soğudu. LoRA'lar zorlanıyor.")
                print(f"      → Zorluk azaltılıyor (İyileşme fırsatı)")
            
            # 4. Doğa Evrimi (Her 50 maçta bir kontrol)
            if result['match_idx'] % 50 == 0 and hasattr(self, 'adaptive_nature'):
                evo_msg = self.adaptive_nature.evolve_nature(
                    self.evolution_manager.population, 
                    result['match_idx']
                )
                if evo_msg:
                    print(f"\n{evo_msg}\n")

        # 8) EVRİM ADIMI (⚠️ SOY AZALMA ALARM!)
        from lora_system.population_alarm import population_alarm
        
        alarm_info = population_alarm.check_alarm_level(len(population))
        evolution_events = self.evolution_manager.post_match_update(alarm_info=alarm_info)
        
        # Alarm seviyesi değiştiyse logla
        if alarm_info['level'] != 'GREEN':
            self.logger._write_log(f"\n⚠️ {alarm_info['message']} (Popülasyon: {alarm_info['population']})\n")
        
        # 9) EVOLUTION LOGGER + WALLET UPDATES
        for event in evolution_events:
            if event['type'] == 'birth':
                child = next((l for l in population if l.name == event['child']), None)
                parent1 = next((l for l in population if l.name == event.get('parent1')), None)
                parent2 = next((l for l in population if l.name == event.get('parent2')), None)
                
                if child:
                    self.logger.log_birth(child, parent1, parent2, birth_type='crossover')
                    
                    # Çocuğun cüzdanını oluştur
                    child_wallet = self.wallet_manager.get_or_create_wallet(child, population)
                    child_wallet.log_evolution_event(
                        result['match_idx'],
                        "DOĞUM",
                        f"Anne: {parent1.name if parent1 else 'Yok'}, Baba: {parent2.name if parent2 else 'Yok'}"
                    )
                    
                    # Ebeveynlerin cüzdanına da kaydet
                    if parent1:
                        p1_wallet = self.wallet_manager.get_or_create_wallet(parent1, population)
                        p1_wallet.log_evolution_event(result['match_idx'], "ÇOCUK", f"{child.name} doğdu")
                    
                    if parent2:
                        p2_wallet = self.wallet_manager.get_or_create_wallet(parent2, population)
                        p2_wallet.log_evolution_event(result['match_idx'], "ÇOCUK", f"{child.name} doğdu")
            
            elif event['type'] == 'death':
                # 💀 LoRA ÖLDÜ - MUCİZE KONTROLÜ YAP!
                dead_lora = event.get('lora_obj')
                
                if dead_lora:
                    # 📚 TÜM ZAMANLAR KAYDINA EKLE (ÖLÜLER DE SAYILIR!)
                    death_reason = event.get('death_reason', 'Bilinmiyor')
                    
                    # ✅ LoRA PARAMETRELERİNİ KAYDET! (Export için gerekli!)
                    self.all_loras_ever[dead_lora.id] = {
                        'lora': dead_lora,  # Objeyi saklıyoruz (export için)
                        'lora_params': dead_lora.get_all_lora_params(),  # ✅ Parametreleri kaydet!
                        'final_fitness': dead_lora.get_recent_fitness(),
                        'death_match': result['match_idx'],
                        'death_reason': death_reason,
                        'age': result['match_idx'] - dead_lora.birth_match,
                        'alive': False  # 💀 ÖLDÜ!
                    }
                    
                    # 👻 HAYALET OLARAK KAYDET! (Ghost Fields!)
                    influence_score = dead_lora.get_recent_fitness()
                    # TES skoru al
                    tes_score = getattr(dead_lora, '_tes_scores', {}).get('total_tes', 0.5)
                    
                    self.ghost_fields.register_ghost(dead_lora, influence_score, tes_score)
                    
                    # 👻 GHOST KAYIT LOGLA!
                    self.ghost_logger.log_ghost_registration(
                        dead_lora_name=dead_lora.name,
                        dead_lora_id=dead_lora.id,
                        influence_score=influence_score,
                        tes_score=tes_score,
                        match_idx=result['match_idx']
                    )
                    
                    # Mucize kriterlerini kontrol et (ÇOKLU UZMANLIK DAHİL!)
                    # Ölümsüzlük hesapla (uzmanlık sayısı)
                    from lora_system.death_immunity_system import calculate_death_immunity
                    top_5_cache = getattr(self.team_spec_manager, 'top_5_cache', {})
                    _, specialization_count = calculate_death_immunity(dead_lora, top_5_cache)
                    
                    miracle_check = self.miracle_system.check_miracle_criteria(
                        dead_lora, 
                        result['match_idx'],
                        specialization_count=specialization_count  # 🌟 Uzmanlık sayısı!
                    )
                    
                    if miracle_check['is_miracle']:
                        # 🏆 MUCİZE! KAYDET!
                        miracle_id = self.miracle_system.save_miracle(dead_lora, result['match_idx'], miracle_check)
                        
                        # Logger'a kaydet
                        self.logger.log_miracle_saved(dead_lora, result['match_idx'], miracle_id, miracle_check)
                        
                        # 🆕 MUCİZE HALL TXT GÜNCELLE!
                        miracle_hall_manager.generate_miracle_hall_txt(match_count=result['match_idx'])
                        
                        # Evolution logger'a da event ekle
                        self.logger.log_death(dead_lora, reason="miracle_ascension", 
                                            lucky_survived=False, 
                                            death_reason_detail=death_reason)
                        
                        # Wallet'a not ekle
                        wallet = self.wallet_manager.get_or_create_wallet(dead_lora, population)
                        wallet.log_evolution_event(
                            result['match_idx'],
                            "💀 ÖLÜM",
                            f"Sebep: {death_reason}"
                        )
                    
                    # 💔 SOSYAL KAYIP TEPKİLERİ! (Akışkan!)
                    # Bu LoRA'ya bağlı olanlar tepki versin!
                    from lora_system.psychological_responses import psychological_responses
                    
                    for survivor_lora in population:
                        if hasattr(survivor_lora, 'social_bonds'):
                            if dead_lora.id in survivor_lora.social_bonds:
                                bond_strength = survivor_lora.social_bonds[dead_lora.id]
                                
                                if bond_strength > 0.3:  # Anlamlı bağ varsa
                                    # Tepki ver! (Mizaç bazlı!)
                                    loss_response = psychological_responses.react_to_loss(
                                        survivor_lora,
                                        dead_lora.id,
                                        bond_strength,
                                        loss_type="death"
                                    )
                                    
                                    # Travma ekle
                                    if hasattr(survivor_lora, 'trauma_history'):
                                        survivor_lora.trauma_history.append({
                                            'type': 'social_loss',
                                            'match': result['match_idx'],
                                            'severity': loss_response['trauma_gain'],
                                            'lost_lora': dead_lora.name
                                        })
                                    
                                    # Motivasyon değişimi
                                    if not hasattr(survivor_lora, '_current_motivation'):
                                        survivor_lora._current_motivation = 0.0
                                    survivor_lora._current_motivation += loss_response['motivation_change']
                                    
                                    # Bağı sil (artık yok!)
                                    del survivor_lora.social_bonds[dead_lora.id]
                        wallet.log_evolution_event(
                            result['match_idx'],
                            "🏆 HALL OF FAME",
                            f"Mucize LoRA olarak kaydedildi! Puan: {miracle_check['total_points']}/100"
                        )
                    else:
                        # Mucize değil, sadece ölüm kaydı
                        wallet = self.wallet_manager.get_or_create_wallet(dead_lora, population)
                        wallet.log_evolution_event(
                            result['match_idx'],
                            "💀 ÖLÜM",
                            f"Sebep: {death_reason} | Final Fitness: {dead_lora.get_recent_fitness():.3f}"
                        )
            
            elif event['type'] == 'spontaneous_birth':
                alien = next((l for l in population if l.name == event['lora']), None)
                if alien:
                    self.logger.log_birth(alien, birth_type='spontaneous')
                    
                    alien_wallet = self.wallet_manager.get_or_create_wallet(alien, population)
                    alien_wallet.log_evolution_event(
                        result['match_idx'],
                        "SPONTANE DOĞUM",
                        "Hiçlikten doğdu! 👽"
                    )
        
        # 10) GOALLESS DRIFT + TÜM ZAMANLAR KAYDI (YAŞ SİSTEMİ!)
        for lora in population:
            self.goalless_system.update_goalless_lora(lora, population, current_match=result['match_idx'])
            
            # 📚 TÜM ZAMANLAR KAYDINA EKLE/GÜNCELLE (YAŞAYANLAR!)
            self.all_loras_ever[lora.id] = {
                'lora': lora,
                'final_fitness': lora.get_recent_fitness(),
                'current_match': result['match_idx'],
                'age': result['match_idx'] - lora.birth_match,
                'alive': True  # ⭐ YAŞIYOR!
            }
        
        # 10.5) UZMANLIK TESPİTİ VE EVRİMİ (Advanced Categorization!)
        # Artık Multi-Dimensional Categorization kullanıyoruz!
        
        # 🕸️ SIEVE ANALİZİ ÇALIŞTIR (Her 10 maçta)
        if result['match_idx'] % 10 == 0:
            tribes = self.background_sieve.run_sieve(population)
            # 🔥 KABİLE EĞİTİMİ (Toplu Eğitim)
            if tribes:
                self.tribe_trainer.train_tribes(tribes, self.buffer)

        for lora in population:
            lora_correct = any(l[0].id == lora.id for l in correct_loras)
            
            # Gelişmiş kategori güncellemesi (Dominant expertise otomatik set edilir)
            weights = self.advanced_categorization.update_lora_expertise(
                lora, 
                match_data,  # raw match data yeterli, internal olarak extract eder
                lora_correct
            )
            
            # Yeni uzmanlık (otomatik set edildiği için buradan okuyabiliriz)
            new_spec = getattr(lora, 'specialization', None)
            
            # Loglama (Değişim varsa)
            # Not: AdvancedCategorization içinde log mekanizması var ama burada da loglayabiliriz
            # Şimdilik LivingLoRAsReporter ve diğer sistemler lora.specialization'ı kullanacak.
        
        # 🧬 KOLEKTİF ÖĞRENME (Sürü Zekası)
        # Global hata oranına göre tüm popülasyonu hafifçe düzelt
        if not correct: # Sürü (konsensus) yanıldıysa
            global_error = mistake_severity
            self.collective_learner.collective_backprop(
                population, features, base_proba, actual_idx, global_error
            )

        # META-LoRA bilgisini al (başta tanımla!)
        lora_info = result.get('lora_info', {})  # ✅ Result'tan al!
        
        # 10.7) HİBERNATION KONTROLÜ (KOLONİ MANTIĞI!)
        # Her 10 maçta bir kontrol et: Zayıf/orta LoRA'lar uyur (ölmez!)
        if result['match_idx'] % 10 == 0 and len(population) >= 30:
            print(f"\n🌙 HİBERNATION KONTROLÜ...")
            hibernated_count = self.advanced_mechanics.hibernation.check_and_hibernate(
                population, 
                lora_info.get('attention_weights', []),
                match_idx=result['match_idx']  # 🔍 Debug için
            )
            if hibernated_count > 0:
                print(f"   😴 {hibernated_count} LoRA uyudu (diske kaydedildi)")
                # Log için uyuyan LoRA'ları bul (son N tane)
                # Not: check_and_hibernate zaten log'a yazıyor, tekrar loglamaya gerek yok
        
        # 11) META-LoRA KARAR SÜRECİ
        if 'attention_weights' in lora_info and len(lora_info['attention_weights']) > 0:
            self.logger.log_meta_lora_decision(
                lora_info['attention_weights'],
                population,
                top_k=5
            )
        
        # 12) DOĞA NABZI GRAFİĞİ
        self.logger.log_nature_graph(self.nature_system.nature, population_size)
        
        # 13) POPÜLASYON GRAFİĞİ (her 10 maçta)
        if result['match_idx'] % 10 == 0:
            self.logger.log_population_graph(self.logger.population_history, last_n=50)
        
        # 14) POPÜLASYON SNAPSHOT
        self.logger.log_population_snapshot(population)
        
        # 15) SOSYAL BAĞLAR (rastgele 2 LoRA'nın)
        if len(population) >= 2:
            sample_loras = np.random.choice(population, size=min(2, len(population)), replace=False)
            for sample_lora in sample_loras:
                if hasattr(sample_lora, 'social_bonds') and len(sample_lora.social_bonds) > 0:
                    self.logger.log_social_bonds(sample_lora, population, top_k=3)
        
        # 🕸️ SOSYAL AĞ GÖRSELLEŞTİRME (Her 10 maçta)
        if result['match_idx'] % 10 == 0:
            self.social_visualizer.export_snapshot(
                self.evolution_manager.social_network, 
                population, 
                result['match_idx']
            )
            # 🕸️ MENTOR AĞACI RAPORU
            self.social_visualizer.export_mentor_tree(
                self.evolution_manager.social_network, 
                population, 
                result['match_idx']
            )
        
        # ⏰ 16) AKILLI UYANMA KONTROLÜ! (5 Faktör!)
        # Popülasyon düşükse veya uzman eksikse uyandır!
        if result['match_idx'] % 10 == 0:
            # Son zamanda felaket oldu mu?
            # Son zamanda felaket oldu mu?
            recent_disaster = False # Legacy trigger system removed
            
            awakened, wake_reason = self.advanced_mechanics.hibernation.intelligent_wake_up(
                population,
                match_data=match_data.to_dict() if hasattr(match_data, 'to_dict') else {},
                attention_weights=lora_info.get('attention_weights', []),
                recent_disaster=recent_disaster
            )
            
            if awakened:
                population.extend(awakened)
                self.evolution_manager.population.extend(awakened)
                print(f"\n⏰ AKILLI UYANMA: {len(awakened)} LoRA uyandırıldı!")
                print(f"   📋 Sebep: {wake_reason}")
        
        # 🧠 17) KİŞİSEL ÖĞRENME SİSTEMİ (YENİ!)
        # Her LoRA:
        # 1. Kendi öğrenmesini kaydet
        # 2. Başkalarının öğrenmelerini oku
        # 3. Mizacına göre yorumla ve benimse/reddet!
        
        from lora_system.temperament_learning import temperament_learning
        
        print(f"\n🧠 KİŞİSEL ÖĞRENME:")
        print(f"{'='*80}")
        
        for lora in population[:3]:  # İlk 3 LoRA göster (çok uzun olmasın)
            # 1) Başkalarının öğrenmelerini al
            others_learning = self.collective_memory.get_others_learning(
                lora.id,
                last_n_matches=20  # Son 20 maç
            )
            
            if len(others_learning) == 0:
                continue  # Henüz yeterli veri yok
            
            # 2) Mizaç bazlı yorumla
            interpretation = temperament_learning.interpret_others_learning(
                lora,
                others_learning,
                self.collective_memory
            )
            
            # 3) Kendi hafızasına kaydet
            lora.personal_memory['observed_others'][result['match_idx']] = {
                'adopted': interpretation['adopted_learnings'],
                'rejected': interpretation['rejected_learnings'],
                'insight': interpretation['personal_insights']
            }
            
            # 4) Konsola yazdır
            print(f"\n   🎭 {lora.name} ({lora.temperament.get('independence', 0.5):.2f} bağımsızlık):")
            print(f"      • {len(others_learning)} LoRA'nın deneyimini gözlemledi")
            print(f"      • {len(interpretation['adopted_learnings'])} öğrenme benimsedi")
            print(f"      • {len(interpretation['rejected_learnings'])} öğrenme reddetti")
            
            if interpretation['personal_insights']:
                print(f"      💭 \"{interpretation['personal_insights']}\"")
        
        if len(population) > 3:
            print(f"\n   ... ve {len(population)-3} LoRA daha kendi öğrenmesini yaptı.")
        
        print(f"{'='*80}")
        
        # 🌊 FLUID TEMPERAMENT: Mizaçları evrimleştir! (Her 10 maçta)
        if result['match_idx'] % 10 == 0:
            print(f"\n🌊 FLUID TEMPERAMENT GÜNCELLEMESI:")
            
            for lora in population[:3]:  # İlk 3 göster
                # Olayları topla
                events = []
                if len(lora.trauma_history) > 0 and getattr(lora.trauma_history[-1], 'match', 0) == result['match_idx']:
                    events.append('trauma')
                if getattr(lora, '_current_motivation', 0) > 1.0:
                    events.append('success_streak')
                
                # Mizacı evrimleştir!
                new_temp = self.fluid_temperament.evolve_temperament(lora, result['match_idx'], events)
                
                # İlk LoRA için göster
                if lora == population[0] and events:
                    old_independence = lora.temperament.get('independence', 0.5)
                    new_independence = new_temp.get('independence', 0.5)
                    if abs(new_independence - old_independence) > 0.01:
                        print(f"   🌊 {lora.name}: Bağımsızlık {old_independence:.3f} → {new_independence:.3f}")
            
            if len(population) > 3:
                print(f"   ... ve {len(population)-3} LoRA daha evrimleşti (sessizce)")
        
        # 🌡️ NATURE THERMOSTAT: Doğanın sıcaklığını güncelle!
        if result['match_idx'] % 5 == 0:
            # Popülasyon entropisini hesapla
            all_probas = [proba for _, proba in individual_predictions]
            pop_entropy = self.nature_thermostat.calculate_population_entropy(all_probas)
            
            # Sıcaklığı güncelle
            temp_update = self.nature_thermostat.update_temperature(pop_entropy, dt=1.0)
            
            # Doğaya etkile!
            temp_effects = self.nature_thermostat.apply_temperature_effects(self.nature_system.nature)
            
            # 🌪️ SYNERGY: NATURE -> SOCIAL NETWORK
            # Doğa çok sıcaksa (kaos), sosyal bağlar stres altına girer!
            if self.nature_thermostat.temperature > 1.2:
                stress_factor = (self.nature_thermostat.temperature - 1.0) * 0.1
                # Tüm bağları zayıflat (Stres testi!)
                total_bonds = len(self.evolution_manager.social_network.bonds)
                weakened_count = 0
                for key in list(self.evolution_manager.social_network.bonds.keys()):
                    self.evolution_manager.social_network.bonds[key] -= stress_factor
                    if self.evolution_manager.social_network.bonds[key] < 0:
                        del self.evolution_manager.social_network.bonds[key]
                    else:
                        weakened_count += 1
                if weakened_count > 0:
                    print(f"   🌪️ DOĞA STRESİ: Yüksek sıcaklık sosyal bağları zayıflattı! (-{stress_factor:.3f})")
        
        # 17) DURUM YAZDIRMA + SCOREBOARD TEPKİLERİ (her 10 maçta)
        print(f"\n🔍 DEBUG: match_idx={result['match_idx']}, mod 10 = {result['match_idx'] % 10}")
        if result['match_idx'] % 10 == 0 and result['match_idx'] > 0:
            print(f"   ✅ 10. MAÇ TETİKLENDİ!")
            self.evolution_manager.print_status()
            self.nature_system.print_nature_status(population_size)
            # self.trigger_system.print_status()
            print(f"\n🌡️ DOĞA SICAKLIĞI: {self.nature_thermostat.temperature:.2f} ({temp_update['status']})")
            print(f"   Entropi: {pop_entropy:.3f} (Hedef: {self.nature_thermostat.target_entropy:.2f})")
            self.logger.log_top_loras(population, top_k=5)
            
            # ⚡ YAŞAYAN LoRA'LAR CANLI RAPORU!
            self.living_reporter.update_living_loras(
                population, 
                result['match_idx'],
                hibernation_manager=self.advanced_mechanics.hibernation  # 😴 Uyuyanları da ekle!
            )
            
            # 📚 POPULATION HISTORY SNAPSHOT!
            print(f"\n   📚 POPULATION HISTORY SNAPSHOT...")
            try:
                hibernated_count = len(self.advanced_mechanics.hibernation.hibernated_loras)
                self.population_history.record_match_snapshot(result['match_idx'], population, hibernated_count)
                self.population_history.save_history(result['match_idx'])
                print(f"      ✅ Snapshot kaydedildi (Olay sayısı: {self.population_history.stats['total_events']})")
            except Exception as e:
                print(f"      ❌ HATA: Population history kaydedilemedi!")
                print(f"         {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 🔄 DİNAMİK YER DEĞİŞTİRME! (Her 10 maçta!)
            print(f"\n🔄 CANLI DİNAMİK YER DEĞİŞTİRME...")
            try:
                relocation_result = self.relocation_engine.evaluate_and_relocate_all(
                    population=population,
                    match_idx=result['match_idx'],
                    tes_triple_scoreboard=self.tes_triple_scoreboard,
                    team_spec_manager=self.team_spec_manager,
                    global_spec_manager=self.global_spec_manager
                )
                
                if relocation_result['relocations']:
                    print(f"   🎭 {len(relocation_result['relocations'])} rol değişikliği yapıldı!")
                    print(f"   ⬆️  Terfi: {relocation_result['stats']['promotions']}")
                    print(f"   ⬇️  Düşme: {relocation_result['stats']['demotions']}")
                else:
                    print(f"   ✅ Rol değişikliği yok (herkes yerinde)")
            except Exception as e:
                print(f"   ❌ Dynamic Relocation hatası: {str(e)}")
                import traceback
                traceback.print_exc()
            
                # 🔬 TES HALL OF FAME - Her 50 maçta!
            if result['match_idx'] % 50 == 0 and result['match_idx'] > 0:
                # 🧬 GENETİK ÇEŞİTLİLİK KONTROLÜ! (Uyarı + Debug!)
                try:
                    self.lazarus.check_population_diversity(population, result['match_idx'])
                except Exception as e:
                    print(f"   ⚠️ Diversity check hatası: {str(e)}")
                
                print(f"\n🔬 TES HALL OF FAME GÜNCELLENİYOR (Maç #{result['match_idx']})...")
            
            # 🧬 GENETİK ÇEŞİTLİLİK KONTROLÜ - Her 10 maçta! (Daha sık!)
            if result['match_idx'] % 10 == 0 and result['match_idx'] > 0:
                try:
                    self.lazarus.check_population_diversity(population, result['match_idx'])
                except Exception as e:
                    print(f"   ⚠️ Diversity check hatası: {str(e)}")
                self.tes_triple_scoreboard.export_all_types(
                    population=self.evolution_manager.population,
                    all_loras_ever=self.all_loras_ever,
                    match_count=result['match_idx'],
                    top_n=15
                )
                print(f"   ✅ Einstein/Newton/Darwin/Potansiyel Hall güncellendi!")
                
                # 🆕 MUCİZE HALL TXT GÜNCELLE
                from lora_system.miracle_hall_manager import miracle_hall_manager
                miracle_hall_manager.generate_miracle_hall_txt(match_count=result['match_idx'])
            
            # 📊 SCOREBOARD DEĞİŞİMLERİNE PSİKOLOJİK TEPKİ! (Akışkan!)
            # Her LoRA'nın rank değişimini kontrol et
            from lora_system.psychological_responses import psychological_responses
            from lora_system.advanced_score_calculator import advanced_score_calculator
            
            # Mevcut sıralamayı hesapla
            lora_scores = []
            for lora in population:
                score = advanced_score_calculator.calculate_advanced_score(lora, result['match_idx'])
                lora_scores.append((lora, score))
            
            lora_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Rank kontrol et (her LoRA için eski rank varsa)
            for rank, (lora, score) in enumerate(lora_scores, start=1):
                old_rank = getattr(lora, '_last_known_rank', None)
                
                if old_rank and old_rank != rank:
                    # RANK DEĞİŞTİ!
                    if rank > old_rank:
                        # DÜŞTÜ!
                        response = psychological_responses.react_to_rank_drop(
                            lora, old_rank, rank, result['match_idx']
                        )
                        
                        # Direnci kaydet
                        if response['death_resistance'] > 0.10:
                            self.experience_resistance.add_rank_drop_survival(
                                lora.id, rank - old_rank, survived_how="determination"
                            )
                
                # Güncel ranki kaydet
                lora._last_known_rank = rank
    
    def _apply_nature_event(self, event: Dict, match_num: int):
        """Doğa olayını popülasyona uygula (ZIRHLI!) - UYUYANLAR DA ETKİLENİR!"""
        event_type = event['type']
        population = self.evolution_manager.population
        
        if event_type in ['kara_veba', 'mass_extinction', 'overpopulation_purge']:
            # 🌍 ADAPTIF DOĞA: Severity'yi ayarla!
            base_kill_ratio = event.get('kill_ratio', 1.0 - event.get('survival_rate', 0.2))
            adaptive_kill_ratio = self.adaptive_nature.calculate_adaptive_severity(
                population,
                event_type,
                base_kill_ratio
            )
            
            print(f"   🌍 Base kill: %{base_kill_ratio*100:.0f} → Adaptive: %{adaptive_kill_ratio*100:.0f}")
            
            # ELİT DİRENCİ + SAĞ KALAN SENDROMU ile ölüm
            survivors, guilt_list = self.advanced_mechanics.apply_disaster_with_armor(
                population,
                adaptive_kill_ratio,
                event_type,
                match_num,
                all_loras_ever=self.all_loras_ever,  # 🏆 Elite kontrolü için!
                miracle_system=self.miracle_system   # 🏆 Mucize kontrolü için!
            )
            
            # Ölenleri logla
            for lora in population:
                if lora not in survivors:
                    # Doğa olayından ölüm sebebi
                    if event_type == 'kara_veba':
                        death_detail = f"Kara Veba felaketi (Armor: {getattr(lora, 'elite_armor', 0)*100:.0f}%)"
                    elif event_type == 'nufus_patlamasi':
                        death_detail = "Nüfus patlaması cezası (Aşırı popülasyon)"
                    else:
                        death_detail = event_type
                    

                    physics_data = self._get_physics_snapshot(lora)
                    self.logger.log_death(lora, reason=event_type, 
                                        death_reason_detail=death_detail,
                                        physics_data=physics_data)
            
            # Sağ kalan sendromunu uygula + BAĞIŞIKLIK KAZANDIR!
            guilt_count = 0
            immunity_count = 0
            
            for lora, guilt in guilt_list:
                self.advanced_mechanics.apply_survivor_guilt(lora, guilt)
                guilt_count += 1
                
                # 🛡️ BAĞIŞIKLIK KAZANDIR! (Zırh ile kurtuldu = adaptasyon!)
                self.adaptive_nature.lora_survived_event(lora, event_type, survived_by="armor")
                immunity_count += 1
                
                print(f"   😢 {lora.name}: Sağ kalan sendromu (zırh ile kurtuldu)")
                print(f"      Fitness: -{guilt.fitness_penalty*100:.1f}%, Travma: +{guilt.trauma_gain:.2f}")
                print(f"      🛡️ Bağışıklık kazandı! ({event_type})")
            
            # Şanslı kurtulanlar da bağışıklık kazanır (ama daha az!)
            for lora in survivors:
                if lora not in [l for l, g in guilt_list]:
                    # Şanslı kurtuldu (zırh yok)
                    self.adaptive_nature.lora_survived_event(lora, event_type, survived_by="luck")
            
            self.evolution_manager.population = survivors
            print(f"   💀 {len(population) - len(survivors)} LoRA öldü (aktif)")
            print(f"   🛡️ {guilt_count} LoRA zırh ile kurtuldu (ama bedel ödedi)")
            print(f"   🧬 {len(survivors)} LoRA bağışıklık kazandı!")
            
            # 💤 UYUYANLARA DA UYGULAN! (Gerçek dünya mantığı!)
            hibernated = self.advanced_mechanics.hibernation.hibernated_loras
            if len(hibernated) > 0:
                print(f"\n   💤 UYUYANLAR DA ETKİLENİYOR! ({len(hibernated)} uyuyan)")
                
                # Aynı kill_ratio'yu uygula
                hibernated_ids = list(hibernated.keys())
                kill_count_hibernated = int(len(hibernated_ids) * adaptive_kill_ratio)
                
                # Rastgele seç (zırh yok, uyuyanlar savunmasız!)
                import random
                to_kill_hibernated = random.sample(hibernated_ids, min(kill_count_hibernated, len(hibernated_ids)))
                
                # Öldür (dosyalarını sil!)
                for lora_id in to_kill_hibernated:
                    file_path = hibernated[lora_id]
                    
                    # Dosyayı sil
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # Hibernated listesinden çıkar
                    del self.advanced_mechanics.hibernation.hibernated_loras[lora_id]
                    
                    # Fake log (uyurken öldüler!)
                    # Not: Tam LoRA objesi yok, sadece dosya
                    print(f"      💀😴 {lora_id[:8]} uyurken öldü!")
                
                print(f"   💀 {len(to_kill_hibernated)} LoRA öldü (uyurken!)")
                print(f"   😴 {len(self.advanced_mechanics.hibernation.hibernated_loras)} LoRA hayatta (uyuyan)")
            
            print(f"   👥 TOPLAM HAYATTA: {len(survivors)} aktif + {len(self.advanced_mechanics.hibernation.hibernated_loras)} uyuyan")
            
            # 🌍 DOĞA GÖZLEMLER: Bu olay ne kadar etkili oldu?
            initial_population = len(population) + len(self.advanced_mechanics.hibernation.hibernated_loras)
            final_population = len(survivors) + len(self.advanced_mechanics.hibernation.hibernated_loras)
            death_rate = (initial_population - final_population) / initial_population if initial_population > 0 else 0
            
            # Doğa öğrenir!
            immunity_detected = self.adaptive_nature.observe_lora_immunity(
                survivors,
                event_type,
                death_rate
            )
            
            # LoRA'lar çok bağışıksa → Doğa evrimleşir!
            if immunity_detected:
                evolution_msg = self.adaptive_nature.evolve_nature(survivors, match_num)
                if evolution_msg:
                    print(f"\n{evolution_msg}")
        
        # ============================================
        # KÜÇÜK-ORTA TEPKİLER (Sık olur!)
        # ============================================
        elif event_type in ['minor_shake', 'stress_wave', 'quake', 'health_crisis', 'major_quake', 'perfect_storm']:
            # Sosyal bağlar ve çekimler sarsılır
            affected_count = int(len(population) * event.get('affected_ratio', 0.5))
            affected_loras = np.random.choice(population, size=min(affected_count, len(population)), replace=False)
            
            severity = event.get('severity', 0.5)
            
            # MİZAÇ BAZLI ETKİ (Her LoRA farklı etkilenir!)
            from lora_system.psychological_responses import psychological_responses
            
            for lora in affected_loras:
                # Sosyal bağları zayıflat (severity'e göre)
                if hasattr(lora, 'social_bonds'):
                    for bond_id in list(lora.social_bonds.keys()):
                        # Dayanıklı LoRA daha az etkilenir!
                        resilience = lora.temperament.get('resilience', 0.5)
                        impact = severity * (1.0 - resilience * 0.5)
                        lora.social_bonds[bond_id] *= (1.0 - impact * 0.4)  # Max %40 azalır
                
                # Pattern çekimlerini sarsıt
                if hasattr(lora, 'pattern_attractions'):
                    for pattern in lora.pattern_attractions:
                        stress_tolerance = lora.temperament.get('stress_tolerance', 0.5)
                        impact = severity * (1.0 - stress_tolerance * 0.5)
                        lora.pattern_attractions[pattern] *= (1.0 - impact * 0.3)
                
                # Travma ekle (hafif)
                if hasattr(lora, 'trauma_history'):
                    emotional_depth = lora.temperament.get('emotional_depth', 0.5)
                    trauma_amount = severity * 0.3 * emotional_depth
                    
                    if trauma_amount > 0.1:  # Sadece anlamlıysa kaydet
                        lora.trauma_history.append({
                            'type': event_type,
                            'match': match_num,
                            'severity': trauma_amount
                        })
            
            print(f"   🌍 {affected_count} LoRA etkilendi (Mizaç bazlı!)")
    
    def run(self, csv_path: str, start_match: int = 0, max_matches: Optional[int] = None, 
            results_csv: str = 'results_matches.csv'):
        """
        Ana döngü
        
        Args:
            csv_path: Tahmin dosyası (sonuçsuz maçlar)
            results_csv: Gerçek sonuçlar dosyası
        """
        # Veriyi yükle (tahmin için - sonuçsuz)
        df = self.load_data(csv_path)
        
        # Sonuç dosyasını yükle (gerçek sonuçlar)
        print(f"\n📂 Sonuç dosyası yükleniyor: {results_csv}")
        self.results_df = pd.read_csv(results_csv)
        print(f"   ✅ {len(self.results_df)} maçın sonucu yüklendi")
        
        # Maç sayısı
        total_matches = len(df) if max_matches is None else min(len(df), max_matches)
        
        # 📊 BAŞLANGIÇ POPÜLASYONU KAYDET
        initial_population = len(self.evolution_manager.population)
        
        print(f"\n{'='*80}")
        print(f"🚀 EVRİMSEL ÖĞRENME BAŞLIYOR!")
        print(f"{'='*80}")
        print(f"  Toplam Maç: {total_matches}")
        print(f"  Başlangıç Maçı: {start_match}")
        print(f"  Başlangıç Popülasyonu: {initial_population} LoRA")
        print(f"  📋 Tahmin Dosyası: {csv_path} (SONUÇSUZ)")
        print(f"  ✅ Sonuç Dosyası: {results_csv} (GERÇEK SONUÇLAR)")
        print(f"{'='*80}\n")
        
        # ============================================
        # 📚 BAŞLANGIÇTA TÜM LoRA'LAR GEÇMİŞİ OKUSUN!
        # ============================================
        
        if len(self.collective_memory.memory) > 0 or len(self.all_loras_ever) > 0:
            print(f"\n{'='*80}")
            print(f"📚 LoRA'LAR GEÇMİŞİ OKUYOR! (Atalardan Öğrenme)")
            print(f"{'='*80}")
            print(f"   📖 Ortak Hafıza: {len(self.collective_memory.memory)} maç")
            print(f"   🏛️ Tüm Zamanlar: {len(self.all_loras_ever)} LoRA kaydı")
            print(f"{'='*80}\n")
            
            # Her LoRA geçmişi okusun (ilk 5 göster)
            for idx, lora in enumerate(self.evolution_manager.population[:5]):
                historical_insights = self.historical_learning.lora_reads_collective_history(
                    lora,
                    self.collective_memory.memory,
                    self.all_loras_ever
                )
                
                # Kişisel hafızasına kaydet
                lora.personal_memory['historical_insights'] = historical_insights
            
            # Geri kalanlar sessizce okusun
            if len(self.evolution_manager.population) > 5:
                print(f"\n   ... ve {len(self.evolution_manager.population)-5} LoRA daha geçmişi okudu (sessizce)")
            
            print(f"\n{'='*80}")
            print(f"✅ TÜM LoRA'LAR HAZIR! Atalardan öğrendiler!")
            print(f"{'='*80}\n")
        
        # ============================================
        # 🔬 FİZİK ÖZELLİKLERİNİ VER! (Life Energy + Fluid Temperament!)
        # ============================================
        
        print(f"\n{'='*80}")
        print(f"⚡ FİZİK ÖZELLİKLERİ BAŞLATILIYOR!")
        print(f"{'='*80}")
        
        for lora in self.evolution_manager.population:
            # Life Energy başlat
            if not hasattr(lora, 'life_energy'):
                self.life_energy.initialize_life_energy(lora)
            
            # Fluid Temperament başlat
            if lora.id not in self.fluid_temperament.temperament_dynamics:
                self.fluid_temperament.initialize_dynamics(lora)
        
        print(f"   ✅ {len(self.evolution_manager.population)} LoRA'ya fizik özellikleri verildi!")
        print(f"{'='*80}\n")
        

        
        print(f"{'='*80}\n")
        
        # Oturum başlangıcını logla
        resume_mode = len(self.evolution_manager.population) > 5  # İlk popülasyon 20, yüklendiyse > 5
        self.match_logger.log_session_start(total_matches, resume=resume_mode)
        
        # Ana döngü
        for idx in range(start_match, start_match + total_matches):
            if idx >= len(df):
                break
            
            # 💀 SOY TÜKENMESİ KONTROLÜ
            if len(self.evolution_manager.population) == 0:
                print(f"\n{'💀'*80}")
                print(f"💀 SOY TÜKENDİ! (Maç #{idx})")
                print(f"{'💀'*80}")
                print(f"\n⚡ DİRİLTME KOMUTU:")
                print(f"   python run_evolutionary_learning.py --resurrect --start {idx}")
                print(f"\n📚 ORTAK HAFIZA KORUNDU! {len(self.collective_memory.memory)} maç bilgisi güvende!")
                print(f"{'💀'*80}\n")
                break  # Döngüyü durdur!
            
            # Tahmin için maç bilgisi al (SONUÇSUZ)
            match_data = df.iloc[idx]
            
            # Gerçek sonucu sonuç dosyasından al
            if idx < len(self.results_df):
                match_data = match_data.copy()
                # Gerçek sonucu ekle (öğrenme için)
                result_row = self.results_df.iloc[idx]
                match_data['_actual_home_goals'] = result_row['home_goals']
                match_data['_actual_away_goals'] = result_row['away_goals']
                match_data['_actual_result'] = result_row['result']
            
            result = self.run_match(match_data, idx)
            
            # 📊 LoRA PANEL GÜNCELLE (Her 10 maçta)
            if idx % 10 == 0:
                self.panel_generator.generate_panel(
                    population=self.evolution_manager.population,
                    match_count=idx,
                    nature_thermostat=self.nature_thermostat
                )
            
            # Her 50 maçta kaydet + TAKIM UZMANLIK EXPORT + LOG VALİDASYON!
            if idx % 50 == 0 and idx > 0:
                self.save_state()
                
                # 🔍 LOG VALİDASYONU! (Tutarlılık kontrolü)
                print(f"\n🔍 LOG VALİDASYONU YAPILIYOR...")
                validation_result = self.log_validator.validate_all(
                    match_idx=idx,
                    active_population=self.evolution_manager.population,
                    all_loras_ever=self.all_loras_ever,
                    miracle_system=self.miracle_system,
                    tes_scoreboard=None,  # TES için ayrı kontrol yapacağız
                    team_spec_manager=self.team_spec_manager,
                    global_spec_manager=self.global_spec_manager
                )
                
                if not validation_result['valid']:
                    print(f"   ⚠️ {len(validation_result['errors'])} validasyon hatası bulundu!")
                    for error in validation_result['errors'][:3]:
                        print(f"      • {error}")
                else:
                    print(f"   ✅ Tüm loglar geçerli!")
                
                # 🏆 TAKIM + GENEL UZMANLIK EXPORT! (Her 50 maçta)
                print(f"\n🏆 UZMANLIK SİSTEMLERİ GÜNCELLENİYOR (Maç #{idx})...")
                
                # 1) TAKIM UZMANLIKLARI (Top 5 her takım için)
                print(f"   📊 Takım uzmanlıkları hesaplanıyor...")
                spec_results = self.team_spec_manager.calculate_team_specialization_scores(
                    self.evolution_manager.population,
                    idx
                )
                
                # Cache'e kaydet (ölümsüzlük hesabı için!)
                self.team_spec_manager.top_5_cache = spec_results
                
                # Export et (.pt + .txt)
                self.team_spec_manager.export_team_specializations(spec_results, idx)
                print(f"   ✅ {len(spec_results)} takım için uzmanlık skorları güncellendi!")
                
                # 2) GENEL UZMANLIKLAR (Top 10 - tüm maçlar!)
                print(f"\n   🌍 Genel uzmanlıklar hesaplanıyor...")
                global_results = self.global_spec_manager.calculate_global_specialization_scores(
                    self.evolution_manager.population,
                    idx
                )
                
                # Export et
                self.global_spec_manager.export_global_specializations(global_results, idx)
                print(f"   ✅ Genel uzmanlar güncellendi!")
                
                # 🔬 HALL & UZMANLIK AUDIT! (Her 50 maçta!)
                print(f"\n🔬 HALL & UZMANLIK AUDIT YAPILIYOR...")
                audit_report = self.hall_auditor.full_audit(
                    match_idx=idx,
                    population=self.evolution_manager.population,
                    all_loras_ever=self.all_loras_ever,
                    miracle_system=self.miracle_system,
                    tes_triple_scoreboard=self.tes_triple_scoreboard,
                    team_spec_manager=self.team_spec_manager,
                    global_spec_manager=self.global_spec_manager
                )
                
                print(f"   📊 Audit Sonuçları:")
                print(f"      • Kategorisiz: {audit_report['uncategorized_count']} LoRA")
                print(f"      • Superhybrid: {audit_report['superhybrid_count']} LoRA")
                print(f"      • Yanlış Kategori: {audit_report['miscategorized_count']} LoRA")
                print(f"      • Eksik Dosya: {audit_report['missing_files_count']} dosya")
                
                if audit_report['superhybrid_count'] > 0:
                    print(f"\n   ⭐ SUPERHYBRID LoRA'LAR BULUNDU!")
                    superhybrids = self.hall_auditor.superhybrids[:3]
                    for lora, spec_count, categories in superhybrids:
                        print(f"      • {lora.name}: {spec_count} uzmanlık!")
                
                if audit_report['uncategorized_count'] > 0:
                    print(f"\n   ⚠️ {audit_report['uncategorized_count']} KATEGORİSİZ LoRA VAR!")
                    print(f"      Detaylar için: evolution_logs/🔬_HALL_SPEC_AUDIT.log")
                
                # 🔄 DİNAMİK YER DEĞİŞTİRME! (Her 10 maçta dosya işlemleri!)
                print(f"\n🔄 CANLI DİNAMİK YER DEĞİŞTİRME YAPILIYOR...")
                relocation_result = self.relocation_engine.evaluate_and_relocate_all(
                    population=self.evolution_manager.population,
                    match_idx=idx,
                    tes_triple_scoreboard=self.tes_triple_scoreboard,
                    team_spec_manager=self.team_spec_manager,
                    global_spec_manager=self.global_spec_manager
                )
                
                # İstatistikler
                if relocation_result['relocations']:
                    print(f"   🎭 Rol Değişikliği: {len(relocation_result['relocations'])} LoRA")
                    print(f"   ⬆️  Terfi: {relocation_result['stats']['promotions']}")
                    print(f"   ⬇️  Düşme: {relocation_result['stats']['demotions']}")
                    
                    # 📚 ROL DEĞİŞİKLİKLERİNİ HISTORY'YE KAYDET!
                    for relocation in relocation_result['relocations']:
                        # İlgili LoRA'yı bul
                        lora_id = relocation['lora_id']
                        matching_lora = next((l for l in self.evolution_manager.population if l.id == lora_id), None)
                        
                        if matching_lora:
                            self.population_history.record_role_change(
                                matching_lora,
                                idx,
                                relocation['added'],
                                relocation['removed']
                            )
                
                # Dağılımı göster
                self.relocation_engine.print_current_distribution(idx)
                
                # 🔍 TAKIM UZMANLIK DENETİMİ!
                print(f"\n🔍 TAKIM UZMANLIK DENETİMİ...")
                audit_result = self.team_spec_auditor.full_audit(
                    population=self.evolution_manager.population,
                    match_idx=idx,
                    team_spec_manager=self.team_spec_manager
                )
                
                if audit_result['total_issues'] == 0:
                    print(f"   ✅ Takım uzmanlıkları kusursuz!")
                else:
                    print(f"   ⚠️  {audit_result['total_issues']} sorun tespit edildi")
                    print(f"   📋 Detaylar: evolution_logs/🔍_TEAM_SPEC_AUDIT_M{idx}.log")
                
                # 🔄 TOPLU SENKRONIZASYON! (Tüm kopyaları güncelle!)
                print(f"\n🔄 TOPLU SENKRONIZASYON...")
                sync_result = self.lora_sync.sync_entire_population(
                    self.evolution_manager.population,
                    idx,
                    self.population_history
                )
                
                # Senkronizasyon istatistikleri
                sync_stats = self.lora_sync.get_sync_stats()
                print(f"   📊 Toplam takip edilen: {sync_stats['total_loras_tracked']} LoRA")
                print(f"   📁 Toplam kopya: {sync_stats['total_copies_tracked']} dosya")
                print(f"   📈 Ortalama kopya/LoRA: {sync_stats['average_copies_per_lora']:.1f}")
                
                # 3) 🔄 PT SYNC! (Çoklu uzmanlığa sahip LoRA'ların kopyalarını güncelle!)
                print(f"\n   🔄 PT dosyaları sync ediliyor...")
                sync_count = 0
                for lora in self.evolution_manager.population:
                    # Bu LoRA'nın uzmanlıklarını topla
                    team_specs = {}
                    global_specs = []
                    
                    # Takım uzmanlıkları
                    for team, team_data in spec_results.items():
                        lora_id = lora.id
                        
                        # Win expert mi?
                        if any(l.id == lora_id for l, _ in team_data['win_experts']):
                            if team not in team_specs:
                                team_specs[team] = []
                            team_specs[team].append('Win')
                        
                        # Goal expert mi?
                        if any(l.id == lora_id for l, _ in team_data['goal_experts']):
                            if team not in team_specs:
                                team_specs[team] = []
                            team_specs[team].append('Goal')
                        
                        # Hype expert mi?
                        if any(l.id == lora_id for l, _ in team_data['hype_experts']):
                            if team not in team_specs:
                                team_specs[team] = []
                            team_specs[team].append('Hype')
                    
                    # Genel uzmanlıklar
                    if any(l.id == lora.id for l, _ in global_results['win_experts']):
                        global_specs.append('Win')
                    if any(l.id == lora.id for l, _ in global_results['goal_experts']):
                        global_specs.append('Goal')
                    if any(l.id == lora.id for l, _ in global_results['hype_experts']):
                        global_specs.append('Hype')
                    
                    # Eğer herhangi bir uzmanlığı varsa, sync et!
                    if team_specs or global_specs:
                        self.sync_manager.register_lora_specializations(
                            lora,
                            team_specs,
                            global_specs,
                            {
                                'team': 'takım_uzmanlıkları',
                                'global': 'en_iyi_loralar/🌍_GENEL_UZMANLAR'
                            }
                        )
                        sync_count += 1
                
                print(f"   ✅ {sync_count} LoRA sync edildi!")
        
        # Oturum bitişini logla
        self.match_logger.log_session_end(total_matches, len(self.evolution_manager.population))
        
        # Final kayıt
        self.save_state()
        self.logger.save_all()
        self.logger.generate_summary_report()
        
        # ⭐ EN İYİ LoRA'LARI KAYDET!
        # GENEL TOP 50 (Eski sistem - geriye uyumluluk)
        from lora_system.top_lora_exporter import TopLoRAExporter
        exporter = TopLoRAExporter(export_dir="best_loras")
        exporter.export_all(
            population=self.evolution_manager.population,
            miracle_system=self.miracle_system,
            match_count=total_matches,
            all_loras_ever=self.all_loras_ever,
            top_n=50,
            collective_memory=self.collective_memory  # 🆕 H2H İÇİN GEREKLİ!
        )
        
        # 🔬 TES TRIPLE SCOREBOARD! (Einstein/Newton/Darwin ayrı!)
        print(f"\n🔬 TES TRIPLE SCOREBOARD (3 Ayrı Hall of Fame!)")
        from lora_system.tes_triple_scoreboard import TESTripleScoreboard
        tes_exporter = TESTripleScoreboard(export_dir="best_loras")
        tes_exporter.export_all_types(
            population=self.evolution_manager.population,
            all_loras_ever=self.all_loras_ever,
            match_count=total_matches,
            top_n=15  # Her tipten 15!
        )
        
        print(f"\n{'='*80}")
        print(f"✅ EVRİMSEL ÖĞRENME TAMAMLANDI!")
        print(f"{'='*80}")
        print(f"  İşlenen Maç: {total_matches}")
        print(f"  Başlangıç Popülasyon: {initial_population} LoRA")
        print(f"  Final Popülasyon: {len(self.evolution_manager.population)} LoRA")
        print(f"  📊 Maç sonuçları: evolution_logs/match_results.log")
        print(f"  📁 Detaylı loglar: evolution_logs/")
        print(f"  ⭐ En iyi LoRA'lar: en_iyi_loralar/")
        print(f"{'='*80}\n")
    
    def save_state(self):
        """Sistem durumunu kaydet"""
        print("\n💾 Durum kaydediliyor...")
        
        # Buffer
        self.buffer.save(self.paths['buffer'])
        
        # Evolution state (LoRA'lar + tam metadata)
        torch.save({
            'population': [lora.get_all_lora_params() for lora in self.evolution_manager.population],
            'metadata': [
                {
                    'id': lora.id, 
                    'name': lora.name, 
                    'generation': lora.generation,
                    'fitness_history': getattr(lora, 'fitness_history', []),
                    'match_history': getattr(lora, 'match_history', []),
                    'specialization': getattr(lora, 'specialization', None),
                    'birth_match': getattr(lora, 'birth_match', 0),
                    'parents': getattr(lora, 'parents', []),
                    'temperament': getattr(lora, 'temperament', {})  # 🎭 KİŞİLİK!
                } 
                for lora in self.evolution_manager.population
            ],
            'nature_state': {
                'health': self.nature_system.nature.health,
                'anger': self.nature_system.nature.anger
            },
            'collective_memory': self.collective_memory.memory,  # 🌐 ORTAK HAFIZA!
            'all_loras_summary': {  # 📚 TÜM ZAMANLAR ÖZET (ölüler dahil!)
                lora_id: {
                    'name': info['lora'].name,
                    'final_fitness': info['final_fitness'],
                    'alive': info['alive'],
                    'age': info.get('age', 0),
                    'death_match': info.get('death_match'),
                    'death_reason': info.get('death_reason')  # 💀 ÖLÜM SEBEBİ!
                }
                for lora_id, info in self.all_loras_ever.items()
            },
            'adaptive_nature': {  # 🌍 EVRİMLEŞEN DOĞA!
                'version': self.adaptive_nature.nature_version,
                'evolution_history': self.adaptive_nature.evolution_history,
                'lora_immunity': self.adaptive_nature.lora_immunity,
                'nature_memory': self.adaptive_nature.nature_memory
            },
            'experience_resistance': {  # 🛡️ DENEYİM DİRENCİ!
                'lora_resistances': self.experience_resistance.lora_resistances
            },
            'dynamic_specialization': {  # 🔍 DİNAMİK UZMANLIK!
                'discovered_patterns': self.dynamic_spec.discovered_patterns
            },
            'meta_learning': {  # 🧠 META-ADAPTIF ÖĞRENME!
                'learning_rates': self.meta_learning.learning_rates,
                'lr_history': self.meta_learning.lr_history,
                'performance_history': self.meta_learning.performance_history
            },
            'ghost_fields': {  # 👻 HAYALET ALANLAR!
                'ghost_influence': self.ghost_fields.ghost_influence
                # Parametreler çok ağır, sadece influence kaydet
            },
            'fluid_temperament': {  # 🌊 AKIŞKAN MİZAÇ!
                'temperament_dynamics': self.fluid_temperament.temperament_dynamics
            },
            'nature_thermostat': {  # 🌡️ DOĞA TERMOSTATI!
                'temperature': self.nature_thermostat.temperature,
                'entropy_history': self.nature_thermostat.entropy_history[-50:]  # Son 50
            }
        }, self.paths['lora_population'])
        
        # Meta-LoRA
        if isinstance(self.meta_lora, MetaLoRA):
            torch.save(self.meta_lora.state_dict(), self.paths['meta_lora'])
        
        print("   ✅ Durum kaydedildi")
    
    def load_state(self):
        """Kaydedilmiş durumu yükle"""
        import os
        
        print("\n📂 Kaydedilmiş durum yükleniyor...")
        
        # LoRA populasyonunu yükle
        if os.path.exists(self.paths['lora_population']):
            checkpoint = torch.load(self.paths['lora_population'], weights_only=False)
            
            # Mevcut popülasyonu temizle
            self.evolution_manager.population.clear()
            
            # Temperament eksik anahtarlarını doldur (eski LoRA'lar için)
            def _fix_temperament(lora):
                """Eksik temperament anahtarlarını varsayılan değerlerle doldur"""
                default_temperament = {
                    # TEMEL (4)
                    'independence': 0.6,
                    'social_intelligence': 0.6,
                    'herd_tendency': 0.4,
                    'contrarian_score': 0.3,
                    # DUYGUSAL (3)
                    'emotional_depth': 0.5,
                    'empathy': 0.5,
                    'anger_tendency': 0.5,
                    # PERFORMANS (4)
                    'ambition': 0.6,
                    'competitiveness': 0.5,
                    'resilience': 0.6,
                    'will_to_live': 0.7,
                    # DAVRANIŞSAL (4)
                    'patience': 0.6,
                    'impulsiveness': 0.4,
                    'stress_tolerance': 0.6,
                    'risk_appetite': 0.5
                }
                
                # Eksik anahtarları doldur
                for key, default_value in default_temperament.items():
                    if key not in lora.temperament:
                        lora.temperament[key] = default_value
            
            # FORMAT KONTROLÜ: Emergency/Spawn formatı mı, normal format mı?
            if 'resurrection_info' in checkpoint or 'spawn_info' in checkpoint or ('metadata' not in checkpoint and 'population' in checkpoint):
                # EMERGENCY/SPAWN FORMATI: Direkt LoRA objeleri listesi
                print("   🔄 Emergency/Spawn formatı tespit edildi...")
                if 'resurrection_info' in checkpoint:
                    print(f"   📋 Resurrection Tipi: {checkpoint['resurrection_info'].get('type', 'UNKNOWN')}")
                elif 'spawn_info' in checkpoint:
                    print(f"   📋 Spawn Tipi: {checkpoint['spawn_info'].get('type', 'UNKNOWN')}")
                    print(f"   🌊 Çeşitlilik: {checkpoint['spawn_info'].get('diversity_level', 'NORMAL')}")
                    print(f"   🔥 Hafıza Reset: {checkpoint['spawn_info'].get('memory_reset', False)}")
                
                for old_lora in checkpoint['population']:
                    try:
                        # GÜVENLI YÖNTEM: Yeni LoRA yarat, eski state'i yükle
                        # Bu sayede device uyumsuzluğu olmaz
                        
                        # 1. Yeni LoRA yarat (doğru device'da - artık __init__ içinde .to(device) çağrılıyor!)
                        new_lora = LoRAAdapter(
                            input_dim=self.config.get('lora', {}).get('input_dim', 63),
                            hidden_dim=self.config.get('lora', {}).get('hidden_dim', 128),
                            rank=self.config.get('lora', {}).get('rank', 16),
                            alpha=self.config.get('lora', {}).get('alpha', 16.0),
                            device=self.device
                        )  # .to(self.device) artık gerekli değil, __init__ içinde yapılıyor
                        
                        # 2. Eski LoRA'nın state_dict'ini al ve yükle
                        try:
                            old_state = old_lora.state_dict()
                            # Tüm tensörleri doğru device'a taşı
                            new_state = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in old_state.items()}
                            new_lora.load_state_dict(new_state, strict=False)
                            print(f"      ✅ State dict yüklendi: {new_lora.id}")
                        except Exception as state_err:
                            # state_dict başarısız olursa, hata yazdır
                            print(f"      ⚠️ State dict yüklenemedi ({new_lora.id}): {state_err}")
                            print(f"      🔄 Manuel parametre kopyalama deneniyor...")
                            
                            # Manuel olarak parametreleri kopyala
                            try:
                                # fc1, fc2, fc3 parametrelerini manuel kopyala
                                for layer_name in ['fc1', 'fc2', 'fc3']:
                                    old_layer = getattr(old_lora, layer_name)
                                    new_layer = getattr(new_lora, layer_name)
                                    
                                    # Weight ve LoRA parametrelerini kopyala
                                    new_layer.weight.data = old_layer.weight.data.to(self.device)
                                    new_layer.lora_A.data = old_layer.lora_A.data.to(self.device)
                                    new_layer.lora_B.data = old_layer.lora_B.data.to(self.device)
                                
                                print(f"      ✅ Manuel kopyalama başarılı!")
                            except Exception as manual_err:
                                print(f"      ❌ Manuel kopyalama da başarısız: {manual_err}")
                        
                        # 3. Metadata'yı kopyala
                        new_lora.id = old_lora.id
                        new_lora.name = old_lora.name
                        new_lora.generation = old_lora.generation
                        new_lora.parents = old_lora.parents if hasattr(old_lora, 'parents') else []
                        new_lora.birth_match = old_lora.birth_match if hasattr(old_lora, 'birth_match') else 0
                        new_lora.fitness_history = old_lora.fitness_history if hasattr(old_lora, 'fitness_history') else [0.5]
                        new_lora.match_history = old_lora.match_history if hasattr(old_lora, 'match_history') else []
                        new_lora.specialization = old_lora.specialization if hasattr(old_lora, 'specialization') else None
                        new_lora.temperament = old_lora.temperament.copy() if hasattr(old_lora, 'temperament') else {}
                        new_lora.trauma_history = old_lora.trauma_history if hasattr(old_lora, 'trauma_history') else []
                        new_lora.social_bonds = old_lora.social_bonds if hasattr(old_lora, 'social_bonds') else {}
                        new_lora.lucky_survivals = old_lora.lucky_survivals if hasattr(old_lora, 'lucky_survivals') else 0
                        new_lora.resurrection_count = getattr(old_lora, 'resurrection_count', 0) + 1
                        new_lora.children_count = old_lora.children_count if hasattr(old_lora, 'children_count') else 0
                        
                        # Temperament'ı düzelt (eski LoRA'lar eksik anahtarlara sahip olabilir)
                        _fix_temperament(new_lora)
                        
                        self.evolution_manager.population.append(new_lora)
                    except Exception as e:
                        print(f"   ⚠️ Bir LoRA yüklenemedi: {e}")
                        import traceback
                        traceback.print_exc()
            elif 'metadata' in checkpoint and 'population' in checkpoint:
                # NORMAL FORMAT: params + metadata
                print("   🔄 Normal format tespit edildi (params + metadata)...")
                if not checkpoint['population'] or not checkpoint['metadata']:
                    print("   ⚠️ Boş popülasyon veya metadata!")
                    return
                
                # Parametreler listesi + metadata listesi
                for params, meta in zip(checkpoint['population'], checkpoint['metadata']):
                    try:
                        lora = LoRAAdapter(
                            input_dim=self.config.get('lora', {}).get('input_dim', 63),
                            hidden_dim=self.config.get('lora', {}).get('hidden_dim', 128),
                            rank=self.config.get('lora', {}).get('rank', 16),
                            alpha=self.config.get('lora', {}).get('alpha', 16.0),
                            device=self.device
                        ).to(self.device)
                        
                        # Parametreleri yükle
                        lora.set_all_lora_params(params)
                        
                        # Metadata'yı geri yükle
                        lora.id = meta['id']
                        lora.name = meta['name']
                        lora.generation = meta['generation']
                        lora.fitness_history = meta.get('fitness_history', [])
                        lora.match_history = meta.get('match_history', [])
                        lora.specialization = meta.get('specialization', None)
                        lora.birth_match = meta.get('birth_match', 0)
                        lora.parents = meta.get('parents', [])
                        lora.temperament = meta.get('temperament', lora.temperament)  # 🎭 KİŞİLİĞİ YÜKLE!
                        
                        # Temperament'ı düzelt
                        _fix_temperament(lora)
                        
                        self.evolution_manager.population.append(lora)
                    except Exception as e:
                        print(f"   ⚠️ Bir LoRA yüklenemedi: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print("   ⚠️ Bilinmeyen format! Direkt population deneniyor...")
                # Son çare: Population direkt LoRA listesi olabilir
                if 'population' in checkpoint:
                    for item in checkpoint['population']:
                        if hasattr(item, 'id'):  # LoRA objesi
                            try:
                                item.to(self.device)
                                _fix_temperament(item)
                                self.evolution_manager.population.append(item)
                            except Exception as e:
                                print(f"   ⚠️ Bir LoRA yüklenemedi: {e}")
                else:
                    print("   ❌ 'population' key'i bulunamadı!")
                    return
            
            print(f"   ✅ {len(self.evolution_manager.population)} LoRA yüklendi")
            
            # Doğa durumunu yükle
            if 'nature_state' in checkpoint:
                self.nature_system.nature.health = checkpoint['nature_state']['health']
                self.nature_system.nature.anger = checkpoint['nature_state']['anger']
                print(f"   ✅ Doğa durumu yüklendi (Sağlık: {self.nature_system.nature.health:.2f})")
            
            # 🌐 ORTAK HAFIZAYI YÜKLE (MODEL İÇİNDEN!)
            if 'collective_memory' in checkpoint:
                self.collective_memory.load_from_dict(checkpoint['collective_memory'])
                print(f"   ✅ Ortak Hafıza yüklendi ({len(self.collective_memory.memory)} maç)")
            
            # 📚 TÜM ZAMANLAR KAYDINI YÜKLE (Emergency resurrection formatı)
            if 'all_loras_ever' in checkpoint:
                # Emergency resurrection formatı: Direkt all_loras_ever dict'i
                self.all_loras_ever = checkpoint['all_loras_ever']
                print(f"   ✅ Tüm zamanlar kaydı yüklendi (Emergency format: {len(self.all_loras_ever)} LoRA)")
            elif 'all_loras_summary' in checkpoint:
                # Önce summary'den yükle (ölüler için)
                for lora_id, summary in checkpoint['all_loras_summary'].items():
                    # Yaşayanları bul (eğer varsa)
                    living_lora = None
                    for lora in self.evolution_manager.population:
                        if lora.id == lora_id:
                            living_lora = lora
                            break
                    
                    if living_lora:
                        # Yaşayan - objeyi kullan
                        self.all_loras_ever[lora_id] = {
                            'lora': living_lora,
                            'final_fitness': living_lora.get_recent_fitness(),
                            'current_match': summary.get('age', 0) + living_lora.birth_match,
                            'age': summary.get('age', 0),
                            'alive': True
                        }
                    else:
                        # Ölü - summary'den yükle (obje yok ama bilgi var!)
                        # NOT: LoRA objesini diriltmeden tutamayız, sadece bilgiyi saklarız
                        # Export sırasında bu bilgi kullanılacak
                        pass
                
                print(f"   ✅ Tüm zamanlar kaydı yüklendi ({len(self.all_loras_ever)} LoRA)")
            
            # 🌍 EVRİMLEŞEN DOĞAYI YÜKLE!
            if 'adaptive_nature' in checkpoint:
                adaptive_data = checkpoint['adaptive_nature']
                self.adaptive_nature.nature_version = adaptive_data.get('version', 1)
                self.adaptive_nature.evolution_history = adaptive_data.get('evolution_history', [])
                self.adaptive_nature.lora_immunity = adaptive_data.get('lora_immunity', {})
                self.adaptive_nature.nature_memory = adaptive_data.get('nature_memory', {})
                
                print(f"   ✅ Evrimleşen Doğa yüklendi (V{self.adaptive_nature.nature_version})")
                print(f"   🧬 {len(self.adaptive_nature.lora_immunity)} LoRA'nın bağışıklık kaydı")
            
            # 🛡️ DENEYİM DİRENCİNİ YÜKLE!
            if 'experience_resistance' in checkpoint:
                resist_data = checkpoint['experience_resistance']
                self.experience_resistance.lora_resistances = resist_data.get('lora_resistances', {})
                
                print(f"   ✅ Deneyim Direnci yüklendi ({len(self.experience_resistance.lora_resistances)} LoRA)")
            
            # 🔍 DİNAMİK UZMANLIĞI YÜKLE!
            if 'dynamic_specialization' in checkpoint:
                spec_data = checkpoint['dynamic_specialization']
                self.dynamic_spec.discovered_patterns = spec_data.get('discovered_patterns', {})
                
                print(f"   ✅ Dinamik Uzmanlık yüklendi ({len(self.dynamic_spec.discovered_patterns)} LoRA)")
            
            # 🛡️ DENEYİM DİRENCİNİ YÜKLE!
            if 'experience_resistance' in checkpoint:
                resist_data = checkpoint['experience_resistance']
                self.experience_resistance.lora_resistances = resist_data.get('lora_resistances', {})
                
                print(f"   ✅ Deneyim Direnci yüklendi ({len(self.experience_resistance.lora_resistances)} LoRA)")
            
            # 🔍 DİNAMİK UZMANLIĞI YÜKLE!
            if 'dynamic_specialization' in checkpoint:
                spec_data = checkpoint['dynamic_specialization']
                self.dynamic_spec.discovered_patterns = spec_data.get('discovered_patterns', {})
                
                print(f"   ✅ Dinamik Uzmanlık yüklendi ({len(self.dynamic_spec.discovered_patterns)} LoRA)")
            
            # 🧠 META-ADAPTIF ÖĞRENME YÜKLE!
            if 'meta_learning' in checkpoint:
                ml_data = checkpoint['meta_learning']
                self.meta_learning.learning_rates = ml_data.get('learning_rates', {})
                self.meta_learning.lr_history = ml_data.get('lr_history', {})
                self.meta_learning.performance_history = ml_data.get('performance_history', {})
                
                print(f"   ✅ Meta-Adaptif Öğrenme yüklendi ({len(self.meta_learning.learning_rates)} LoRA)")
            
            # 👻 GHOST FIELDS YÜKLE!
            if 'ghost_fields' in checkpoint:
                ghost_data = checkpoint['ghost_fields']
                self.ghost_fields.ghost_influence = ghost_data.get('ghost_influence', {})
                
                print(f"   ✅ Ghost Fields yüklendi ({len(self.ghost_fields.ghost_influence)} hayalet)")
            
            # 🌊 FLUID TEMPERAMENT YÜKLE!
            if 'fluid_temperament' in checkpoint:
                fluid_data = checkpoint['fluid_temperament']
                self.fluid_temperament.temperament_dynamics = fluid_data.get('temperament_dynamics', {})
                
                print(f"   ✅ Fluid Temperament yüklendi ({len(self.fluid_temperament.temperament_dynamics)} LoRA)")
            
            # 🌡️ NATURE THERMOSTAT YÜKLE!
            if 'nature_thermostat' in checkpoint:
                thermo_data = checkpoint['nature_thermostat']
                self.nature_thermostat.temperature = thermo_data.get('temperature', 0.5)
                self.nature_thermostat.entropy_history = thermo_data.get('entropy_history', [])
                
                print(f"   ✅ Nature Thermostat yüklendi (Sıcaklık: {self.nature_thermostat.temperature:.2f})")
            
            # ============================================
            # SİSTEMLERİ BİRBİRİNE BAĞLA! (YÜKLEME SONRASI!)
            # ============================================
            self.evolution_manager.experience_resistance = self.experience_resistance
            self.evolution_manager.ultra_mating = self.ultra_mating

            print(f"   ✅ Sistemler birbirine bağlandı!")
            
            # Yaşamayanları da ekle (sadece yaşayanlar yüklendi)
            for lora in self.evolution_manager.population:
                if lora.id not in self.all_loras_ever:
                    self.all_loras_ever[lora.id] = {
                        'lora': lora,
                        'final_fitness': lora.get_recent_fitness(),
                        'current_match': 0,
                        'age': 0,
                        'alive': True
                    }
        else:
            print("   ⚠️ Kaydedilmiş LoRA bulunamadı, yeni popülasyon oluşturulacak")
        
        # Buffer yükle
        if os.path.exists(self.paths['buffer']):
            # Yeni buffer oluştur ve yükle
            temp_buffer = ReplayBuffer(max_size=self.config.get('buffer', {}).get('max_size', 1000))
            temp_buffer.load(self.paths['buffer'])
            self.buffer = temp_buffer
            print(f"   ✅ Buffer yüklendi ({len(self.buffer)} örnek)")
        
        # Meta-LoRA yükle
        if os.path.exists(self.paths['meta_lora']) and isinstance(self.meta_lora, MetaLoRA):
            self.meta_lora.load_state_dict(torch.load(self.paths['meta_lora'], weights_only=False))
            print(f"   ✅ Meta-LoRA yüklendi")
        
        print("   ✅ Tüm durum yüklendi!")


def main():
    parser = argparse.ArgumentParser(description='Evrimsel Öğrenme Sistemi')
    parser.add_argument('--config', type=str, default='evolutionary_config.yaml', help='Config dosyası')
    parser.add_argument('--csv', type=str, default='prediction_matches.csv', 
                        help='Tahmin dosyası (SONUÇSUZ maçlar)')
    parser.add_argument('--results', type=str, default='results_matches.csv',
                        help='Sonuç dosyası (GERÇEK SONUÇLAR)')
    parser.add_argument('--start', type=int, default=0, help='Başlangıç maçı')
    parser.add_argument('--max', type=int, default=None, help='Maksimum maç sayısı')
    parser.add_argument('--resume', action='store_true', help='Kaydedilmiş durumdan devam et')
    parser.add_argument('--load-legends', action='store_true', help='🏆 Mucize LoRA\'ları yükle (Hall of Fame)')
    parser.add_argument('--resurrect', action='store_true', help='⚡ Top 50 LoRA\'ları dirilt (Soy tükenmesi)')
    
    args = parser.parse_args()
    
    # Sistemi başlat
    system = EvolutionaryLearningSystem(config_path=args.config)
    
    # KOLONİ MANTIĞI: Otomatik yükle (varsa)
    import os
    model_exists = os.path.exists(system.paths['lora_population'])
    
    if model_exists:
        print("\n🏛️ KOLONİ BULUNDU! Kaydedilmiş durumdan devam ediliyor...")
        system.load_state()
        # Resume için start'ı güncelle!
        args.start = system.evolution_manager.match_count
        print(f"   🔄 Kaldığı yerden devam ediyor: Maç #{args.start}")
    elif args.resume:
        print("\n⚠️ Resume istendi ama kayıt bulunamadı, yeni koloni başlatılıyor...")
    else:
        print("\n🐣 YENİ KOLONİ BAŞLATILIYOR!")
    
    # ⚡ DİRİLTME (SOY TÜKENMESİ İÇİN!) - V2 (3 Aşamalı)
    if args.resurrect:
        print("\n" + "⚡"*40)
        print("DİRİLTME KOMUTU! 3 Aşamalı Sistem!")
        print("⚡"*40)
        
        from lora_system.resurrection_system_v2 import ResurrectionSystemV2
        res_system = ResurrectionSystemV2()
        
        current_alive = len(system.evolution_manager.population)
        
        resurrected, stats = res_system.resurrect_to_50(
            current_population=current_alive,
            target=250,  # 🌊 BÜYÜK HEDEF!
            export_dir="en_iyi_loralar",
            miracle_dir="mucizeler",
            device=system.device  # 🔧 DEVICE PARAMETRES İNİ GEÇTIK!
        )
        
        if resurrected:
            # Mevcut yaşayanlarla birleştir!
            system.evolution_manager.population.extend(resurrected)
            
            print(f"\n📔 WALLET SİSTEMİ:")
            print(f"   📚 Ortak Hafıza: KORUNDU!")
            
            # 📝 WALLET KAYITLARINI OLUŞTUR/GÜNCELLE
            from datetime import datetime
            
            for lora in resurrected:
                wallet_dir = "lora_wallets"
                os.makedirs(wallet_dir, exist_ok=True)
                wallet_file_path = os.path.join(wallet_dir, f"{lora.id}.txt")
                
                # Spawn edilenler: Dengeli, Uç, veya Alien
                is_spawn = ("Balanced_" in lora.name or 
                           "Alien_" in lora.name or
                           any(arch in lora.name for arch in ["ZenMaster", "MadWarrior", "LoneWolf", 
                                                                "SocialButterfly", "ContrarianRebel", 
                                                                "Perfectionist", "Gambler", "Analyst",
                                                                "Optimist", "Pessimist", "ChaosAgent", "HypeBeast"]))
                
                if is_spawn:
                    # TİP BELİRLE
                    archetype_name = "Bilinmiyor"
                    archetype_desc = "Özel kişilik profili"
                    archetype_emoji = "❓"
                    
                    # Balanced mi?
                    if "Balanced_" in lora.name:
                        archetype_type = "Dengeli Normal İnsan"
                        archetype_emoji = "⚖️"
                        # "Balanced_ZenMaster" → "Zen Master"
                        name_parts = lora.name.replace("Balanced_", "").split('_')
                        if len(name_parts) >= 1:
                            potential_archetype = name_parts[0]
                    # Gerçek Alien mi? (sadece rakam)
                    elif lora.name.startswith("Alien_") and lora.name.split('_')[1].isdigit():
                        archetype_type = "ALIEN (Nörotipik Farklılık)"
                        archetype_emoji = "👽"
                        archetype_name = "Alien"
                        archetype_desc = "Hiçbir arketipe uymuyor, tamamen rastgele, otizm spektrum, nörotipik bozukluk"
                        potential_archetype = None
                    # Uç karakter (arketip adı direkt)
                    else:
                        archetype_type = "Uç Karakter"
                        archetype_emoji = "🎭"
                        name_parts = lora.name.split('_')
                        if len(name_parts) >= 1:
                            potential_archetype = name_parts[0]
                        else:
                            potential_archetype = None
                    
                    # Arketip listesinden detay bul
                    if potential_archetype and archetype_name == "Bilinmiyor":
                        from lora_system.lora_archetypes import LoRAArchetypes
                        for key, data in LoRAArchetypes.ARCHETYPES.items():
                            if data['name'].replace(' ', '') == potential_archetype:
                                archetype_name = data['name']
                                archetype_desc = data['description']
                                archetype_emoji = data['emoji']
                                break
                    
                    # YENİ SPAWN WALLET OLUŞTUR
                    with open(wallet_file_path, 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write(f"{archetype_emoji} {archetype_type.upper()} - KİŞİSEL CÜZDANI\n")
                        f.write("="*80 + "\n")
                        f.write(f"İsim: {lora.name}\n")
                        f.write(f"ID: {lora.id}\n")
                        f.write(f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Tip: SPAWN ({archetype_type})\n")
                        if archetype_name != "Bilinmiyor":
                            f.write(f"Arketip: {archetype_name}\n")
                            f.write(f"Özellik: {archetype_desc}\n")
                        f.write("="*80 + "\n\n")
                        
                        # Kişilik profili
                        f.write("🎭 KİŞİLİK PROFİLİ:\n")
                        f.write("─"*80 + "\n")
                        temp = lora.temperament
                        f.write(f"  • Sabır: {temp.get('patience', 0.5):.2f}\n")
                        f.write(f"  • Risk Toleransı: {temp.get('risk_tolerance', 0.5):.2f}\n")
                        f.write(f"  • Stres Dayanımı: {temp.get('stress_tolerance', 0.5):.2f}\n")
                        f.write(f"  • Dürtüsellik: {temp.get('impulsiveness', 0.5):.2f}\n")
                        f.write(f"  • Bağımsızlık: {temp.get('independence', 0.5):.2f}\n")
                        f.write(f"  • Sosyal Zeka: {temp.get('social_intelligence', 0.5):.2f}\n")
                        f.write(f"  • Sürü Eğilimi: {temp.get('herd_tendency', 0.5):.2f}\n")
                        f.write(f"  • Karşıt Skor: {temp.get('contrarian_score', 0.5):.2f}\n")
                        f.write(f"  • Hırs: {temp.get('ambition', 0.5):.2f}\n")
                        f.write("─"*80 + "\n\n")
                        
                        # Başlangıç mesajı
                        if "Alien_" in lora.name and lora.name.split('_')[1].isdigit():
                            f.write("💬 Ben farklıyım. Hiçbir şablona uymuyorum. Kendi yolumu bulacağım!\n\n")
                        elif "Balanced_" in lora.name:
                            f.write("💬 Dengeli bir yaklaşımla başlıyorum. Orta yol benim yolum!\n\n")
                        else:
                            f.write("💬 Yeni bir dünyadayım. Kendi yolumu bulacağım!\n\n")
                        
                        # Mizaç detayları (SAFE GET!)
                        f.write("🎭 KİŞİLİK PROFİLİ:\n")
                        f.write("─"*80 + "\n")
                        temp = lora.temperament
                        f.write(f"  • Sabır: {temp.get('patience', 0.5):.2f}\n")
                        f.write(f"  • Risk Toleransı: {temp.get('risk_tolerance', 0.5):.2f}\n")
                        f.write(f"  • Stres Dayanımı: {temp.get('stress_tolerance', 0.5):.2f}\n")
                        f.write(f"  • Dürtüsellik: {temp.get('impulsiveness', 0.5):.2f}\n")
                        f.write(f"  • Bağımsızlık: {temp.get('independence', 0.5):.2f}\n")
                        f.write(f"  • Sosyal Zeka: {temp.get('social_intelligence', 0.5):.2f}\n")
                        f.write(f"  • Sürü Eğilimi: {temp.get('herd_tendency', 0.5):.2f}\n")
                        f.write(f"  • Karşıt Skor: {temp.get('contrarian_score', 0.5):.2f}\n")
                        f.write(f"  • Hırs: {temp.get('ambition', 0.5):.2f}\n")
                        f.write("─"*80 + "\n\n")
                        
                        f.write("💬 Yeni bir dünyadayım. Kendi yolumu bulacağım!\n\n")
                else:
                    # DİRİLEN LoRA - ESKİ WALLET'A EKLE
                    with open(wallet_file_path, 'a', encoding='utf-8') as f:
                        f.write("\n\n")
                        f.write("═" * 100 + "\n")
                        f.write(f"⚡⚡⚡ DİRİLME! ({lora.resurrection_count}. kez) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ⚡⚡⚡\n")
                        f.write("═" * 100 + "\n")
                        f.write(f"Yeni isim: {lora.name}\n")
                        
                        if hasattr(lora, 'original_fitness'):
                            f.write(f"Önceki final fitness: {lora.original_fitness:.3f}\n")
                        if hasattr(lora, 'was_dead'):
                            status = "Öldü" if lora.was_dead else "Yaşıyordu"
                            f.write(f"Önceki durum: {status}\n")
                        
                        f.write("\n💬 Yeni başlangıç! Türü kurtarmak için savaşacağım!\n")
                        f.write("═" * 100 + "\n\n")
                
                # Wallet manager'a register et
                wallet = system.wallet_manager.get_or_create_wallet(lora, system.evolution_manager.population)
            
            print(f"   ✅ {stats['from_miracles']} Mucize LoRA (wallet korundu - en öncelikli!)")
            print(f"   ✅ {stats['from_top_list']} Scoreboard LoRA (wallet güncellendi)")
            print(f"   ✅ {stats['balanced_spawned']} Dengeli LoRA (yeni wallet - normal insanlar)")
            print(f"   ✅ {stats['extreme_spawned']} Uç LoRA (yeni wallet - ekstrem arketip)")
            print(f"   ✅ {stats.get('alien_spawned', 0)} Alien LoRA (yeni wallet - nörotipik farklılık)")
            
            # 📊 EXCEL'E DİRİLTME KAYDI EKLE!
            print(f"\n📊 Excel'e diriltme dönemi kaydediliyor...")
            system.logger.log_resurrection_era(resurrected, stats)
            print(f"   ✅ Diriltme dönemi Excel'e kaydedildi!")
            
            # 📅 ÖLÜM RAPORUNA DÖNEM AYIRICI EKLE!
            additional_info = f"{len(resurrected)} LoRA dirildi/spawn edildi"
            system.logger.log_era_separator_to_death_report(
                era_type="Resurrection",
                match_start=system.evolution_manager.match_count,
                additional_info=additional_info
            )
            
            # ✅ DİRİLTME BİTTİ! STATE KAYDET VE BİTİR!
            print(f"\n💾 State kaydediliyor...")
            system.save_state()
            system.logger.save_all()
            
            print(f"\n{'⚡'*80}")
            print(f"✅ DİRİLTME TAMAMLANDI VE KAYDEDİLDİ!")
            print(f"{'⚡'*80}")
            print(f"\n🚀 Şimdi maç oynamak için:")
            print(f"   python run_evolutionary_learning.py --csv prediction_matches.csv --results results_matches.csv --max 100")
            print(f"\n{'⚡'*80}\n")
            
            return  # ✅ PROGRAMI BİTİR! Maç oynama!
        else:
            print("   ⚠️ Diriltilecek LoRA bulunamadı. Önce bir test çalıştır!")
            return  # ✅ BİTİR!
        
        print("⚡"*40 + "\n")
    
    # 🏆 MUCİZE LoRA'LARI YÜKLE (SADECE MANUEL!)
    # Kullanım: Herkes öldüyse veya sıfırdan başlamak istersen --load-legends
    if args.load_legends:
        print("\n" + "🏆"*40)
        print("HALL OF FAME: MUCİZE LoRA'LAR YÜKLENİYOR!")
        print("🏆"*40)
        
        legends = system.miracle_system.load_all_miracles(device=system.device)
        
        if legends:
            # Mevcut popülasyona ekle
            system.evolution_manager.population.extend(legends)
            print(f"   ✅ {len(legends)} Mucize LoRA popülasyona eklendi!")
            print(f"   📊 Yeni popülasyon: {len(system.evolution_manager.population)} LoRA")
            print(f"\n{system.miracle_system.get_miracle_summary()}")
            
            # State kaydet
            print(f"\n💾 State kaydediliyor...")
            system.save_state()
            system.logger.save_all()
            
            print(f"\n{'🏆'*80}")
            print(f"✅ MUCİZE LoRA'LAR YÜKLENDİ VE KAYDEDİLDİ!")
            print(f"{'🏆'*80}\n")
            
            return  # ✅ BİTİR!
        else:
            print("   ⚠️ Henüz mucize LoRA yok.")
            return  # ✅ BİTİR!
        
        print("🏆"*40 + "\n")
    
    # 📅 NORMAL RUN İÇİN DÖNEM AYIRICI EKLE!
    # (Her yeni başlatmada bu çağrılır)
    print(f"\n📅 Yeni dönem başlıyor, ölüm raporuna separator ekleniyor...")
    population_info = f"{len(system.evolution_manager.population)} LoRA ile başlanıyor"
    system.logger.log_era_separator_to_death_report(
        era_type="Normal Run",
        match_start=system.evolution_manager.match_count,
        additional_info=population_info
    )
    
    # Çalıştır
    system.run(csv_path=args.csv, start_match=args.start, max_matches=args.max,
               results_csv=args.results)


if __name__ == "__main__":
    main()

