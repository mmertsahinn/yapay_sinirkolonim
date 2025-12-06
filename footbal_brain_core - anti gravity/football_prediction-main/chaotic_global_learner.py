"""
KAOTİK GLOBAL ÖĞRENME SİSTEMİ
Chaos Theory + Temporal Dynamics + Global Context

VİZYON:
- Son 10 gündeki DÜNYA genelindeki büyük takım maçlarını analiz et
- Hype dalgaları, momentum, global shocklar
- Yeni pattern'lere isim ver, keşfet
- Kaotik düzenden düzen çıkar
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import signal, stats
from scipy.fft import fft, ifft
from sklearn.cluster import DBSCAN
import joblib

class ChaoticGlobalLearner:
    """
    Kaotik Sistemler + Global Temporal Analysis
    
    Yeni Kavramlar:
    1. GLOBAL MOMENTUM WAVE - Dünya genelinde hype/sürpriz dalgası
    2. TEMPORAL CHAOS INDEX - Belirsizlik ölçüsü
    3. ELITE TEAM CORRELATION - Büyük takımlar arası bağıntı
    4. UNEXPECTED PATTERN NAMING - Yeni fenomenlere isim verme
    5. HYPE PROPAGATION - Hype'ın yayılma dinamiği
    """
    
    def __init__(self):
        # Elite teams (Big 50)
        self.elite_teams = [
            'Real Madrid', 'Barcelona', 'Bayern München', 'Liverpool', 'Manchester City',
            'Paris Saint-Germain', 'Juventus', 'Inter', 'Milan', 'Chelsea',
            'Arsenal', 'Manchester Utd', 'Tottenham', 'Atletico Madrid', 'Dortmund',
            'Leverkusen', 'Leipzig', 'Napoli', 'Roma', 'Lazio',
            'Sevilla', 'Valencia', 'Galatasaray', 'Fenerbahce', 'Besiktas',
            # ... 50 total
        ]
        
        # Global state tracking
        self.global_momentum = 0.0  # [-1, +1] range
        self.chaos_index = 0.0      # [0, 1] range (0=predictable, 1=chaotic)
        
        # Temporal windows
        self.window_10d = []  # Son 10 gün
        self.window_30d = []  # Son 30 gün
        
        # Pattern database
        self.discovered_patterns = {}
        self.pattern_counter = 0
        
        # Correlation tensors (3D: time × feature × feature)
        self.temporal_correlation = []
        
        # Hype propagation graph
        self.hype_network = {}
        
        # HISTORY - Tüm global events
        self.history = []
        
        print("🌍 Chaotic Global Learning System initialized")
    
    # ========================================================================
    # GLOBAL TEMPORAL WINDOW ANALYSIS
    # ========================================================================
    
    def update_temporal_windows(self, df_all, current_date):
        """
        Son 10 gün ve 30 gündeki GLOBAL maçları analiz et
        
        Returns:
            window_10d: Son 10 gündeki tüm elite maçlar
            window_30d: Son 30 gündeki tüm elite maçlar
        """
        
        date_10d = current_date - timedelta(days=10)
        date_30d = current_date - timedelta(days=30)
        
        # Elite takımların maçları
        mask_elite = (
            df_all['home_team'].isin(self.elite_teams) |
            df_all['away_team'].isin(self.elite_teams)
        )
        
        df_elite = df_all[mask_elite]
        
        self.window_10d = df_elite[
            (df_elite['date'] >= date_10d) & 
            (df_elite['date'] < current_date)
        ]
        
        self.window_30d = df_elite[
            (df_elite['date'] >= date_30d) & 
            (df_elite['date'] < current_date)
        ]
        
        print(f"   📅 10-day window: {len(self.window_10d)} elite matches")
        print(f"   📅 30-day window: {len(self.window_30d)} elite matches")
    
    # ========================================================================
    # GLOBAL MOMENTUM WAVE
    # ========================================================================
    
    def calculate_global_momentum(self):
        """
        Dünya genelinde momentum dalgası
        
        Momentum = Σ (unexpected_results × hype_magnitude) / N
        
        unexpected_results: Favori kaybetti mi?
        hype_magnitude: Maçın hype seviyesi
        """
        
        if len(self.window_10d) == 0:
            return 0.0
        
        momentum_sum = 0.0
        
        for _, match in self.window_10d.iterrows():
            # Favori kim? (odds'a göre)
            if pd.notna(match.get('odds_b365_h')) and pd.notna(match.get('odds_b365_a')):
                odds_h = match['odds_b365_h']
                odds_a = match['odds_b365_a']
                
                favorite = 'home' if odds_h < odds_a else 'away'
                
                # Gerçek sonuç
                home_goals = match.get('home_goals', 0)
                away_goals = match.get('away_goals', 0)
                
                if home_goals > away_goals:
                    actual = 'home'
                elif away_goals > home_goals:
                    actual = 'away'
                else:
                    actual = 'draw'
                
                # Sürpriz mi?
                surprise = 1.0 if (favorite != actual and actual != 'draw') else 0.0
                
                # Hype magnitude
                hype_mag = match.get('total_tweets', 0) / 1000.0  # Normalize
                
                momentum_sum += surprise * hype_mag
        
        self.global_momentum = np.tanh(momentum_sum / len(self.window_10d))
        
        return self.global_momentum
    
    # ========================================================================
    # CHAOS INDEX (Lyapunov Exponent benzeri)
    # ========================================================================
    
    def calculate_chaos_index(self):
        """
        Son 10 günün ne kadar kaotik/tahmin edilemez olduğunu ölç
        
        Chaos Index = Entropy + Variance + Surprise_Rate
        
        Lyapunov-inspired:
        λ = (1/N) × Σ ln(|actual - expected|)
        """
        
        if len(self.window_10d) < 5:
            return 0.0
        
        surprises = []
        entropies = []
        
        for _, match in self.window_10d.iterrows():
            # Odds'dan beklenen
            if pd.notna(match.get('odds_b365_h')):
                odds_h = match.get('odds_b365_h', 3.0)
                odds_d = match.get('odds_b365_d', 3.5)
                odds_a = match.get('odds_b365_a', 3.0)
                
                # Implied probabilities
                total = (1/odds_h + 1/odds_d + 1/odds_a)
                p_h = (1/odds_h) / total
                p_d = (1/odds_d) / total
                p_a = (1/odds_a) / total
                
                # Entropy (belirsizlik)
                probs = np.array([p_h, p_d, p_a])
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
                
                # Gerçek sonuç
                home_goals = match.get('home_goals', 0)
                away_goals = match.get('away_goals', 0)
                
                if home_goals > away_goals:
                    actual_p = p_h
                elif away_goals > home_goals:
                    actual_p = p_a
                else:
                    actual_p = p_d
                
                # Surprise: Gerçekleşen olasılığı düşükse yüksek surprise
                surprise = 1.0 - actual_p
                surprises.append(surprise)
        
        if not surprises:
            return 0.0
        
        # Chaos Index: Normalize edilmiş entropy + surprise
        avg_entropy = np.mean(entropies) / np.log(3)  # [0, 1] normalize
        avg_surprise = np.mean(surprises)
        
        self.chaos_index = (avg_entropy + avg_surprise) / 2.0
        
        return self.chaos_index
    
    # ========================================================================
    # HYPE PROPAGATION NETWORK
    # ========================================================================
    
    def analyze_hype_propagation(self):
        """
        Hype nasıl yayılıyor? Network analizi
        
        Graph: Teams = Nodes, Hype transfer = Edges
        
        If Team A plays Team B:
            Hype_A → Hype_B transfer?
        
        Dijkstra-style shortest path for hype influence
        """
        
        if len(self.window_10d) < 10:
            return {}
        
        # Hype transfer matrix
        hype_transfers = {}
        
        for _, match in self.window_10d.iterrows():
            home = match['home_team']
            away = match['away_team']
            
            if pd.notna(match.get('home_support')) and pd.notna(match.get('away_support')):
                home_hype = match['home_support']
                away_hype = match['away_support']
                total_hype = match.get('total_tweets', 0)
                
                # Hype transfer coefficient
                transfer = abs(home_hype - away_hype) * total_hype / 1000.0
                
                # Directed edge
                if home not in hype_transfers:
                    hype_transfers[home] = {}
                if away not in hype_transfers:
                    hype_transfers[away] = {}
                
                hype_transfers[home][away] = hype_transfers[home].get(away, 0) + transfer
                hype_transfers[away][home] = hype_transfers[away].get(home, 0) + transfer
        
        self.hype_network = hype_transfers
        
        return hype_transfers
    
    # ========================================================================
    # PATTERN DISCOVERY & NAMING
    # ========================================================================
    
    def discover_and_name_pattern(self, features, error_magnitude, context):
        """
        Yeni pattern keşfet ve isim ver
        
        Clustering + Pattern Recognition + Automatic Naming
        """
        
        if len(self.history) < 30:
            return None
        
        # Son hataların feature'larını topla
        recent_errors = [h for h in self.history[-100:] if not h['correct']]
        
        if len(recent_errors) < 10:
            return None
        
        X_errors = np.array([e['features'] for e in recent_errors])
        
        # DBSCAN clustering - Yoğun hata bölgeleri
        clustering = DBSCAN(eps=0.5, min_samples=3)
        labels = clustering.fit_predict(X_errors)
        
        # Yeni cluster bulundu mu?
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            # Bu cluster daha önce isimlendirilmiş mi?
            cluster_key = f"cluster_{label}"
            
            if cluster_key not in self.discovered_patterns:
                # YENİ PATTERN BULUNDU!
                
                cluster_points = X_errors[labels == label]
                cluster_center = cluster_points.mean(axis=0)
                
                # Pattern'i karakterize et
                dominant_features = np.argsort(np.abs(cluster_center))[-3:][::-1]
                
                # İSİM VER!
                pattern_name = self._generate_pattern_name(
                    cluster_center, 
                    dominant_features, 
                    context
                )
                
                self.discovered_patterns[cluster_key] = {
                    'name': pattern_name,
                    'center': cluster_center,
                    'dominant_features': dominant_features,
                    'discovery_date': datetime.now(),
                    'occurrence_count': len(cluster_points),
                    'severity': error_magnitude
                }
                
                print(f"\n🔍 YENİ PATTERN KEŞFEDİLDİ!")
                print(f"   İsim: {pattern_name}")
                print(f"   Oluşum sayısı: {len(cluster_points)}")
                print(f"   Dominant features: {dominant_features}")
                
                return pattern_name
        
        return None
    
    def _generate_pattern_name(self, center, dominant_features, context):
        """
        Otomatik pattern isimlendirme
        
        Örnek:
        - "HIGH_HYPE_LOW_ODDS_PARADOX"
        - "ELITE_UNDERDOG_SURGE"
        - "GLOBAL_MOMENTUM_REVERSAL"
        """
        
        feature_names = [
            'home_strength', 'away_strength', 'home_defense', 'away_defense',
            'home_xG', 'away_xG', 'xG_diff', 'home_form', 'away_form',
            'h2h_home', 'h2h_away', 'h2h_draw', 'goal_ratio', 'xG_ratio',
            'day', 'month', 'home_support', 'away_support', 'support_diff', 'support_ratio',
            'sentiment', 'sent_pos', 'sent_neg', 'tweets', 'log_tweets', 'high_hype', 'hype_score',
            'impl_h', 'impl_d', 'impl_a', 'hype_fav', 'odds_fav', 'alignment',
            'home_infl', 'away_infl', 'infl_score', 'high_engage', 'discrepancy', 'tweets_odds',
            'mkt_eff', 'entropy', 'consist_h', 'consist_a', 'consist_d',
            'diff_h', 'diff_a', 'diff_d', 'infl_score_h', 'infl_score_a', 'total_infl',
            'tweets_var', 'tweets_per', 'impl_over', 'impl_under', 'tweets_over',
            'ah_h', 'ah_a', 'hype_ah_h', 'hype_ah_a', 'consensus'
        ]
        
        name_parts = []
        
        # Global context
        if context.get('global_momentum', 0) > 0.5:
            name_parts.append("GLOBAL_SURGE")
        elif context.get('global_momentum', 0) < -0.5:
            name_parts.append("GLOBAL_DECLINE")
        
        # Chaos level
        if context.get('chaos_index', 0) > 0.7:
            name_parts.append("HIGH_CHAOS")
        elif context.get('chaos_index', 0) < 0.3:
            name_parts.append("STABLE")
        
        # Dominant feature characteristics
        for feat_idx in dominant_features[:2]:
            if feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
                feat_val = center[feat_idx]
                
                if 'hype' in feat_name.lower() and feat_val > 0.6:
                    name_parts.append("HIGH_HYPE")
                elif 'hype' in feat_name.lower() and feat_val < 0.3:
                    name_parts.append("LOW_HYPE")
                
                if 'xG' in feat_name and feat_val > 2.0:
                    name_parts.append("HIGH_XG")
                
                if 'odds' in feat_name.lower() or 'impl' in feat_name.lower():
                    if feat_val < 0.4:
                        name_parts.append("UNDERDOG")
                    elif feat_val > 0.6:
                        name_parts.append("FAVORITE")
        
        # Combine
        if not name_parts:
            name_parts = [f"PATTERN_{self.pattern_counter}"]
            self.pattern_counter += 1
        
        pattern_name = "_".join(name_parts) + "_ANOMALY"
        
        return pattern_name
    
    # ========================================================================
    # FOURIER ANALYSIS - TEMPORAL PATTERNS
    # ========================================================================
    
    def fourier_temporal_analysis(self, time_series):
        """
        Zaman serisinde periyodik pattern'ler var mı?
        
        FFT (Fast Fourier Transform) ile frekans analizi
        
        Örnek:
        - Haftalık döngü? (7 gün periyot)
        - Momentum dalgaları?
        """
        
        if len(time_series) < 20:
            return None
        
        # FFT
        fft_result = fft(time_series)
        frequencies = np.fft.fftfreq(len(time_series))
        
        # Power spectrum
        power = np.abs(fft_result)**2
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1  # Skip DC component
        dominant_freq = frequencies[dominant_freq_idx]
        
        if abs(dominant_freq) > 0:
            period = 1.0 / abs(dominant_freq)
        else:
            period = np.inf
        
        return {
            'dominant_frequency': dominant_freq,
            'period_days': period,
            'power_spectrum': power
        }
    
    # ========================================================================
    # ELITE TEAM CORRELATION MATRIX
    # ========================================================================
    
    def build_elite_correlation_matrix(self):
        """
        Büyük takımlar arası korelasyon
        
        Corr(Team_i, Team_j) = Cov(Results_i, Results_j) / (σ_i × σ_j)
        
        Örnek:
        - Real Madrid kaybedince Barcelona kazanır mı?
        - Bayern şampiyon olunca PSG de mi olur?
        - Global pattern'ler
        """
        
        if len(self.window_30d) < 20:
            return None
        
        n_elite = len(self.elite_teams)
        corr_matrix = np.zeros((n_elite, n_elite))
        
        # Her takım için son 30 gündeki performans vektörü
        team_performance = {}
        
        for team in self.elite_teams:
            matches = self.window_30d[
                (self.window_30d['home_team'] == team) |
                (self.window_30d['away_team'] == team)
            ]
            
            results = []
            for _, match in matches.iterrows():
                if match['home_team'] == team:
                    score = 1 if match['home_goals'] > match['away_goals'] else (0 if match['home_goals'] == match['away_goals'] else -1)
                else:
                    score = 1 if match['away_goals'] > match['home_goals'] else (0 if match['away_goals'] == match['home_goals'] else -1)
                results.append(score)
            
            team_performance[team] = results
        
        # Correlation matrix
        for i, team_i in enumerate(self.elite_teams):
            for j, team_j in enumerate(self.elite_teams):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    perf_i = team_performance.get(team_i, [])
                    perf_j = team_performance.get(team_j, [])
                    
                    if len(perf_i) > 2 and len(perf_j) > 2:
                        # Pearson correlation
                        min_len = min(len(perf_i), len(perf_j))
                        if min_len >= 3:
                            corr, _ = stats.pearsonr(perf_i[:min_len], perf_j[:min_len])
                            corr_matrix[i, j] = corr
        
        return corr_matrix
    
    # ========================================================================
    # ANOMALY DETECTION & PREDICTION ADJUSTMENT
    # ========================================================================
    
    def detect_anomaly_and_adjust(self, features, base_proba, context):
        """
        Anomali tespit et ve tahmini ayarla
        
        Mahalanobis Distance kullanarak anomaly detection:
        D² = (x - μ)^T × Σ^{-1} × (x - μ)
        
        D² > threshold → ANOMALY!
        """
        
        if len(self.history) < 50:
            return base_proba, []
        
        # Normal maçların feature mean ve covariance
        normal_matches = [h for h in self.history if h['correct']]
        
        if len(normal_matches) < 20:
            return base_proba, []
        
        X_normal = np.array([m['features'] for m in normal_matches])
        μ = X_normal.mean(axis=0)
        Σ = np.cov(X_normal.T)
        
        # Mahalanobis distance
        try:
            Σ_inv = inv(Σ + np.eye(len(Σ)) * 1e-6)  # Regularization
            diff = features - μ
            mahalanobis = np.sqrt(diff @ Σ_inv @ diff)
            
            # Threshold (Chi-squared distribution, 99% confidence)
            threshold = stats.chi2.ppf(0.99, df=len(features))
            
            if mahalanobis > threshold:
                # ANOMALY DETECTED!
                anomaly_type = "UNKNOWN_SCENARIO"
                
                # Pattern database'de var mı?
                for pattern_key, pattern in self.discovered_patterns.items():
                    pattern_dist = np.linalg.norm(features - pattern['center'])
                    if pattern_dist < 1.0:  # Benzer pattern
                        anomaly_type = pattern['name']
                        break
                else:
                    # YENİ PATTERN! İsim ver
                    anomaly_type = self.discover_and_name_pattern(features, mahalanobis, context)
                
                # Anomaly durumunda güven azalt
                adjusted_proba = base_proba * 0.7  # %30 daha az güven
                adjusted_proba = adjusted_proba / adjusted_proba.sum()
                
                anomalies = [{
                    'type': anomaly_type,
                    'distance': mahalanobis,
                    'severity': 'HIGH' if mahalanobis > threshold * 1.5 else 'MEDIUM'
                }]
                
                return adjusted_proba, anomalies
        
        except np.linalg.LinAlgError:
            pass
        
        return base_proba, []
    
    # ========================================================================
    # FULL PREDICTION PIPELINE
    # ========================================================================
    
    def predict_with_global_context(self, features, base_proba, df_all, current_date):
        """
        GLOBAL CONTEXT ile tahmin
        
        Pipeline:
        1. Temporal windows güncelle (10d, 30d)
        2. Global momentum hesapla
        3. Chaos index hesapla
        4. Elite correlations
        5. Hype propagation
        6. Anomaly detection
        7. Final adjusted prediction
        """
        
        print(f"\n🌍 Global Context Analysis for {current_date.strftime('%Y-%m-%d')}")
        print("-" * 80)
        
        # 1. Update windows
        self.update_temporal_windows(df_all, current_date)
        
        # 2. Global momentum
        momentum = self.calculate_global_momentum()
        print(f"   Global Momentum: {momentum:+.3f} {'📈' if momentum > 0 else '📉'}")
        
        # 3. Chaos index
        chaos = self.calculate_chaos_index()
        print(f"   Chaos Index: {chaos:.3f} {'⚠️ HIGH' if chaos > 0.7 else '✓ Normal'}")
        
        # 4. Elite correlations
        elite_corr = self.build_elite_correlation_matrix()
        
        # 5. Hype propagation
        hype_net = self.analyze_hype_propagation()
        
        # 6. Anomaly detection
        context = {
            'global_momentum': momentum,
            'chaos_index': chaos,
            'elite_correlation': elite_corr,
            'hype_network': hype_net
        }
        
        adjusted_proba, anomalies = self.detect_anomaly_and_adjust(features, base_proba, context)
        
        if anomalies:
            for anomaly in anomalies:
                print(f"   ⚠️ ANOMALY: {anomaly['type']} (severity: {anomaly['severity']})")
        
        # 7. Global momentum adjustment
        if abs(momentum) > 0.3:
            # Yüksek momentum → Sürprizler daha muhtemel
            # Favoriye daha az güven
            max_idx = np.argmax(adjusted_proba)
            adjusted_proba[max_idx] *= (1.0 - abs(momentum) * 0.2)
            adjusted_proba = adjusted_proba / adjusted_proba.sum()
            print(f"   🌊 Momentum adjustment applied")
        
        # 8. Chaos adjustment
        if chaos > 0.7:
            # Yüksek chaos → Daha uniform distribution
            uniform = np.ones(3) / 3
            adjusted_proba = 0.7 * adjusted_proba + 0.3 * uniform
            print(f"   🌀 Chaos adjustment applied (reducing confidence)")
        
        return adjusted_proba, context
    
    # ========================================================================
    # SAVE/LOAD
    # ========================================================================
    
    def save(self, filename='chaotic_global_state.joblib'):
        """Tüm global öğrenme state'ini kaydet"""
        state = {
            'global_momentum': self.global_momentum,
            'chaos_index': self.chaos_index,
            'discovered_patterns': self.discovered_patterns,
            'hype_network': self.hype_network,
            'temporal_correlation': self.temporal_correlation,
            'history': self.history,
            'pattern_counter': self.pattern_counter
        }
        joblib.dump(state, filename)
        print(f"💾 Global learning state saved: {filename}")
    
    def load(self, filename='chaotic_global_state.joblib'):
        """Global öğrenme state'ini yükle"""
        try:
            state = joblib.load(filename)
            self.global_momentum = state['global_momentum']
            self.chaos_index = state['chaos_index']
            self.discovered_patterns = state['discovered_patterns']
            self.hype_network = state['hype_network']
            self.temporal_correlation = state['temporal_correlation']
            self.history = state['history']
            self.pattern_counter = state['pattern_counter']
            
            print(f"📂 Global state loaded!")
            print(f"   {len(self.discovered_patterns)} patterns discovered")
            print(f"   {len(self.history)} global events tracked")
            return True
        except:
            print("ℹ️  No global state found, starting fresh")
            return False

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Chaotic Global Learning System\n")
    
    learner = ChaoticGlobalLearner()
    
    # Simulated global data
    dates = pd.date_range('2024-07-01', '2024-11-30', freq='D')
    
    print("Simulating global football ecosystem...\n")
    
    for i, date in enumerate(dates[:30]):
        # Simulate 5 elite matches per day
        for _ in range(5):
            features = np.random.randn(58) * 0.5 + 1.5
            features = np.clip(features, 0, 5)
            
            base_proba = np.random.dirichlet([2, 1, 2])
            
            # Simulate full dataframe (minimal)
            df_sim = pd.DataFrame({
                'date': [date] * 10,
                'home_team': np.random.choice(learner.elite_teams, 10),
                'away_team': np.random.choice(learner.elite_teams, 10),
                'home_goals': np.random.randint(0, 4, 10),
                'away_goals': np.random.randint(0, 4, 10),
                'home_support': np.random.rand(10),
                'away_support': np.random.rand(10),
                'total_tweets': np.random.randint(100, 5000, 10),
                'odds_b365_h': np.random.uniform(1.5, 4.0, 10),
                'odds_b365_d': np.random.uniform(3.0, 4.0, 10),
                'odds_b365_a': np.random.uniform(1.5, 4.0, 10),
            })
            
            # Global context prediction
            adjusted_proba, context = learner.predict_with_global_context(
                features, base_proba, df_sim, date
            )
        
        if (i + 1) % 10 == 0:
            print(f"\nDay {i+1}: Global Momentum = {learner.global_momentum:+.3f}, Chaos = {learner.chaos_index:.3f}")
    
    print("\n✅ Simulation complete!")
    print(f"\n🔍 Discovered {len(learner.discovered_patterns)} unique patterns!")
    
    for key, pattern in learner.discovered_patterns.items():
        print(f"   - {pattern['name']} (occurred {pattern['occurrence_count']} times)")
    
    learner.save()

