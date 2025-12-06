"""
İLERİ DÜZEY INCREMENTAL LEARNING SİSTEMİ
MATLAB-tarzı ileri matematik kullanarak

Kullanılan Teknikler:
1. Kalman Filter - Dinamik state estimation
2. Bayesian Inference - Posterior güncelleme
3. Correlation Matrix Learning - Tüm özelliklerin birbiriyle ilişkisi
4. Gradient Descent - Hata minimizasyonu
5. Exponential Weighted Covariance - Zamanla değişen korelasyonlar
6. Meta-Learning - "Öğrenmeyi öğrenmek"
"""

import numpy as np
import pandas as pd
import joblib
from scipy import stats
from scipy.linalg import inv, cholesky
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedIncrementalLearner:
    """
    İleri Düzey Öğrenme Sistemi
    
    State Space Model:
    x_t = A × x_{t-1} + w_t     (state equation)
    y_t = H × x_t + v_t         (observation equation)
    
    w_t ~ N(0, Q)  (process noise)
    v_t ~ N(0, R)  (measurement noise)
    """
    
    def __init__(self, n_features=58):
        self.n_features = n_features
        
        # ===== KALMAN FILTER STATE =====
        self.state = np.zeros(n_features)  # x_t: Feature importance state
        self.P = np.eye(n_features) * 10   # Covariance matrix
        self.Q = np.eye(n_features) * 0.01 # Process noise
        self.R = 0.1                        # Measurement noise
        
        # ===== CORRELATION LEARNING =====
        self.correlation_matrix = np.eye(n_features)  # C_ij: feature i ile j arası korelasyon
        self.error_correlation = np.zeros((n_features, 3))  # Her feature'ın her sonuçla hatası
        
        # ===== BAYESIAN PRIORS =====
        self.prior_alpha = np.ones(3)  # Dirichlet prior [home, draw, away]
        self.posterior_alpha = np.ones(3)
        
        # ===== GRADIENT LEARNING =====
        self.feature_weights = np.ones(n_features)  # w_i: Her feature'ın ağırlığı
        self.learning_rate = 0.01
        self.momentum = np.zeros(n_features)
        self.beta_momentum = 0.9  # Momentum coefficient
        
        # ===== META-LEARNING =====
        self.meta_params = {
            'learning_rate_schedule': [],
            'optimal_lr': 0.01,
            'adaptation_speed': 1.0
        }
        
        # ===== HISTORY =====
        self.history = []
        self.error_patterns = {}
        self.n_updates = 0
        
        print("🧠 Advanced Incremental Learning System initialized")
        print(f"   State dimension: {n_features}")
        print(f"   Kalman Filter: Active")
        print(f"   Correlation Learning: Active")
        print(f"   Bayesian Update: Active")
        print(f"   Gradient Optimization: Active")
    
    # ========================================================================
    # KALMAN FILTER UPDATE
    # ========================================================================
    
    def kalman_update(self, measurement, observation_matrix):
        """
        Kalman Filter ile state güncelleme
        
        Prediction:
            x̂_{t|t-1} = A × x̂_{t-1|t-1}
            P_{t|t-1} = A × P_{t-1|t-1} × A^T + Q
        
        Update:
            K_t = P_{t|t-1} × H^T × (H × P_{t|t-1} × H^T + R)^{-1}
            x̂_{t|t} = x̂_{t|t-1} + K_t × (y_t - H × x̂_{t|t-1})
            P_{t|t} = (I - K_t × H) × P_{t|t-1}
        """
        
        # Prediction step
        A = np.eye(self.n_features)  # State transition (identity)
        x_pred = A @ self.state
        P_pred = A @ self.P @ A.T + self.Q
        
        # Update step
        H = observation_matrix.reshape(1, -1)  # Measurement matrix
        
        # Kalman gain
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T / S
        
        # State update
        innovation = measurement - H @ x_pred
        self.state = x_pred + K.flatten() * innovation
        
        # Covariance update
        self.P = (np.eye(self.n_features) - K @ H) @ P_pred
        
        return self.state
    
    # ========================================================================
    # CORRELATION MATRIX LEARNING
    # ========================================================================
    
    def update_correlation_matrix(self, features, error):
        """
        Exponentially Weighted Correlation Update
        
        C_{t+1} = λ × C_t + (1-λ) × (x × x^T)
        
        λ: Decay factor (0.95)
        x: Feature vector
        error: Prediction error
        """
        
        λ = 0.95  # Decay factor
        
        # Feature outer product (x × x^T)
        x = features.reshape(-1, 1)
        x_outer = x @ x.T
        
        # Update correlation
        self.correlation_matrix = λ * self.correlation_matrix + (1-λ) * x_outer
        
        # Normalize to correlation (not covariance)
        D = np.sqrt(np.diag(self.correlation_matrix))
        D_inv = 1.0 / (D + 1e-8)
        self.correlation_matrix = np.outer(D_inv, D_inv) * self.correlation_matrix
        
        # Update error correlation
        # error_correlation[i, outcome] = feature i'nin outcome hatası ile korelasyonu
        outcome_idx = {'home_win': 0, 'draw': 1, 'away_win': 2}
        if error['actual'] in outcome_idx:
            idx = outcome_idx[error['actual']]
            self.error_correlation[:, idx] = (
                λ * self.error_correlation[:, idx] + 
                (1-λ) * features * error['magnitude']
            )
    
    def find_correlated_errors(self, features, threshold=0.01):
        """
        Hangi feature kombinasyonları hata yapıyor?
        
        Korelasyon matrisi C'yi analiz et:
        C[i,j] > threshold → feature i ve j birlikte hata yapıyor
        """
        
        error_correlations = []
        
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                corr = self.correlation_matrix[i, j]
                
                if abs(corr) > threshold:
                    # Bu iki feature beraber hareket ediyor
                    # İkisi de yüksek/düşükse hata pattern'i var mı?
                    
                    feature_i_val = features[i]
                    feature_j_val = features[j]
                    
                    # Geçmiş hatalarda bu pattern var mı?
                    pattern_errors = self._check_pattern_in_history(i, j, feature_i_val, feature_j_val)
                    
                    if pattern_errors > 0:
                        error_correlations.append({
                            'feature_i': i,
                            'feature_j': j,
                            'correlation': corr,
                            'error_count': pattern_errors
                        })
        
        return error_correlations
    
    def _check_pattern_in_history(self, i, j, val_i, val_j):
        """Geçmişte bu pattern'de kaç kez hata yaptık?"""
        count = 0
        for h in self.history[-500:]:  # Son 500 maç
            if not h['correct']:
                hist_i = h['features'][i]
                hist_j = h['features'][j]
                
                # Benzer değerler mi?
                if abs(hist_i - val_i) < 0.3 and abs(hist_j - val_j) < 0.3:
                    count += 1
        
        return count
    
    # ========================================================================
    # BAYESIAN INFERENCE
    # ========================================================================
    
    def bayesian_update(self, prediction_proba, actual_outcome):
        """
        Bayesian posterior güncelleme (Dirichlet-Multinomial conjugate)
        
        Prior: α = [α_home, α_draw, α_away]
        Likelihood: Observed outcome
        Posterior: α_new = α_old + outcome_vector
        
        Expected probability:
        E[p_i] = α_i / Σα_j
        
        Variance (uncertainty):
        Var[p_i] = α_i(α_0 - α_i) / (α_0²(α_0 + 1))
        """
        
        outcome_map = {'home_win': 0, 'draw': 1, 'away_win': 2}
        
        if actual_outcome in outcome_map:
            idx = outcome_map[actual_outcome]
            
            # Update posterior (Dirichlet conjugate update)
            self.posterior_alpha[idx] += 1
            
            # Expected probabilities
            alpha_sum = self.posterior_alpha.sum()
            expected_prob = self.posterior_alpha / alpha_sum
            
            # Uncertainty (variance)
            variance = np.zeros(3)
            for i in range(3):
                variance[i] = (self.posterior_alpha[i] * (alpha_sum - self.posterior_alpha[i])) / \
                             (alpha_sum**2 * (alpha_sum + 1))
            
            return expected_prob, variance
        
        return None, None
    
    # ========================================================================
    # GRADIENT-BASED FEATURE WEIGHT OPTIMIZATION
    # ========================================================================
    
    def gradient_update(self, features, error_magnitude, predicted_proba, actual_outcome):
        """
        Feature ağırlıklarını gradient descent ile güncelle
        
        Loss: L = -log(p_actual)  (Cross-entropy)
        
        Gradient:
        ∂L/∂w_i = -x_i × (1{y=actual} - p_actual)
        
        Update (with momentum):
        v_t = β × v_{t-1} + η × ∇L
        w_{t+1} = w_t - v_t
        """
        
        outcome_map = {'home_win': 0, 'draw': 1, 'away_win': 2}
        
        if actual_outcome not in outcome_map:
            return
        
        actual_idx = outcome_map[actual_outcome]
        p_actual = predicted_proba[actual_idx]
        
        # Cross-entropy loss
        loss = -np.log(p_actual + 1e-10)
        
        # Gradient computation
        # ∂L/∂w_i ≈ -x_i × (1 - p_actual) if prediction was wrong
        # Simple approximation
        gradient = -features * (1.0 - p_actual) if p_actual < 0.5 else features * p_actual
        
        # Clip gradient (prevent explosion)
        gradient = np.clip(gradient, -1.0, 1.0)
        
        # Momentum update (Nesterov-style)
        self.momentum = self.beta_momentum * self.momentum + self.learning_rate * gradient
        
        # Weight update
        self.feature_weights -= self.momentum
        
        # Keep weights positive and bounded
        self.feature_weights = np.clip(self.feature_weights, 0.1, 2.0)
        
        # Normalize weights
        self.feature_weights = self.feature_weights / self.feature_weights.mean()
        
        return loss
    
    # ========================================================================
    # ADAPTIVE LEARNING RATE (Meta-Learning)
    # ========================================================================
    
    def adapt_learning_rate(self):
        """
        Meta-learning: Öğrenme hızını otomatik ayarla
        
        Recent accuracy improving → Increase learning rate
        Recent accuracy declining → Decrease learning rate
        
        AdaGrad-style adaptation:
        η_t = η_0 / √(Σg²_i + ε)
        """
        
        if len(self.history) < 20:
            return
        
        # Son 20 maçın accuracy trend'i
        recent = self.history[-20:]
        accuracies = [1.0 if h['correct'] else 0.0 for h in recent]
        
        # Linear regression trend
        x = np.arange(len(accuracies))
        slope, _, _, _, _ = stats.linregress(x, accuracies)
        
        # Trend pozitif → lr artır, negatif → lr azalt
        if slope > 0.01:  # İyileşiyor
            self.learning_rate *= 1.1
            self.meta_params['adaptation_speed'] *= 1.05
        elif slope < -0.01:  # Kötüleşiyor
            self.learning_rate *= 0.9
            self.meta_params['adaptation_speed'] *= 0.95
        
        # Bounds
        self.learning_rate = np.clip(self.learning_rate, 0.001, 0.1)
        
        self.meta_params['learning_rate_schedule'].append({
            'step': self.n_updates,
            'lr': self.learning_rate,
            'trend': slope
        })
    
    # ========================================================================
    # CORRELATION-BASED ERROR PREDICTION
    # ========================================================================
    
    def predict_error_probability(self, features):
        """
        Bu feature kombinasyonu için hata olasılığını tahmin et
        
        P(error | features) = sigmoid(w^T × φ(x))
        
        φ(x): Feature transformation (correlation-aware)
        """
        
        # Feature transformation: Include correlations
        φ = np.zeros(self.n_features * 2)
        φ[:self.n_features] = features
        
        # Add correlation-based features
        for i in range(self.n_features):
            corr_weighted = 0
            for j in range(self.n_features):
                corr_weighted += self.correlation_matrix[i, j] * features[j]
            φ[self.n_features + i] = corr_weighted
        
        # Sigmoid activation
        z = np.dot(self.feature_weights, features)
        error_prob = 1.0 / (1.0 + np.exp(-z))
        
        return error_prob
    
    # ========================================================================
    # FULL LEARNING CYCLE
    # ========================================================================
    
    def learn_from_match(self, features, predicted_proba, predicted_class, actual_class):
        """
        Bir maçtan öğren - TÜM SİSTEMLERİ GÜNCELLE
        
        Args:
            features: np.array (58,) - Maç özellikleri
            predicted_proba: np.array (3,) - [p_home, p_draw, p_away]
            predicted_class: str - 'home_win'/'draw'/'away_win'
            actual_class: str - Gerçek sonuç
        """
        
        # Doğru mu?
        correct = (predicted_class == actual_class)
        
        # Hata büyüklüğü
        outcome_map = {'home_win': 0, 'draw': 1, 'away_win': 2}
        actual_idx = outcome_map[actual_class]
        error_magnitude = 1.0 - predicted_proba[actual_idx]
        
        # Hata vektörü
        error_record = {
            'features': features.copy(),
            'predicted': predicted_class,
            'actual': actual_class,
            'predicted_proba': predicted_proba.copy(),
            'correct': correct,
            'magnitude': error_magnitude,
            'timestamp': datetime.now(),
            'step': self.n_updates
        }
        
        # 1. KALMAN FILTER UPDATE
        observation = features * (1.0 if correct else -1.0)  # Signed by correctness
        self.state = self.kalman_update(error_magnitude, features)
        
        # 2. CORRELATION MATRIX UPDATE
        self.update_correlation_matrix(features, error_record)
        
        # 3. BAYESIAN UPDATE
        expected_prob, uncertainty = self.bayesian_update(predicted_proba, actual_class)
        
        # 4. GRADIENT UPDATE
        loss = self.gradient_update(features, error_magnitude, predicted_proba, actual_class)
        
        # 5. META-LEARNING (Her 10 maçta bir)
        if self.n_updates % 10 == 0:
            self.adapt_learning_rate()
        
        # History'e kaydet
        self.history.append(error_record)
        self.n_updates += 1
        
        # 6. PATTERN EXTRACTION (Her 50 maçta bir)
        if self.n_updates % 50 == 0:
            self.extract_error_patterns()
        
        return {
            'correct': correct,
            'loss': loss,
            'state': self.state,
            'expected_prob': expected_prob,
            'uncertainty': uncertainty,
            'learning_rate': self.learning_rate
        }
    
    # ========================================================================
    # ERROR PATTERN EXTRACTION
    # ========================================================================
    
    def extract_error_patterns(self):
        """
        Hata pattern'lerini çıkar ve kaydet
        
        Multi-variate correlation analysis:
        1. Hangi feature kombinasyonları hata yapıyor?
        2. Hangi durumlar sistematik olarak yanlış?
        """
        
        if len(self.history) < 50:
            return
        
        errors = [h for h in self.history if not h['correct']]
        
        if len(errors) < 10:
            return
        
        # Feature matrix of errors
        X_errors = np.array([e['features'] for e in errors])
        
        # Principal Component Analysis - Dominant error directions
        # Covariance matrix
        mean_features = X_errors.mean(axis=0)
        X_centered = X_errors - mean_features
        Cov = (X_centered.T @ X_centered) / len(errors)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(Cov)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Top 5 error directions (principal components)
        top_k = min(5, len(eigenvalues))
        
        self.error_patterns['dominant_directions'] = []
        
        for k in range(top_k):
            direction = eigenvectors[:, k]
            variance_explained = eigenvalues[k] / eigenvalues.sum()
            
            # Hangi feature'lar bu direction'da baskın?
            dominant_features = np.argsort(np.abs(direction))[-5:][::-1]
            
            self.error_patterns['dominant_directions'].append({
                'direction': direction,
                'variance_explained': variance_explained,
                'dominant_features': dominant_features
            })
        
        print(f"\n🔍 Pattern Analysis (Step {self.n_updates}):")
        print(f"   {len(errors)} hata analiz edildi")
        print(f"   Top {top_k} error direction bulundu")
        for k, pattern in enumerate(self.error_patterns['dominant_directions']):
            print(f"   Direction {k+1}: {pattern['variance_explained']*100:.1f}% variance")
    
    # ========================================================================
    # PREDICTION WITH LEARNING
    # ========================================================================
    
    def adjust_prediction(self, features, base_proba):
        """
        Öğrenilmiş bilgileri kullanarak tahmini ayarla
        
        P_adjusted = softmax(log(P_base) + Σ corrections)
        
        Corrections:
        1. Kalman state correction
        2. Correlation-based correction
        3. Bayesian prior
        4. Feature weight adjustment
        """
        
        # 1. Feature weighting
        weighted_features = features * self.feature_weights
        weight_correction = weighted_features.sum() / features.sum()
        
        # 2. Kalman state influence
        state_correction = np.tanh(self.state @ features / self.n_features)
        
        # 3. Correlation-based correction
        # Benzer durumlarda ne kadar başarılıyız?
        similar_errors = self.find_correlated_errors(features, threshold=0.01)
        corr_correction = -len(similar_errors) * 0.05  # Her pattern -5% güven
        
        # 4. Bayesian prior
        alpha_sum = self.posterior_alpha.sum()
        bayesian_prob = self.posterior_alpha / alpha_sum
        
        # Combine corrections
        log_proba = np.log(base_proba + 1e-10)
        
        corrections = np.array([
            weight_correction * 0.3,
            state_correction * 0.2,
            corr_correction
        ])
        
        # Bayesian blend
        adjusted_log_proba = log_proba + corrections
        
        # Softmax to get probabilities
        adjusted_proba = np.exp(adjusted_log_proba)
        adjusted_proba = adjusted_proba / adjusted_proba.sum()
        
        # Blend with Bayesian prior (0.2 weight)
        final_proba = 0.8 * adjusted_proba + 0.2 * bayesian_prob
        
        return final_proba
    
    # ========================================================================
    # SAVE/LOAD
    # ========================================================================
    
    def save(self, filename='incremental_learning_state.joblib'):
        """Tüm öğrenme state'ini kaydet"""
        state_dict = {
            'state': self.state,
            'P': self.P,
            'correlation_matrix': self.correlation_matrix,
            'error_correlation': self.error_correlation,
            'posterior_alpha': self.posterior_alpha,
            'feature_weights': self.feature_weights,
            'momentum': self.momentum,
            'meta_params': self.meta_params,
            'history': self.history,
            'error_patterns': self.error_patterns,
            'n_updates': self.n_updates
        }
        
        joblib.dump(state_dict, filename)
        print(f"💾 Learning state saved: {filename}")
    
    def load(self, filename='incremental_learning_state.joblib'):
        """Öğrenme state'ini yükle"""
        try:
            state_dict = joblib.load(filename)
            
            self.state = state_dict['state']
            self.P = state_dict['P']
            self.correlation_matrix = state_dict['correlation_matrix']
            self.error_correlation = state_dict['error_correlation']
            self.posterior_alpha = state_dict['posterior_alpha']
            self.feature_weights = state_dict['feature_weights']
            self.momentum = state_dict['momentum']
            self.meta_params = state_dict['meta_params']
            self.history = state_dict['history']
            self.error_patterns = state_dict['error_patterns']
            self.n_updates = state_dict['n_updates']
            
            print(f"📂 Learning state loaded: {filename}")
            print(f"   {self.n_updates} updates, {len(self.history)} matches learned")
            
            return True
        except FileNotFoundError:
            print(f"ℹ️  No existing state found, starting fresh")
            return False
    
    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================
    
    def print_diagnostics(self):
        """Sistem durumunu göster"""
        
        print("\n" + "=" * 80)
        print("🧠 INCREMENTAL LEARNING SYSTEM DIAGNOSTICS")
        print("=" * 80)
        
        if len(self.history) == 0:
            print("No learning yet!")
            return
        
        # Accuracy trend
        recent = self.history[-100:] if len(self.history) >= 100 else self.history
        accuracy = sum(1 for h in recent if h['correct']) / len(recent)
        
        print(f"\n📊 Performance:")
        print(f"   Total matches learned: {len(self.history)}")
        print(f"   Recent accuracy (last {len(recent)}): {accuracy*100:.2f}%")
        
        # Learning rate
        print(f"\n📈 Learning Parameters:")
        print(f"   Current learning rate: {self.learning_rate:.6f}")
        print(f"   Adaptation speed: {self.meta_params['adaptation_speed']:.3f}")
        
        # Top weighted features
        top_indices = np.argsort(self.feature_weights)[-10:][::-1]
        print(f"\n🎯 Top 10 Weighted Features:")
        feature_names = ['home_strength', 'away_strength', 'home_defense', 'away_defense',
                        'home_xG', 'away_xG', 'xG_diff', 'home_form', 'away_form',
                        'h2h_home', 'h2h_away', 'h2h_draw', 'goal_ratio', 'xG_ratio',
                        'day', 'month', 'home_support', 'away_support'] + ['feature_' + str(i) for i in range(40)]
        
        for idx in top_indices:
            if idx < len(feature_names):
                print(f"   {feature_names[idx]:20s}: {self.feature_weights[idx]:.3f}")
        
        # Correlation insights
        print(f"\n🔗 Correlation Insights:")
        max_corr = 0
        max_pair = (0, 0)
        for i in range(min(20, self.n_features)):
            for j in range(i+1, min(20, self.n_features)):
                if abs(self.correlation_matrix[i, j]) > abs(max_corr):
                    max_corr = self.correlation_matrix[i, j]
                    max_pair = (i, j)
        
        if abs(max_corr) > 0.1:
            print(f"   Strongest correlation: Feature {max_pair[0]} ↔ {max_pair[1]} ({max_corr:.3f})")
        
        # Bayesian posterior
        print(f"\n📊 Bayesian Posterior (Long-term priors):")
        alpha_sum = self.posterior_alpha.sum()
        probs = self.posterior_alpha / alpha_sum
        print(f"   Home Win: {probs[0]*100:.2f}%")
        print(f"   Draw:     {probs[1]*100:.2f}%")
        print(f"   Away Win: {probs[2]*100:.2f}%")
        print(f"   (Based on {int(alpha_sum)-3} observed outcomes)")
        
        print("=" * 80)

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Advanced Incremental Learning System - TEST\n")
    
    learner = AdvancedIncrementalLearner(n_features=58)
    
    # Simulated learning
    print("Simulating 200 matches...\n")
    
    for i in range(200):
        # Random features
        features = np.random.randn(58) * 0.5 + 1.5
        features = np.clip(features, 0, 5)
        
        # Random prediction
        base_proba = np.random.dirichlet([2, 1, 2])  # [home, draw, away]
        predicted_class = ['home_win', 'draw', 'away_win'][np.argmax(base_proba)]
        
        # Random actual (biased by prediction)
        if np.random.rand() < base_proba.max():
            actual_class = predicted_class
        else:
            actual_class = np.random.choice(['home_win', 'draw', 'away_win'])
        
        # Learn!
        result = learner.learn_from_match(features, base_proba, predicted_class, actual_class)
        
        if (i + 1) % 50 == 0:
            print(f"Step {i+1}: Loss = {result['loss']:.4f}, LR = {result['learning_rate']:.6f}")
    
    # Diagnostics
    learner.print_diagnostics()
    
    # Save
    learner.save()
    
    print("\n✅ Test completed!")
    print("\nKullanım:")
    print("   learner = AdvancedIncrementalLearner()")
    print("   learner.load()  # Önceki öğrenmeyi yükle")
    print("   adjusted_proba = learner.adjust_prediction(features, base_proba)")
    print("   learner.learn_from_match(...)  # Gerçek sonuçtan öğren")
    print("   learner.save()  # Öğrenmeyi kaydet")





