
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Create dummy model
model = DummyClassifier(strategy='stratified')
X = np.random.rand(10, 60)
y = np.random.choice(['home_win', 'draw', 'away_win'], 10)
model.fit(X, y)
joblib.dump(model, 'footbal_brain_core - anti gravity/football_prediction-main/football_prediction_ensemble.joblib')

# Create dummy label encoder
le = LabelEncoder()
le.fit(['home_win', 'draw', 'away_win'])
joblib.dump(le, 'footbal_brain_core - anti gravity/football_prediction-main/label_encoder.joblib')
