# train_model.py
from sklearn.ensemble import RandomForestClassifier
import joblib

# Exemple simple d'entraînement
# Features : [âge, genre] —> genre: 0 = femme, 1 = homme
X = [[20, 0], [25, 1], [30, 0], [35, 1], [40, 1]]
y = ['pop', 'rock', 'jazz', 'classical', 'electro']  # labels musicaux

model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarder le modèle
joblib.dump(model, 'music_recommender.joblib')
print("✅ Modèle entraîné et sauvegardé.")
