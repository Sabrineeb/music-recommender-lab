# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

model_path = 'music_recommender.joblib'

# Charger le modèle
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Modèle introuvable à : {model_path}")

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"❌ Impossible de charger le modèle : {e}")

# Schéma de données attendu
class UserInput(BaseModel):
    age: int
    gender: int  # 0 = femme, 1 = homme

@app.post("/predict")
def predict(user_input: UserInput):
    try:
        prediction = model.predict([[user_input.age, user_input.gender]])
        return {"genre": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
