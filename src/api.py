from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import numpy as np
import os

# Inicialização da API
app = FastAPI(
    title="AeroMotion Dynamics - Predictive Maintenance API",
    description="Microsserviço RESTful para inferência de falhas mecânicas em bancadas de teste.",
    version="1.0.0"
)

# Caminho dos artefactos de IA serializados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(BASE_DIR, "..", "ficheiro_IA", "modelo_random_forest.joblib")
CAMINHO_SCALER = os.path.join(BASE_DIR, "..", "ficheiro_IA", "scaler_robust.joblib")

# Carregamento do modelo treinado
try:
    modelo = joblib.load(CAMINHO_MODELO)
    scaler = joblib.load(CAMINHO_SCALER)
except Exception as e:
    modelo = None
    scaler = None

MAPA_FALHAS = {
    0: "Normal",
    1: "Desgaste de Rolamento",
    2: "Desbalanceamento de Rotor",
    3: "Cavitação / Superaquecimento"
}

# Schemas de validação Pydantic
class TelemetriaEntrada(BaseModel):
    horas_operacao_acumuladas: float = Field(..., example=1200.0)
    temperatura_mancal_c: float = Field(..., example=62.5)
    vibracao_rms_mms: float = Field(..., example=2.5)
    pressao_oleo_bar: float = Field(..., example=4.3)
    rotacao_rpm: float = Field(..., example=3000.0)
    emissao_acustica_db: float = Field(..., example=52.0)
    corrente_eletrica_amp: float = Field(..., example=28.0)
    temp_rolling_mean_1h: float = Field(..., example=62.0)
    temp_rolling_std_1h: float = Field(..., example=1.2)
    temp_rolling_mean_6h: float = Field(..., example=61.8)
    temp_rolling_std_6h: float = Field(..., example=2.0)
    vib_rolling_mean_1h: float = Field(..., example=2.4)
    vib_rolling_std_1h: float = Field(..., example=0.4)
    vib_rolling_mean_6h: float = Field(..., example=2.5)
    vib_rolling_std_6h: float = Field(..., example=0.6)
    severidade_dinamica: float = Field(..., example=0.00083)

class PrevisaoSaida(BaseModel):
    codigo_falha: int
    diagnostico: str
    probabilidades: List[float]

# 1. Endpoint GET /health
@app.get("/health", tags=["Monitoramento"])
def health_check():
    return {
        "status": "online",
        "versao_api": "1.0.0",
        "modelo_carregado": modelo is not None,
        "algoritmo": "RandomForestClassifier (Multiclasse Balanced)"
    }

# 2. Endpoint POST /predict (Inferência em Tempo Real)
@app.post("/predict", response_model=PrevisaoSaida, tags=["Inferência"])
def predizer_uma_leitura(dados: TelemetriaEntrada):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo preditivo não carregado no servidor.")
        
    df_input = pd.DataFrame([dados.model_dump()])
    pred = int(modelo.predict(df_input)[0])
    prob = modelo.predict_proba(df_input)[0].tolist()
    
    return PrevisaoSaida(
        codigo_falha=pred,
        diagnostico=MAPA_FALHAS[pred],
        probabilidades=[round(p, 4) for p in prob]
    )

# 3. Endpoint POST /predict-batch (Inferência em Lote)
@app.post("/predict-batch", response_model=List[PrevisaoSaida], tags=["Inferência"])
def predizer_lote(dados_lote: List[TelemetriaEntrada]):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo preditivo não carregado no servidor.")
        
    registos = [item.model_dump() for item in dados_lote]
    df_input = pd.DataFrame(registos)
    
    preds = modelo.predict(df_input)
    probs = modelo.predict_proba(df_input)
    
    respostas = []
    for pred, prob in zip(preds, probs):
        codigo = int(pred)
        respostas.append(PrevisaoSaida(
            codigo_falha=codigo,
            diagnostico=MAPA_FALHAS[codigo],
            probabilidades=[round(p, 4) for p in prob.tolist()]
        ))
        
    return respostas