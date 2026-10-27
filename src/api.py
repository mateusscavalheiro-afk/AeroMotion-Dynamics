from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
import json
import sys

# Importação das rotinas de visão computacional
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from computer_vision import inspecionar_motor_completo

# Inicialização da API
app = FastAPI(
    title="AeroMotion Dynamics - Predictive Maintenance API",
    description="Microsserviço RESTful para inferência de falhas mecânicas e inspeção visual multimodal.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho dos artefactos de IA serializados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(BASE_DIR, "..", "ficheiro_IA", "modelo_random_forest.joblib")
CAMINHO_SCALER = os.path.join(BASE_DIR, "..", "ficheiro_IA", "scaler_robust.joblib")

# Carregamento do modelo treinado
try:
    modelo = joblib.load(CAMINHO_MODELO)
    scaler = joblib.load(CAMINHO_SCALER)
except Exception:
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

# 2. Endpoint POST /predict (Inferência Tabular via RandomForest)
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

# 3. Endpoint POST /predict-batch (Inferência Tabular em Lote)
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

# 4. Endpoint POST /inspecionar-multimodal (Visão Computacional + Gráfico Ponderado)
@app.post("/inspecionar-multimodal", tags=["Inspeção Multimodal"])
async def inspecionar_multimodal_endpoint(
    imagem_teste: UploadFile = File(..., description="Imagem do motor sob teste"),
    imagem_gabarito: UploadFile = File(..., description="Imagem gabarito nominal (sem defeitos)"),
    telemetria_json: Optional[str] = Form(None, description="JSON serializado com os sensores IoT"),
    sensibilidade_corte: int = Form(40, description="Limiar de corte Canny"),
    tolerancia_max: float = Form(3.0, description="Tolerância máxima de dano (%) por ROI"),
    min_area_cluster: int = Form(30, description="Área mínima em píxeis para ignorar ruído isolado")
):
    try:
        buffer_teste = await imagem_teste.read()
        buffer_gabarito = await imagem_gabarito.read()
        
        telemetria_dict = json.loads(telemetria_json) if telemetria_json else {}
        
        # Executa a inspeção e a distribuição balanceada
        resultado = inspecionar_motor_completo(
            caminho_teste=buffer_teste,
            caminho_gabarito=buffer_gabarito,
            sensibilidade=sensibilidade_corte,
            tolerancia_max=tolerancia_max,
            min_area_cluster=min_area_cluster,
            telemetria_iot=telemetria_dict
        )
        
        diag = resultado["diagnostico_multimodal"]
        
        return {
            "status_geral": resultado["status_geral"],
            "risco_composto_pct": diag["risco_composto_pct"],
            "classe_diagnosticada": diag["classe_diagnosticada"],
            # Array estritamente balanceado consumido pela série de dados do gráfico:
            "probabilidades_grafico": diag["probabilidades_grafico"], 
            "probabilidade_por_estado": diag["probabilidade_por_estado"],
            "detalhamento_rois": resultado["detalhamento_por_componente"],
            "telemetria_processada": telemetria_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha na inspeção ótico-sensorial: {str(e)}")