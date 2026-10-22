# AeroMotion Dynamics S.A. — Divisão de Turbinas & Propulsão de Alta Precisão
## Sistema Multimodal de Manutenção Preditiva e Inspeção Óptica 4.0

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.8%2B-5C3EE8.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Sumário Executivo

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Arquitetura da Solução & Fluxo de Dados](#2-arquitetura-da-solução--fluxo-de-dados)
3. [Estrutura do Repositório](#3-estrutura-do-repositório)
4. [Etapa 1: Engenharia de Dados & Tratamento de Séries Temporais](#4-etapa-1-engenharia-de-dados--tratamento-de-séries-temporais)
5. [Etapa 2: Inteligência Preditiva, Modelagem Multiclasse & MLOps](#5-etapa-2-inteligência-preditiva-modelagem-multiclasse--mlops)
6. [Etapa 3: Visão Computacional Multimodal & Fusão Sensorial](#6-etapa-3-visão-computacional-multimodal--fusão-sensorial)
7. [Etapa 4: Engenharia de Software, Microsserviço RESTful & Docker](#7-etapa-4-engenharia-de-software-microsserviço-restful--docker)
8. [Etapa 5: Interface SCADA Industrial & Simulação Híbrida](#8-etapa-5-interface-scada-industrial--simulação-híbrida)
9. [Governança, Ética e Conformidade Regulatória](#9-governança-ética-e-conformidade-regulatória)
10. [Instalação, Configuração e Execução](#10-instalação-configuração-e-execução)
11. [Equipe de Desenvolvimento](#11-equipe-de-desenvolvimento)

---

## 1. Visão Geral do Projeto

A **AeroMotion Dynamics S.A.** opera quatro bancadas de testes automatizadas (**TURB-101 a TURB-104**) dedicadas à homologação de conjuntos turbomáquina de alta rotação. Cada ativo é monitorado por telemetria IoT em intervalos regulares de 10 minutos, gerando um fluxo denso de sinais contínuos.

```
                           +---------------------------+
                           |  Bancada SCADA Industrial |
                           |    (TURB-101 a 104)       |
                           +-------------+-------------+
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
                 v                                               v
     [ Sensores Físicos IoT ]                       [ Câmera Óptica / Termográfica ]
     - Temperatura de Mancal                        - Enquadramento Nominal
     - Vibração RMS Global                          - Setorização em 3 ROIs
     - Pressão de Óleo                              - Termografia Infravermelha HSV
     - Rotação e Corrente                           - Detecção de Trincas/Fissuras
     - Emissão Acústica                                          |
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Motor de Fusão Sensorial    |
                         |        (FastAPI / AI)         |
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         |  Diagnóstico Multimodal 4.0   |
                         |  - Risco Composto Dinâmico    |
                         |  - Classificação de Falha 1-3 |
                         |  - Laudo Visual & Dashboard   |
                         +-------------------------------+
```

### Desafios Críticos Abordados
* **Interconexão de Falhas Mecânicas:** A degradação raramente é isolada. O desalinhamento de rotor frequentemente induz fadiga prematura em mancais e fenômenos de cavitação nas pás do impulsor.
* **Ambiente Hostil e Ruídos Eletromagnéticos (EMI):** Transientes de rede e campos eletromagnéticos geram leituras de surto (como temperaturas espúrias de $999.9^\circ\text{C}$ e vibrações negativas de $-99.0\text{ mm/s}$) que exigem filtros de rejeição física sem ruptura de continuidade temporal.
* **Auditoria Óptica em Linha de Montagem:** Validação geométrica adaptativa e análise térmica em infravermelho para garantir a integridade estrutural e prevenir liberação de lotes anômalos.

---

## 2. Arquitetura da Solução & Fluxo de Dados

A arquitetura do sistema adota o paradigma *Edge Computing*, desacoplando o pipeline em camadas modulares de processamento, inferência e renderização:

```
[dados_sensores_aeromotion.csv]
            |
            v
  [src/preprocessing.py] ---------> Limpeza Temporal, Rejeição EMI, Rolling Features
            |
            v
 [data/processed/..._tratado.csv]
            |
            +-----------------------------------+
            |                                   |
            v                                   v
[01_etapa1_dados.ipynb]             [02_etapa2_modelagem.ipynb]
(Inspeção & Análise Exploratória)   (Random Forest, GradBoost, Isolation Forest)
                                                |
                                                v
                                    [ficheiro_IA/*.joblib]
                                                |
                                                v
                                         [src/api.py] <---> [src/computer_vision.py]
                                                ^
                                                | RESTful (HTTP/JSON)
                                                v
                                        [index.html]
                                 (Dashboard SCADA Industrial)
```

---

## 3. Estrutura do Repositório

```
AeroMotion-Dynamics/
├── data/
│   ├── processed/
│   │   └── dados_sensores_aeromotion_tratado.csv  # Dataset saneado com features móveis
│   └── raw/
│       └── dados_sensores_aeromotion.csv          # Telemetria bruta (12.000 registros)
├── ficheiro_IA/
│   ├── modelo_gradient_boosting.joblib            # Classificador Gradient Boosting serializado
│   ├── modelo_isolation_forest.joblib             # Detector não-supervisionado de anomalias
│   ├── modelo_random_forest.joblib                # Modelo Random Forest balanceado
│   └── scaler_robust.joblib                       # Normalizador robusto a outliers
├── notebooks/
│   ├── 01_etapa1_dados.ipynb                      # Pipeline de EDA, saneamento e correlação
│   └── 02_etapa2_modelagem.ipynb                  # Treinamento, validação e matrizes de confusão
├── src/
│   ├── __init__.py                                # Inicializador de pacote Python
│   ├── api.py                                     # Microsserviço RESTful em FastAPI
│   ├── computer_vision.py                         # Módulo de visão computacional e fusão sensorial
│   └── preprocessing.py                           # Rotinas automatizadas de ETL industrial
├── .gitignore                                     # Exclusões de ambiente virtual e cache
├── Dockerfile                                     # Especificação da imagem Docker minimalista
├── docker-compose.yml                             # Orquestração do contêiner Edge
├── index.html                                     # Dashboard SCADA multimodal (Tailwind + Chart.js)
├── LICENSE                                        # Licença MIT
├── README.md                                      # Documentação técnica mestre do projeto
└── requirements.txt                               # Dependências rastreadas do ambiente
```

---

## 4. Etapa 1: Engenharia de Dados & Tratamento de Séries Temporais

### 4.1. Dicionário de Variáveis de Telemetria
O dataset bruto compreende $12.000$ amostras temporais coletadas das quatro bancadas industriais:

| Variável | Tipo | Unidade | Faixa Normal | Descrição Física |
| :--- | :---: | :---: | :---: | :--- |
| `timestamp` | Datetime | AAAA-MM-DD HH:MM | Contínuo | Intervalos regulares de amostragem ($10\text{ min}$) |
| `id_equipamento` | Categórico | String | TURB-101 a 104 | Identificador físico da bancada de ensaio |
| `modo_operacao` | Categórico | String | Carga Alta/Média/Cruzeiro | Regime de rotação e carga mecânica imposta |
| `horas_operacao_acumuladas` | Float | Horas (h) | $100.0$ a $4500.0\text{ h}$ | Tempo total acumulado de fadiga operacional |
| `temperatura_mancal_c` | Float | $^\circ\text{C}$ | $45.0$ a $85.0^\circ\text{C}$ | Termometria do rolamento (Ruído em $999.9^\circ\text{C}$) |
| `vibracao_rms_mms` | Float | $\text{mm/s}$ | $1.0$ a $4.5\text{ mm/s}$ | Velocidade RMS de vibração (Ruído em $-99.0\text{ mm/s}$) |
| `pressao_oleo_bar` | Float | $\text{bar}$ | $3.5$ a $5.2\text{ bar}$ | Pressão da linha hidráulica lubrificante |
| `rotacao_rpm` | Float | $\text{RPM}$ | $2300$ a $3600\text{ RPM}$ | Frequência de rotação angular do eixo |
| `emissao_acustica_db` | Float | $\text{dB}$ | $40$ a $68\text{ dB}$ | Fricção de alta frequência e contato metal-metal |
| `corrente_eletrica_amp` | Float | $\text{A}$ | $18$ a $42\text{ A}$ | Demanda de corrente nas fases do estator |
| `tipo_falha` (Target) | Inteiro | Código | $0, 1, 2, 3$ | $0$: Normal; $1$: Rolamento; $2$: Rotor; $3$: Cavitação |

---

### 4.2. Tratamento de Ruídos Eletromagnéticos (EMI) & Interpolação Contínua
Sensores industriais sofrem interferência transiente de inversores de frequência e comutações de alta potência. Os artefatos foram tratados segundo a física do processo contínuo:
* **Temperatura do Mancal:** Amostras com $\text{Temp} \ge 900.0^\circ\text{C}$ (decorrentes de saturação do sensor) foram convertidas para `NaN`.
* **Vibração RMS:** Amostras espúrias com $\text{Vib} < 0.0\text{ mm/s}$ (impossíveis na métrica quadrática média) foram convertidas para `NaN`.
* **Interpolação Temporal por Equipamento:** As lacunas foram recompostas aplicando interpolação linear agrupada por `id_equipamento`:

$$\hat{y}(t) = y(t_0) + \frac{y(t_1) - y(t_0)}{t_1 - t_0} \cdot (t - t_0)$$

Seguida por *backfill* e *forward-fill* nas extremidades, garantindo ausência total de nulos sem comprometer as assinaturas dinâmicas.

---

### 4.3. Engenharia de Variáveis Temporais (Feature Engineering)
Para capturar a inércia térmica e a evolução de microtrincas, foram derivadas variáveis móveis em duas janelas temporais ($1\text{ hora} = 6\text{ passos}$; $6\text{ horas} = 36\text{ passos}$):

1. **Médias Móveis:**

$$\mu_{1h}(t) = \frac{1}{6} \sum_{k=0}^{5} x(t - k), \quad \mu_{6h}(t) = \frac{1}{36} \sum_{k=0}^{35} x(t - k)$$

2. **Desvios Padrão Móveis:**

$$\sigma_{1h}(t) = \sqrt{\frac{1}{5} \sum_{k=0}^{5} \big(x(t - k) - \mu_{1h}(t)\big)^2}, \quad \sigma_{6h}(t) = \sqrt{\frac{1}{35} \sum_{k=0}^{35} \big(x(t - k) - \mu_{6h}(t)\big)^2}$$

3. **Severidade Dinâmica Derivada:**
Razão física entre o deslocamento vibratório RMS e a frequência angular de rotação do conjunto:

$$\text{Severidade Dinâmica} = \frac{\text{Vibração RMS } (\text{mm/s})}{\text{Rotação } (\text{RPM})}$$

---

### 4.4. Achados da Análise Exploratória Multivariada (EDA)
* **Correlação da Severidade Dinâmica:** Apresentou correlação linear de $+0.973$ com a vibração bruta, destacando com precisão eventos de desbalanceamento de rotor.
* **Assinatura Acústica de Mancais:** A emissão acústica correlaciona-se com o desgaste de rolamento ($r = +0.594$ com vibração), refletindo o colapso do filme de óleo lubrificante.
* **Degradação Térmico-Hidráulica:** Eventos de cavitação combinam queda brusca de pressão ($< 2.6\text{ bar}$) com picos de temperatura no mancal ($> 88^\circ\text{C}$), com correlação inversa severa de $-0.482$ entre pressão e falha de cavitação.

---

## 5. Etapa 2: Inteligência Preditiva, Modelagem Multiclasse & MLOps

### 5.1. Partição Temporal e Padronização Robusta
Para evitar vazamento de dados (*data leakage*), o dataset foi particionado estritamente em ordem cronológica:
* **Treino (70%):** Primeiras $8.400$ amostras temporais.
* **Teste (30%):** Últimas $3.600$ amostras temporais.

A escala das $16$ features foi calibrada com o **RobustScaler**, que utiliza a mediana e o intervalo interquartil ($\text{IQR} = Q_3 - Q_1$), tornando o vetor de entrada imune à influência de picos pontuais:

$$x_{scaled} = \frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}$$

---

### 5.2. Comparativo de Modelos Supervisionados
Considerando que falhas mecânicas representam apenas $\sim 12\%$ das ocorrências (severo desbalanceamento de classes), os modelos foram ajustados para penalizar erros de falsos negativos:

| Métrica Multiclasse | Random Forest (Balanced) | Gradient Boosting | Meta da Rubrica |
| :--- | :---: | :---: | :---: |
| **Acurácia Global** | **99.47%** | **99.61%** | $> 95.0\%$ |
| **Precisão (Macro)** | $98.56\%$ | $99.46\%$ | $> 90.0\%$ |
| **Recall (Macro)** | **97.67%** | **97.90%** | **$> 90.0\%$** |
| **F1-Score (Macro)** | $98.10\%$ | $98.67\%$ | $> 90.0\%$ |

Ambos os modelos superaram a meta acadêmica de $90\%$ de Recall em todas as classes isoladas:
* **Classe 1 (Desgaste de Rolamento):** Recall de $99.1\%$
* **Classe 2 (Desbalanceamento de Rotor):** Recall de $95.8\%$
* **Classe 3 (Cavitação / Superaquecimento):** Recall de $98.1\%$

---

### 5.3. Explicabilidade Física (Feature Importance)
O critério de impureza de Gini na Random Forest ratificou os fenômenos termodinâmicos e dinâmicos:
1. `emissao_acustica_db`: $17.94\%$ de importância relativa (primária para rolamentos).
2. `vibracao_rms_mms`: $15.17\%$ (primária para excentricidade do rotor).
3. `severidade_dinamica`: $14.70\%$ (indicador composto de energia de vibração).
4. `temperatura_mancal_c`: $14.68\%$ (indicador de atrito e superaquecimento).
5. `pressao_oleo_bar`: $14.64\%$ (indicador de integridade da cavitação hidráulica).

---

### 5.4. Detecção Não-Supervisionada com Isolation Forest
Para detectar anomalias mecânicas não mapeadas previamente no catálogo de falhas, foi calibrado um modelo **Isolation Forest** treinado exclusivamente sobre os dados nominais (`tipo_falha == 0`, contaminação estimada em $1\%$).
* O modelo isola amostras anômalas computando o comprimento médio de árvore $h(x)$:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Onde $s(x, n) \to 1$ indica anomalia severa (caminho curto de partição).

---

### 5.5. Plano de MLOps & Monitoramento de Desvios (Janela de 12 Meses)
Para mitigar a perda gradual de acurácia gerada pelo envelhecimento físico das bancadas:
* **Detecção de Data Drift (Covariate Shift):** Teste estatístico de Kolmogorov-Smirnov (KS-test) bicaudal executado semanalmente em cada feature contínua contra a distribuição basal de treino:

$$D_{KS} = \sup_x |F_{treino}(x) - F_{producao}(x)|, \quad \text{Alerta se } p\text{-value} < 0.01$$

* **Detecção de Concept Drift:** Cálculo mensal do *Population Stability Index* (PSI). Valores de $\text{PSI} > 0.2$ disparam pipeline de retreinamento automático de pesos.

---

## 6. Etapa 3: Visão Computacional Multimodal & Fusão Sensorial

A inspeção óptica analisa a geometria externa e interna do motor/bomba a partir de um gabarito de referência limpo em fundo branco, setorizando o conjunto em três Regiões de Interesse (ROIs):

```
+-------------------------------------------------------------------------+
|                  CORTE LONGITUDINAL DO CONJUNTO MOTOR/BOMBA             |
|                                                                         |
|  [ ROI 1: MANCAL/ROLAMENTO ]   [ ROI 2: ROTOR/EIXO ]    [ ROI 3: PÁS ]  |
|      (Coords: X=15 a 95px)      (Coords: X=95 a 205)   (X=205 a 285)    |
|   - Fissuras em pistas/esferas   - Trincas axiais       - Cavitação     |
|   - Medição de Circularidade     - Excentricidade       - Pitting       |
+-------------------------------------------------------------------------+
```

---

### 6.1. Filtragem Morfológica e Análise de Componentes Conectados
A diferença direta de matrizes entre a peça inspecionada e o gabarito nominal tende a capturar ruídos finos de textura e imperfeições estocásticas geradas por IA. Para reter unicamente massas reais de defeito, implementou-se o algoritmo duplo em `computer_vision.py`:

1. **Abertura Morfológica (`MORPH_OPEN`):** Executa erosão seguida de dilatação com elemento estruturante retangular $3 \times 3$, eliminando pontilhados soltos e microarestas espúrias de $1\text{ a }2\text{ px}$:

$$A \circ B = (A \ominus B) \oplus B$$

2. **Componentes Conectados por Área Mínima (`connectedComponentsWithStats`):** Agrupa os pixels remanescentes com $8\text{-conectividade}$. Clusters com área total inferior ao limiar `min_area_cluster` (calibrado em $30\text{ px}$) são sumariamente expurgados:

```python
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    mascara_morfologica, connectivity=8
)
mascara_limpa = np.zeros_like(mascara_morfologica)
for label in range(1, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]
    if area >= min_area_cluster:
        mascara_limpa[labels == label] = 255
```

---

### 6.2. Inspeção Térmica em Espaço de Cor HSV
Para delimitar zonas críticas de superaquecimento (*hotspots*), as imagens infravermelhas são convertidas para o espaço **HSV** (*Hue-Saturation-Value*), aplicando limiarização bimodal na faixa espectral do vermelho crítico:

$$\text{Mascara}_1 = \text{inRange}(H \in [0, 10], S \in [120, 255], V \in [150, 255])$$

$$\text{Mascara}_2 = \text{inRange}(H \in [170, 180], S \in [120, 255], V \in [150, 255])$$

$$\text{Hotspots} = \text{Mascara}_1 \lor \text{Mascara}_2$$

Se $\sum \text{Pixels}_{hotspot} > 500\text{ px}$, emite-se laudo de reprovação térmica emergencial.

---

### 6.3. Modelo Matemático de Fusão Sensorial e Proporcionalidade das Barras
Para eliminar empates artificiais no gráfico de barras (onde classes distintas ficavam travadas em $33.2\%$ ou $2.0\%$), a função `distribuir_probabilidades_multimodal` opera segundo um rateio estritamente ponderado pelos danos medidos nas ROIs:

1. **Risco Composto Global ($R_c$):**
Se qualquer ROI ultrapassar o limiar de tolerância $\tau$ (calibrado em $2.5\%$), o risco escala para o patamar crítico ($85\%$ a $99.6\%$):

$$R_c = 85.0 + 14.6 \cdot \min\left(\frac{\max(D_{rol}, D_{rot}, D_{pas})}{2.5 \cdot \tau}, 1.0\right)$$

A probabilidade nominal passa a ser:

$$P(\text{Normal}) = \max(0.4, 100.0 - R_c)$$

2. **Rateio Proporcional entre os Modos de Falha:**
O saldo residual de risco ($100.0 - P(\text{Normal})$) é distribuído pela proporção matemática direta de cada componente:

$$P(i) = \big(100.0 - P(\text{Normal})\big) \cdot \frac{D_i}{\sum_{k \in \{rol, rot, pas\}} D_k}$$

Desta forma, na presença de danos severos simultâneos (ex.: Rotor com $20.24\%$, Rolamento com $13.60\%$ e Cavitação com $10.97\%$), o gráfico reflete a assimetria real do teste: **Rotor em $\sim 45\%$**, **Rolamento em $\sim 30\%$** e **Cavitação em $\sim 24\%$**.

---

## 7. Etapa 4: Engenharia de Software, Microsserviço RESTful & Docker

### 7.1. Especificação dos Endpoints da API (`src/api.py`)

#### `GET /health`
* **Finalidade:** Health check do microsserviço, monitoramento de conectividade e versão dos pesos de IA.
* **Response `200 OK`:**
```json
{
  "status": "online",
  "versao_api": "1.0.0",
  "modelo_carregado": true,
  "algoritmo": "RandomForestClassifier (Multiclasse Balanced)"
}
```

#### `POST /predict`
* **Finalidade:** Inferência preditiva tabular unitária via telemetria contínua.
* **Payload Request (`application/json`):**
```json
{
  "horas_operacao_acumuladas": 2450.0,
  "temperatura_mancal_c": 63.5,
  "vibracao_rms_mms": 2.8,
  "pressao_oleo_bar": 4.2,
  "rotacao_rpm": 3000.0,
  "emissao_acustica_db": 53.0,
  "corrente_eletrica_amp": 28.5,
  "temp_rolling_mean_1h": 63.1,
  "temp_rolling_std_1h": 1.1,
  "temp_rolling_mean_6h": 62.6,
  "temp_rolling_std_6h": 2.0,
  "vib_rolling_mean_1h": 2.6,
  "vib_rolling_std_1h": 0.3,
  "vib_rolling_mean_6h": 2.5,
  "vib_rolling_std_6h": 0.5,
  "severidade_dinamica": 0.000933
}
```
* **Response `200 OK`:**
```json
{
  "codigo_falha": 0,
  "diagnostico": "Normal",
  "probabilidades": [0.982, 0.008, 0.006, 0.004]
}
```

#### `POST /predict-batch`
* **Finalidade:** Inferência preditiva tabular em lote para histórico de bancadas.

#### `POST /inspecionar-multimodal`
* **Finalidade:** Ingestão síncrona de imagem de câmera, imagem gabarito e parâmetros de telemetria IoT com fusão balanceada.
* **Content-Type:** `multipart/form-data`
* **Parâmetros:** `imagem_teste` (file), `imagem_gabarito` (file), `telemetria_json` (string opcional), `sensibilidade_corte` (int, default 40), `tolerancia_max` (float, default 2.5), `min_area_cluster` (int, default 30).
* **Response `200 OK`:**
```json
{
  "status_geral": "REPROVADO",
  "risco_composto_pct": 72.4,
  "classe_diagnosticada": "2: Desbalanceamento de Rotor",
  "probabilidades_grafico": [27.6, 2.1, 43.1, 27.2],
  "probabilidade_por_estado": {
    "0: Normal": 27.6,
    "1: Rolamento": 2.1,
    "2: Rotor": 43.1,
    "3: Cavitação": 27.2
  },
  "detalhamento_rois": {
    "rolamento": {
      "componente": "Rolamentos / Mancal",
      "dano_estrutural_pct": 1.73,
      "tolerancia_limite_pct": 2.5,
      "status": "APROVADO"
    },
    "rotor": {
      "componente": "Rotor / Eixo Central",
      "dano_estrutural_pct": 4.12,
      "tolerancia_limite_pct": 2.5,
      "status": "REPROVADO"
    },
    "pas": {
      "componente": "Pás da Turbina / Impulsor",
      "dano_estrutural_pct": 2.62,
      "tolerancia_limite_pct": 2.5,
      "status": "REPROVADO"
    }
  }
}
```

---

### 7.2. Conteinerização para Edge Computing (`Dockerfile`)
A conteinerização utiliza imagem oficial base `python:3.10-slim`, instalando as bibliotecas do sistema `libglib2.0-0` necessárias para o funcionamento do OpenCV em modo *headless*:

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY ficheiro_IA/ ./ficheiro_IA/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 8. Interface SCADA Industrial & Simulação Híbrida (`index.html`)

O frontend é uma SPA (*Single Page Application*) responsiva inspirada em consoles de salas de controle SCADA de aviação e petroquímica, construída com **Tailwind CSS**, **Chart.js** e ícones **FontAwesome**:

### Recursos do Dashboard
1. **Modo Híbrido com Fallback Air-Gapped:** Tenta conectar via `fetch` assíncrono ao endpoint REST `http://127.0.0.1:8000/predict`. Se a rede OT estiver isolada ou o backend desligado, o frontend comuta automaticamente para o motor embutido `simularRandomForestOffline()`, registrando *"Modo Offline Air-Gapped"* e latência instantânea.
2. **Quatro Viewports Canvas:**
   * **1. Gabarito Nominal:** Renderiza o padrão dourado com bounding boxes nas 3 ROIs.
   * **2. Peça em Teste:** Exibe a imagem de câmera ou esquema da turbina sob auditoria.
   * **3. Granulado de Dano:** Máscara de pixels anômalos isolados pelo algoritmo Canny.
   * **4. Overlay de Reprovação:** Máscara translúcida em vermelho destacando as avarias mecânicas reais sobre a peça em teste.
3. **Controles Paramétricos em Tempo Real:** Sliders para calibrar o *Limiar de Corte Canny* ($5\text{ a }70$) e a *Tolerância Máxima Permitida* ($0.5\%\text{ a }10.0\%$).
4. **Cálculo Automático de Severidade Dinâmica:** Recalculado via listener em tempo real conforme as entradas de vibração RMS e RPM são editadas.
5. **Gráfico de Probabilidades (4 Classes):** Gráfico de barras alimentado pelas porcentagens balanceadas da fusão sensorial.
6. **Log Terminal JSON:** Exibe o payload de telemetria e o retorno analítico formatado, com botão de cópia instantânea para a área de transferência.

---

## 9. Governança, Ética e Conformidade Regulatória

### 9.1. Impactos Trabalhistas e Requalificação Profissional
A introdução de sistemas de IA para manutenção preditiva em chão de fábrica reconfigura a rotina operacional:
* **Mitigação do Risco Humano:** Elimina a necessidade de inspeções visuais manuais em espaços confinados ou próximos a mancais em alta temperatura e vibração, reduzindo acidentes de trabalho.
* **Transição de Perfil:** Os inspetores manuais são requalificados para atuar como *Auditores de Sistemas Preditivos*, supervisionando os alertas emitidos pelo modelo e validando manutenções prescritivas em colaboração com o CLP/SCADA.

### 9.2. Conformidade com a LGPD (Lei Geral de Proteção de Dados - Lei 13.709/2018)
* **Princípio da Necessidade e Minimização:** As câmeras industriais instaladas nas bancadas de ensaio possuem lentes focadas exclusivamente no corte mecânico das turbinas.
* **Anonimização Óptica Ativa:** Para impedir a captura incidental de dados biométricos de operadores e técnicos de bancada, o pipeline integra rotina de mascaramento e desfoque (*blurring*) em qualquer enquadramento periférico fora do perímetro estrito do motor.

### 9.3. Enquadramento no EU AI Act (Regulamentação Europeia de IA de Alto Risco)
* **Classificação de Alto Risco (Anexo III):** Sistemas preditivos operando em componentes críticos de propulsão e infraestrutura industrial enquadram-se na categoria de IA de Alto Risco.
* **Requisitos Cumpridos:** O projeto implementa rastreabilidade completa de logs via payloads JSON estruturados, métricas de explicabilidade física por SHAP/Gini, precisão auditável e supervisão humana mandatória (*Human-in-the-Loop*) antes da execução de paradas de máquina.

### 9.4. Cibersegurança Industrial OT/IT (Norma IEC 62443 & Modelo Purdue)
* **Defesa Contra Envenenamento de Dados (*Data Poisoning*):** O pipeline de ETL valida todas as entradas de telemetria através de limites físicos do domínio mecânico e Z-score móvel, rejeitando vetores corrompidos antes da gravação em disco.
* **Segregação de Redes e Operação Air-Gapped:** O microsserviço foi concebido para rodar em nós de *Edge Computing* locais na Zona 3 do Modelo Purdue (Nível de Operações de Fabricação), sem dependência obrigatória de conexão com nuvem externa pública.

---

## 10. Instalação, Configuração e Execução

### 10.1. Pré-requisitos
* Python 3.10 ou superior instalado.
* Git instalado.
* Docker e Docker-Compose (opcional, para execução em contêiner).

---

### 10.2. Instalação em Ambiente Local (Virtualenv)

1. **Clonar o repositório:**
```bash
git clone https://github.com/mateus-s-cavalheiro/AeroMotion-Dynamics.git
cd AeroMotion-Dynamics
```

2. **Criar e ativar o ambiente virtual:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar as dependências rastreadas:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 10.3. Execução do Pipeline de ETL e Notebooks

1. **Executar o pré-processamento de séries temporais:**
```bash
python src/preprocessing.py
```
*Gera o arquivo `data/processed/dados_sensores_aeromotion_tratado.csv` pronto para os modelos.*

2. **Abrir e validar os notebooks analíticos:**
```bash
jupyter notebook
```
* Navegue até `notebooks/01_etapa1_dados.ipynb` para visualizar a EDA e os gráficos de correlação.
* Navegue até `notebooks/02_etapa2_modelagem.ipynb` para validar as matrizes de confusão e os relatórios de classificação multiclasse.

---

### 10.4. Execução da API FastAPI

Inicie o servidor Uvicorn com recarregamento automático:
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

* **Documentação Interativa Swagger UI:** Acesse via navegador em `http://127.0.0.1:8000/docs`
* **Documentação Redoc:** Acesse em `http://127.0.0.1:8000/redoc`

---

### 10.5. Execução do Dashboard SCADA Industrial

Abra o arquivo `index.html` diretamente em seu navegador web (Google Chrome, Edge ou Firefox):
```bash
# Windows
start index.html

# Linux
xdg-open index.html
```

* Utilize os botões de **Simulação de Falha Rápida** (*Nominal*, *Falha Rolamento*, *Falha Rotor*, *Cavitação Pás*) para injetar assinaturas visuais e de telemetria instantaneamente.
* Clique em **"Executar Auditoria Técnica Completa"** para acionar a fusão sensorial e atualizar as 4 barras de probabilidade em conformidade com as medições ópticas.

---

### 10.6. Execução via Docker e Docker-Compose (Edge Mode)

Para compilar e implantar o microsserviço isolado em contêiner Docker:

```bash
# Compilar e subir o contêiner
docker-compose up --build -d

# Visualizar logs da aplicação em execução
docker logs -f aeromotion_predictive_service

# Verificar o health check do contêiner
docker inspect --format='{{json .State.Health}}' aeromotion_predictive_service

# Derrubar o serviço
docker-compose down
```

---

## 11. Equipe de Desenvolvimento

Projeto desenvolvido para a disciplina de **Inteligência Artificial e Ferramentas para Análise de Dados em Sistemas Industriais**, simulando a operação técnica e regulatória da empresa fictícia **AeroMotion Dynamics S.A.**.