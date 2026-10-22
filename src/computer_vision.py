import cv2
import numpy as np
from typing import Dict, Any, Optional

# =====================================================================
# 1. INSPEÇÃO TÉRMICA INFRAVERMELHA (ESPAÇO DE COR HSV)
# =====================================================================
def inspecionar_termografia_hsv(caminho_ou_buffer, limite_area_critica: int = 500) -> Dict[str, Any]:
    """
    Processa imagens infravermelhas convertendo para HSV e aplicando limiarização
    para delimitar zonas críticas de sobreaquecimento (hotspots).
    """
    img = carregar_imagem(caminho_ou_buffer)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Faixas de matiz (Hue) correspondentes ao calor crítico (tons de vermelho)
    vermelho_baixo_1 = np.array([0, 120, 150])
    vermelho_alto_1 = np.array([10, 255, 255])
    vermelho_baixo_2 = np.array([170, 120, 150])
    vermelho_alto_2 = np.array([180, 255, 255])
    
    mascara_1 = cv2.inRange(hsv, vermelho_baixo_1, vermelho_alto_1)
    mascara_2 = cv2.inRange(hsv, vermelho_baixo_2, vermelho_alto_2)
    mascara_hotspots = cv2.bitwise_or(mascara_1, mascara_2)
    
    area_hotspot_px = int(np.sum(mascara_hotspots > 0))
    tem_superaquecimento = area_hotspot_px > limite_area_critica
    
    return {
        "tipo_inspecao": "Termografia HSV",
        "area_hotspot_px": area_hotspot_px,
        "limiar_critico_px": limite_area_critica,
        "alerta_superaquecimento": tem_superaquecimento,
        "status": "REPROVADO_SUPERAQUECIMENTO" if tem_superaquecimento else "APROVADO_TERMICO"
    }

# =====================================================================
# 2. FILTRO DE CLUSTERS E RUÍDO ESTRUTURAL
# =====================================================================
def refinar_mascara_dano(mascara_bruta: np.ndarray, min_area_cluster: int = 30, kernel_size: int = 3) -> np.ndarray:
    """
    Elimina píxeis isolados e variações normais de textura, preservando apenas
    conjuntos contíguos de píxeis (falhas mecânicas reais).
    """
    if len(mascara_bruta.shape) == 3:
        mascara_bruta = cv2.cvtColor(mascara_bruta, cv2.COLOR_BGR2GRAY)
    _, mascara_binaria = cv2.threshold(mascara_bruta, 127, 255, cv2.THRESH_BINARY)

    # 1. Abertura morfológica para limpar ruídos de alta frequência
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mascara_morfologica = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel)

    # 2. Análise de componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mascara_morfologica, connectivity=8
    )

    mascara_limpa = np.zeros_like(mascara_morfologica)

    # Label 0 é o fundo preto; avalia a partir do label 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_cluster:
            mascara_limpa[labels == label] = 255

    return mascara_limpa

# =====================================================================
# 3. INSPEÇÃO GEOMÉTRICA ADAPTATIVA (CANNY, CONTORNOS E CIRCULARIDADE)
# =====================================================================
ROIS_NOMINAIS = {
    "rolamento": {
        "coords": (144, 313, 360, 609),
        "nome": "Rolamentos / Mancal",
        "falha_associada": "1: Desgaste de Rolamento"
    },
    "rotor": {
        "coords": (372, 250, 720, 645),
        "nome": "Rotor / Eixo Central",
        "falha_associada": "2: Desbalanceamento de Rotor"
    },
    "pas": {
        "coords": (900, 197, 1104, 698),
        "nome": "Pás da Turbina / Impulsor",
        "falha_associada": "3: Cavitação / Superaquecimento"
    }
}

def carregar_imagem(caminho_ou_buffer) -> np.ndarray:
    if isinstance(caminho_ou_buffer, str):
        img = cv2.imread(caminho_ou_buffer)
    else:
        nparr = np.frombuffer(caminho_ou_buffer, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    if img is None:
        raise ValueError("Não foi possível carregar ou descodificar a imagem.")
    return img

def analisar_componente_adaptativo(
    img_teste: np.ndarray,
    img_gabarito: np.ndarray,
    chave_roi: str,
    sensibilidade_corte: int = 40,
    tolerancia_max_pct: float = 2.0,
    min_area_cluster: int = 30
) -> Dict[str, Any]:
    info_roi = ROIS_NOMINAIS[chave_roi]
    gx1, gy1, gx2, gy2 = info_roi["coords"]
    
    template_gab = img_gabarito[gy1:gy2, gx1:gx2]
    h_temp, w_temp = template_gab.shape[:2]
    
    # Busca com tolerância espacial de posicionamento
    margem = 35
    h_teste, w_teste = img_teste.shape[:2]
    bx1 = max(0, gx1 - margem)
    by1 = max(0, gy1 - margem)
    bx2 = min(w_teste, gx2 + margem)
    by2 = min(h_teste, gy2 + margem)
    
    regiao_busca = img_teste[by1:by2, bx1:bx2]
    res_match = cv2.matchTemplate(regiao_busca, template_gab, cv2.TM_CCOEFF_NORMED)
    _, max_corr, _, max_loc = cv2.minMaxLoc(res_match)
    
    real_x1 = bx1 + max_loc[0]
    real_y1 = by1 + max_loc[1]
    real_x2 = real_x1 + w_temp
    real_y2 = real_y1 + h_temp
    
    roi_alinhada = img_teste[real_y1:real_y2, real_x1:real_x2]
    
    gray_gab = cv2.cvtColor(template_gab, cv2.COLOR_BGR2GRAY)
    gray_teste = cv2.cvtColor(roi_alinhada, cv2.COLOR_BGR2GRAY)
    
    blur_gab = cv2.GaussianBlur(gray_gab, (5, 5), 0)
    blur_teste = cv2.GaussianBlur(gray_teste, (5, 5), 0)
    
    # Deteção de arestas com Canny
    edges_gab = cv2.Canny(blur_gab, 50, 150)
    edges_teste = cv2.Canny(blur_teste, 50, 150)
    
    diff_arestas = cv2.absdiff(edges_teste, edges_gab)
    _, mask_bruta = cv2.threshold(diff_arestas, sensibilidade_corte, 255, cv2.THRESH_BINARY)
    
    # Eliminação de ruídos finos e píxeis isolados
    mask_dano = refinar_mascara_dano(mask_bruta, min_area_cluster=min_area_cluster)
    
    # Validação de contornos e circularidade para esferas do rolamento
    contornos_teste, _ = cv2.findContours(edges_teste, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularidades = []
    for c in contornos_teste:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if area > 50 and peri > 0:
            circularidades.append(4 * np.pi * (area / (peri ** 2)))
    media_circ = round(float(np.mean(circularidades)), 3) if circularidades else 0.0
    
    total_pixels = h_temp * w_temp
    pixels_defeituosos = int(np.sum(mask_dano > 0))
    dano_pct = round((pixels_defeituosos / total_pixels) * 100, 2)
    
    status_aprovacao = "APROVADO" if dano_pct <= tolerancia_max_pct else "REPROVADO"
    
    return {
        "componente": info_roi["nome"],
        "falha_associada": info_roi["falha_associada"],
        "similaridade_posicionamento": round(float(max_corr), 4),
        "circularidade_media": media_circ,
        "dano_estrutural_pct": dano_pct,
        "tolerancia_limite_pct": tolerancia_max_pct,
        "status": status_aprovacao
    }

# =====================================================================
# 4. FUSÃO SENSORIAL E BALANCEAMENTO DINÂMICO DE PROBABILIDADES
# =====================================================================
# =====================================================================
# 4. FUSÃO SENSORIAL E BALANCEAMENTO DINÂMICO DE PROBABILIDADES
# =====================================================================
def distribuir_probabilidades_multimodal(
    laudos_rois: Dict[str, Dict[str, Any]],
    telemetria_iot: Optional[Dict[str, float]] = None,
    tolerancia_max: float = 3.0
) -> Dict[str, Any]:
    """
    Calcula o Risco Composto e distribui as probabilidades do gráfico
    estritamente proporcional à severidade de cada componente inspecionado.
    """
    dano_rol = float(laudos_rois.get("rolamento", {}).get("dano_estrutural_pct", 0.0))
    dano_rot = float(laudos_rois.get("rotor", {}).get("dano_estrutural_pct", 0.0))
    dano_pas = float(laudos_rois.get("pas", {}).get("dano_estrutural_pct", 0.0))
    
    soma_danos = dano_rol + dano_rot + dano_pas
    dano_maximo = max(dano_rol, dano_rot, dano_pas)
    
    houve_reprovacao_visual = any(
        roi.get("status") == "REPROVADO" for roi in laudos_rois.values()
    ) or (dano_maximo > tolerancia_max)

    # 1. Risco Composto e probabilidade de Operação Normal
    if houve_reprovacao_visual:
        # Escala o risco composto entre 85% e 99.6% com base na severidade máxima
        fator_severidade = min(dano_maximo / (tolerancia_max * 2.5), 1.0)
        risco_composto = 85.0 + (14.6 * fator_severidade)
        prob_normal = max(0.4, round(100.0 - risco_composto, 1))
    else:
        # Operação nominal: risco residual de 1.5% a 5.0%
        fator_nominal = dano_maximo / max(tolerancia_max, 0.1)
        risco_composto = max(1.5, min(5.0, fator_nominal * 4.0))
        prob_normal = round(100.0 - risco_composto, 1)

    risco_distribuivel = 100.0 - prob_normal

    # 2. Rateio estritamente proporcional aos danos medidos nas ROIs
    if soma_danos > 0:
        p_rol = round(risco_distribuivel * (dano_rol / soma_danos), 1)
        p_rot = round(risco_distribuivel * (dano_rot / soma_danos), 1)
        p_pas = round(risco_distribuivel * (dano_pas / soma_danos), 1)
    else:
        p_rol = p_rot = p_pas = round(risco_distribuivel / 3.0, 1)

    # 3. Ajuste de resíduo decimal para garantir soma de 100.0%
    delta = round(100.0 - (prob_normal + p_rol + p_rot + p_pas), 1)
    p_rot = round(p_rot + delta, 1)

    # 4. Diagnóstico primário baseado no maior dano real
    ranking = [
        ("1: Desgaste de Rolamento", p_rol),
        ("2: Desbalanceamento de Rotor", p_rot),
        ("3: Cavitação / Superaquecimento", p_pas)
    ]
    classe_principal = max(ranking, key=lambda x: x[1])[0]

    return {
        "risco_composto_pct": round(float(risco_composto), 1),
        "classe_diagnosticada": "0: Operação Normal" if prob_normal >= 60.0 else classe_principal,
        "probabilidade_por_estado": {
            "0: Normal": prob_normal,
            "1: Rolamento": p_rol,
            "2: Rotor": p_rot,
            "3: Cavitação": p_pas
        },
        # Array ordenado diretamente consumível pelo gráfico do frontend:
        "probabilidades_grafico": [prob_normal, p_rol, p_rot, p_pas]
    }

# =====================================================================
# 5. EXECUÇÃO INTEGRADA DA INSPEÇÃO
# =====================================================================
def inspecionar_motor_completo(
    caminho_teste,
    caminho_gabarito,
    sensibilidade: int = 40,
    tolerancia_max: float = 2.0,
    min_area_cluster: int = 30,
    telemetria_iot: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    img_gab = carregar_imagem(caminho_gabarito)
    img_test = carregar_imagem(caminho_teste)
    
    laudos = {}
    componentes_com_falha = []
    
    for chave in ["rolamento", "rotor", "pas"]:
        resultado = analisar_componente_adaptativo(
            img_test, img_gab, chave, sensibilidade, tolerancia_max, min_area_cluster
        )
        laudos[chave] = resultado
        if resultado["status"] == "REPROVADO":
            componentes_com_falha.append(resultado)
            
    aprovacao_geral = len(componentes_com_falha) == 0
    
    # Cálculo das probabilidades balanceadas para o gráfico
    diagnostico_multimodal = distribuir_probabilidades_multimodal(
        laudos, telemetria_iot, tolerancia_max=tolerancia_max
    )
    
    return {
        "status_geral": "APROVADO" if aprovacao_geral else "REPROVADO",
        "qtd_falhas_detectadas": len(componentes_com_falha),
        "falhas_identificadas": [f["falha_associada"] for f in componentes_com_falha],
        "detalhamento_por_componente": laudos,
        "diagnostico_multimodal": diagnostico_multimodal
    }