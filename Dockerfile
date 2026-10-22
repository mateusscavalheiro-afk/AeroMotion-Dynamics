# Imagem base oficial do Python optimizada
FROM python:3.10-slim

# Evita que o Python escreva ficheiros .pyc e activa buffer de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define directório de trabalho
WORKDIR /app

# Instalação de dependências do sistema necessárias para o OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalação dos pacotes Python via cache de dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cópia do código-fonte e dos artefactos treinados de IA
COPY src/ ./src/
COPY ficheiro_IA/ ./ficheiro_IA/

# Exposição da porta de comunicação da API RESTful
EXPOSE 8000

# Execução do microsserviço assíncrono via Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]