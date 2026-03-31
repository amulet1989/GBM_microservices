FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar otras librerías
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Ya no instalamos torch aquí, solo los requerimientos extra (monai, celery, nibabel, etc.)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando para iniciar el worker de Celery (concurrency=1 es clave en GPU para evitar colisiones de memoria)
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--concurrency=1"]