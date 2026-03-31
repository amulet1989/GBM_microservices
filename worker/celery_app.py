import os
from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery(
    "inference_tasks", 
    broker=broker_url, 
    backend=result_backend,
    include=["tasks"]  # Esto le dice exactamente qué archivo cargar
)