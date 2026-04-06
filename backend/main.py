import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from celery import Celery
from celery import chain
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

# Configurar cliente Celery para enviar tareas
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
celery_app = Celery("inference_tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app = FastAPI(title="MRI Inference API")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directorio base para resultados
RESULTS_DIR = os.getenv("RESULTS_DIR", "/app/shared_data/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODALITY_ORDER = {
    "images_DSC": ["DSC_ap-rCBV", "DSC_PH", "DSC_PSR"],
    "images_DTI": ["DTI_AD", "DTI_FA", "DTI_RD", "DTI_TR"],
    "images_structural": ["FLAIR", "T1.", "T1GD", "T2"],
}

@app.get("/")
def read_root():
    return {"status": "Backend de Inferencia MRI activo"}

@app.get("/cases/")
def list_cases():
    """Devuelve una lista con los IDs de los casos actualmente almacenados."""
    if not os.path.exists(UPLOAD_DIR):
        return {"cases": []}
    cases = [d for d in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
    return {"cases": sorted(cases)}

@app.delete("/cases/{case_id}")
def delete_case(case_id: str):
    """Elimina completamente la carpeta de un caso y todos sus archivos."""
    case_path = os.path.join(UPLOAD_DIR, case_id)
    if os.path.exists(case_path):
        shutil.rmtree(case_path)
        return {"message": f"Caso {case_id} eliminado correctamente."}
    raise HTTPException(status_code=404, detail="Caso no encontrado")


@app.post("/upload_bulk/{case_id}")
async def upload_bulk_files(case_id: str, files: List[UploadFile] = File(...)):
    """Carga masiva con regla de REEMPLAZO por modalidad."""
    case_path = os.path.join(UPLOAD_DIR, case_id)
    os.makedirs(case_path, exist_ok=True)
    
    saved_summary = []
    unmatched_files = []

    for file in files:
        filename = file.filename
        matched_category = None
        matched_pattern = None
        
        # Identificar categoría y el patrón exacto (ej. "DTI_AD")
        for category, patterns in MODALITY_ORDER.items():
            for pattern in patterns:
                if pattern in filename:
                    matched_category = category
                    matched_pattern = pattern
                    break
            if matched_category:
                break
        
        if matched_category:
            category_path = os.path.join(case_path, matched_category)
            os.makedirs(category_path, exist_ok=True)
            
            # --- NUEVA LÓGICA DE REEMPLAZO ---
            # Eliminar cualquier archivo que ya exista para esta modalidad específica
            for existing_file in os.listdir(category_path):
                if matched_pattern in existing_file:
                    os.remove(os.path.join(category_path, existing_file))
            # ---------------------------------
            
            # Guardar el nuevo archivo
            file_location = os.path.join(category_path, filename)
            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())
                
            saved_summary.append({"file": filename, "category": matched_category})
        else:
            unmatched_files.append(filename)

    return {
        "message": f"Procesamiento masivo completado para el caso {case_id}",
        "saved_files": saved_summary,
        "unmatched_files": unmatched_files
    }

# @app.post("/upload/{case_id}")
# async def upload_manual_files(
#     case_id: str,
#     category: str = Form(...),
#     modality: str = Form(...),
#     files: List[UploadFile] = File(...)
# ):
#     """Carga manual con regla de REEMPLAZO por modalidad."""
#     category_path = os.path.join(UPLOAD_DIR, case_id, category)
#     os.makedirs(category_path, exist_ok=True)
    
#     # --- NUEVA LÓGICA DE REEMPLAZO ---
#     for existing_file in os.listdir(category_path):
#         if modality in existing_file:
#             os.remove(os.path.join(category_path, existing_file))
#     # ---------------------------------
            
#     saved_files = []
#     for file in files:
#         file_location = os.path.join(category_path, file.filename)
#         with open(file_location, "wb+") as file_object:
#             file_object.write(await file.read())
#         saved_files.append(file.filename)
        
#     return {"category": category, "modality": modality, "saved_files": saved_files}

@app.post("/upload/{case_id}")
async def upload_manual_files(
    case_id: str,
    category: str = Form(...),
    modality: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Carga manual con aislamiento inteligente para DICOMs."""
    category_path = os.path.join(UPLOAD_DIR, case_id, category)
    os.makedirs(category_path, exist_ok=True)
    
    # Limpiar archivos anteriores de la misma modalidad
    for existing_file in os.listdir(category_path):
        if modality in existing_file:
            os.remove(os.path.join(category_path, existing_file))
            
    # Si viene más de un archivo o no tiene extensión .nii.gz, asumimos que es una serie DICOM
    is_dicom_series = len(files) > 1 or not files[0].filename.endswith('.nii.gz')
    
    if is_dicom_series:
        # Aislar los DICOMs en una carpeta temporal con el nombre de la modalidad
        dicom_dir = os.path.join(category_path, f"{modality}_dicoms")
        os.makedirs(dicom_dir, exist_ok=True)
        
        for file in files:
            file_location = os.path.join(dicom_dir, file.filename)
            with open(file_location, "wb+") as f:
                f.write(await file.read())
    else:
        # Es un NIfTI único, se guarda normal
        for file in files:
            file_location = os.path.join(category_path, file.filename)
            with open(file_location, "wb+") as f:
                f.write(await file.read())
        
    return {"category": category, "modality": modality, "saved_files": [f.filename for f in files]}

# --- NUEVO ENDPOINT PARA ELIMINAR MODALIDADES ESPECÍFICAS ---
@app.delete("/cases/{case_id}/{category}/{modality}")
def delete_modality(case_id: str, category: str, modality: str):
    """Elimina los archivos de una modalidad específica de un caso."""
    category_path = os.path.join(UPLOAD_DIR, case_id, category)
    if os.path.exists(category_path):
        deleted = False
        for existing_file in os.listdir(category_path):
            if modality in existing_file:
                os.remove(os.path.join(category_path, existing_file))
                deleted = True
        if deleted:
            return {"message": f"Modalidad {modality} eliminada correctamente."}
    raise HTTPException(status_code=404, detail="Modalidad no encontrada o ya eliminada.")


@app.get("/status/{case_id}")
def check_upload_status(case_id: str):
    """Revisa el estado de las modalidades para un caso específico."""
    status_report = {}
    case_path = os.path.join(UPLOAD_DIR, case_id)
    
    for category, modalities in MODALITY_ORDER.items():
        category_path = os.path.join(case_path, category)
        uploaded_files = os.listdir(category_path) if os.path.exists(category_path) else []
        status_report[category] = []
        
        for modality in modalities:
            matched_files = [f for f in uploaded_files if modality in f]
            if matched_files:
                status_report[category].append({"modality": modality, "status": "Presente", "files": matched_files})
            else:
                status_report[category].append({"modality": modality, "status": "Faltante", "files": []})
                
    return {"status": status_report}

# Endpoint para iniciar la inferencia y otro para consultar el estado de la tarea
@app.post("/infer/{case_id}")
def start_inference(case_id: str):
    """Envía la tarea de inferencia a la cola de Celery."""
    case_path = os.path.join(UPLOAD_DIR, case_id)
    if not os.path.exists(case_path):
        raise HTTPException(status_code=404, detail="El caso no existe.")
    
    # Llama a la tarea por su nombre registrado en el worker
    task = celery_app.send_task("tasks.run_inference_task", args=[case_id])
    return {"message": "Tarea encolada", "task_id": task.id}

@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    """El frontend consultará este endpoint para ver el progreso."""
    task_result = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }
    
    if task_result.status == 'PROGRESS':
        response["message"] = task_result.info.get('message', 'Procesando...')
    elif task_result.status == 'SUCCESS':
        response["result"] = task_result.result
    elif task_result.status == 'FAILURE':
        response["error"] = str(task_result.info)
        
    return response

@app.get("/results/{case_id}")
def check_results(case_id: str):
    """Verifica si existen resultados procesados para un caso específico."""
    case_results_dir = os.path.join(RESULTS_DIR, case_id)
    seg_file = os.path.join(case_results_dir, f"segmentation_{case_id}.nii.gz")
    prob_file = os.path.join(case_results_dir, f"prob_maps_{case_id}.nii.gz")
    
    return {
        "segmentation": os.path.exists(seg_file),
        "prob_maps": os.path.exists(prob_file)
    }

@app.get("/download/{case_id}/{file_type}")
def download_result(case_id: str, file_type: str):
    """Envía el archivo NIfTI solicitado para ser descargado por el usuario."""
    if file_type not in ["segmentation", "prob_maps"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo inválido.")
    
    file_path = os.path.join(RESULTS_DIR, case_id, f"{file_type}_{case_id}.nii.gz")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado. Quizás la inferencia no ha terminado.")
        
    return FileResponse(
        path=file_path, 
        filename=f"{file_type}_{case_id}.nii.gz", 
        media_type="application/gzip"
    )

@app.post("/process/{case_id}")
def start_processing(case_id: str, run_preprocessing: bool = True):
    """Inicia el pipeline. Puede incluir preprocesamiento o ir directo a inferencia."""
    case_path = os.path.join(UPLOAD_DIR, case_id)
    if not os.path.exists(case_path):
        raise HTTPException(status_code=404, detail="El caso no existe.")
    
    if run_preprocessing:
        task_chain = chain(
            celery_app.signature("tasks.run_preprocessing_task", args=[case_id]),
            # Añadimos immutable=True para que ignore el output de la tarea anterior
            celery_app.signature("tasks.run_inference_task", args=[case_id], immutable=True)
        )()
        return {"message": "Preprocesamiento + Inferencia encolados", "task_id": task_chain.id}
    else:
        task = celery_app.send_task("tasks.run_inference_task", args=[case_id])
        return {"message": "Inferencia encolada", "task_id": task.id}