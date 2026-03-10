import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List

app = FastAPI(title="MRI Inference API")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.post("/upload/{case_id}")
async def upload_manual_files(
    case_id: str,
    category: str = Form(...),
    modality: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Carga manual con regla de REEMPLAZO por modalidad."""
    category_path = os.path.join(UPLOAD_DIR, case_id, category)
    os.makedirs(category_path, exist_ok=True)
    
    # --- NUEVA LÓGICA DE REEMPLAZO ---
    for existing_file in os.listdir(category_path):
        if modality in existing_file:
            os.remove(os.path.join(category_path, existing_file))
    # ---------------------------------
            
    saved_files = []
    for file in files:
        file_location = os.path.join(category_path, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        saved_files.append(file.filename)
        
    return {"category": category, "modality": modality, "saved_files": saved_files}

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