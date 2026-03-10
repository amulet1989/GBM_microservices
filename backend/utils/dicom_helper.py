import os
import pydicom
import dicom2nifti
import shutil

def is_dicom_file(filepath: str) -> bool:
    """Verifica de forma segura si un archivo es un DICOM válido."""
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False

def convert_dicom_to_nifti(input_dir: str, output_dir: str):
    """
    Escanea un directorio en busca de archivos DICOM.
    Si los encuentra, los agrupa y convierte a un volumen NIfTI.
    """
    # 1. Identificar si hay archivos DICOM en el directorio
    dicom_files = [f for f in os.listdir(input_dir) if is_dicom_file(os.path.join(input_dir, f))]
    
    if not dicom_files:
        return {"status": "skipped", "message": "No se encontraron archivos DICOM."}
        
    try:
        # 2. dicom2nifti requiere un directorio de salida. 
        # Convertimos toda la serie DICOM contenida en input_dir a NIfTI en output_dir.
        dicom2nifti.convert_directory(input_dir, output_dir, compression=True, reorient=True)
        
        # 3. Limpieza: Borrar los archivos DICOM originales para no ocupar espacio doble
        for dcm in dicom_files:
            os.remove(os.path.join(input_dir, dcm))
            
        return {"status": "success", "message": f"Convertidos {len(dicom_files)} archivos DICOM a NIfTI."}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}