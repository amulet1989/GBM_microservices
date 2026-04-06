import os
import pydicom
import dicom2nifti

def is_dicom_file(filepath: str) -> bool:
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False

def convert_dicom_to_nifti_for_case(case_dir: str):
    """
    Escanea las subcarpetas del caso. Si encuentra DICOMs, 
    los agrupa, los convierte a NIfTI y borra los DICOM originales.
    """
    dicom_found = False
    for root, dirs, files in os.walk(case_dir):
        dicom_files = [f for f in files if is_dicom_file(os.path.join(root, f))]
        
        if dicom_files:
            dicom_found = True
            print(f"Convirtiendo {len(dicom_files)} archivos DICOM en {root}...")
            # dicom2nifti convierte la serie contenida en 'root' y deja un .nii.gz ahí mismo
            try:
                dicom2nifti.convert_directory(root, root, compression=True, reorient=True)
                # Limpiar DICOMs
                for dcm in dicom_files:
                    os.remove(os.path.join(root, dcm))
            except Exception as e:
                print(f"Error convirtiendo DICOM en {root}: {e}")
                
    return dicom_found