import os
import shutil
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
    Busca carpetas temporales '_dicoms', convierte la serie a NIfTI,
    y renombra el archivo resultante con la modalidad exacta.
    """
    dicom_found = False
    
    for root, dirs, files in os.walk(case_dir, topdown=False):
        # Si estamos dentro de una de las carpetas aisladas que creó el backend
        if root.endswith("_dicoms"):
            dicom_found = True
            modality_name = os.path.basename(root).replace("_dicoms", "")
            parent_category_dir = os.path.dirname(root)
            
            print(f"Convirtiendo serie DICOM para modalidad: {modality_name}...")
            try:
                # dicom2nifti generará un .nii.gz dentro de 'root'
                dicom2nifti.convert_directory(root, root, compression=True, reorient=True)
                
                # Buscar el archivo NIfTI generado
                generated_niftis = [f for f in os.listdir(root) if f.endswith('.nii.gz')]
                
                if generated_niftis:
                    original_nifti_path = os.path.join(root, generated_niftis[0])
                    # Renombramos forzosamente para que el pipeline lo reconozca
                    new_nifti_path = os.path.join(parent_category_dir, f"{modality_name}.nii.gz")
                    
                    # Mover y renombrar
                    shutil.move(original_nifti_path, new_nifti_path)
                    print(f"✅ DICOM convertido y renombrado a {modality_name}.nii.gz")
                
                # Eliminar la carpeta temporal de DICOMs
                shutil.rmtree(root)
                
            except Exception as e:
                print(f"❌ Error convirtiendo DICOM en {root}: {e}")
                
    return dicom_found