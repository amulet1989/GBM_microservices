import os
import ants

def run_brats_pipeline(case_dir: str, atlas_path: str, atlas_mask_path: str):
    print(f"Iniciando preprocesamiento BraTS en: {case_dir}")
    
    if not os.path.exists(atlas_path) or not os.path.exists(atlas_mask_path):
        raise FileNotFoundError("Atlas o Máscara no encontrados.")
        
    # 1. Cargar el Atlas y su máscara de tejidos
    atlas_img = ants.image_read(atlas_path)
    atlas_tissues = ants.image_read(atlas_mask_path)
    
    # 2. Binarizar máscara (>0 es cerebro)
    brain_mask = ants.threshold_image(atlas_tissues, low_thresh=0.5, high_thresh=10.0)
    
    # IMPORTANTE: Forzar que el atlas comparta EXACTAMENTE el espacio físico de su máscara
    # (Por si tienen diferencias minúsculas en la cabecera NIfTI)
    atlas_img = ants.resample_image_to_target(atlas_img, brain_mask, interp_type='linear')

    # 3. Recolectar archivos NIfTI del caso
    files = {}
    for root, _, filenames in os.walk(case_dir):
        for f in filenames:
            if f.endswith('.nii.gz'):
                files[f] = os.path.join(root, f)

    t1gd_key = next((k for k in files.keys() if "T1GD" in k), None)
    if not t1gd_key:
        raise ValueError("No se encontró la modalidad T1Gd.")

    t1gd_path = files[t1gd_key]
    t1gd_raw = ants.image_read(t1gd_path)

    # 4. Corrección N4 a T1Gd
    print("Aplicando N4 a T1Gd y registrando al Atlas SRI...")
    t1gd_n4 = ants.n4_bias_field_correction(t1gd_raw)
    
    # 5. Registro Rígido al Atlas
    reg_t1gd_to_atlas = ants.registration(
        fixed=atlas_img, 
        moving=t1gd_n4, 
        type_of_transform='Rigid'
    )
    
    # 6. Forzar que la imagen resultante ocupe el espacio EXACTO de la máscara
    # Esto asegura resolución de 1x1x1mm si el atlas la tiene, y evita el error de "physical space"
    t1gd_aligned = ants.apply_transforms(
        fixed=brain_mask,  # El objetivo es la cuadrícula de la máscara
        moving=t1gd_n4, 
        transformlist=reg_t1gd_to_atlas['fwdtransforms'],
        interpolator='linear'
    )

    # 7. Skull Stripping seguro (multiplicación)
    print("Aplicando Skull Stripping desde el Atlas...")
    t1gd_final = t1gd_aligned * brain_mask
    ants.image_write(t1gd_final, t1gd_path)

    # 8. Procesar las demás modalidades
    for filename, filepath in files.items():
        if filename == t1gd_key:
            continue
            
        print(f"Procesando {filename}...")
        mod_raw = ants.image_read(filepath)
        
        # N4 solo a estructurales
        if "T1." in filename or "T2." in filename:
            mod_for_reg = ants.n4_bias_field_correction(mod_raw)
        else:
            mod_for_reg = mod_raw

        # Co-registro a T1Gd original
        reg_mod_to_t1gd = ants.registration(
            fixed=t1gd_raw, 
            moving=mod_for_reg, 
            type_of_transform='Rigid'
        )
        
        # Aplicar ambas transformaciones (A -> B -> Atlas) 
        # y forzar salida en el grid de la máscara (brain_mask)
        transforms_to_apply = reg_t1gd_to_atlas['fwdtransforms'] + reg_mod_to_t1gd['fwdtransforms']
        mod_aligned = ants.apply_transforms(
            fixed=brain_mask, # Target exacto
            moving=mod_raw,
            transformlist=transforms_to_apply,
            interpolator='linear'
        )
        
        # Skull Stripping
        mod_final = mod_aligned * brain_mask
        ants.image_write(mod_final, filepath)

    return True