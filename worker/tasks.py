import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from celery_app import celery_app
from monai import transforms
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
import subprocess
from brats_preprocess import run_brats_pipeline
import sys

# Asegúrate de importar el dicom_helper. Como está en el backend, 
# la forma más fácil es tener una copia en /worker/utils/dicom_helper.py 
# o compartir la carpeta.
from utils.dicom_helper import convert_dicom_to_nifti_for_case

# Variables globales
models_loaded = False
model1 = None
model2 = None
projection_head1 = None
projection_head2 = None
classifier1 = None
classifier2 = None
device = None

decoder_features_model1 = None
decoder_features_model2 = None

def decoder_hook_fn_model1(module, input, output):
    global decoder_features_model1
    decoder_features_model1 = output

def decoder_hook_fn_model2(module, input, output):
    global decoder_features_model2
    decoder_features_model2 = output

# Orden estricto de los 11 canales esperado por el modelo
MODALITY_ORDER_LIST = [
    "DSC_ap-rCBV", "DSC_PH", "DSC_PSR",       # DSC (0, 1, 2)
    "DTI_AD", "DTI_FA", "DTI_RD", "DTI_TR",   # DTI (3, 4, 5, 6)
    "FLAIR", "T1.", "T1GD", "T2"              # Structural (7, 8, 9, 10)
]

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=48, hidden_dim1=256, hidden_dim2=128, output_dim=128, dropout_p=0.3):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x): return self.net(x)

class Classifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim1=256, hidden_dim2=128, num_classes=3, dropout_p=0.3):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )
    def forward(self, x): return self.net(x)

def define_model(model_path, device):
    model = SwinUNETR(
        img_size=(128, 128, 128), in_channels=11, out_channels=2,
        feature_size=48, drop_rate=0.0, attn_drop_rate=0.0,
        dropout_path_rate=0.0, use_checkpoint=True
    )
    loaded_model = torch.load(model_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(loaded_model)
    return model

def get_execution_device():
    env_device = os.getenv("EXECUTION_DEVICE", "auto").lower()
    if env_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    elif env_device == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_all_models():
    global models_loaded, model1, model2, projection_head1, projection_head2, classifier1, classifier2, device
    if models_loaded: return

    device = get_execution_device()
    models_dir = os.getenv("MODELS_DIR", "/app/shared_data/trained_models")
    
    print(f"Montando modelos en el dispositivo: {device}...")
    
    # 1. SwinUNETR
    model1 = define_model(os.path.join(models_dir, "vtzpbajf_best_model_pipe1", "model.pt"), device).to(device)
    model2 = define_model(os.path.join(models_dir, "1dhzmigz_best_model_pipe2", "model.pt"), device).to(device)
    model1.eval()
    model2.eval()

    # --- REGISTRO DE HOOKS (Exactamente como en el código experimental) ---
    model1.decoder1.conv_block.register_forward_hook(decoder_hook_fn_model1)
    model2.decoder1.conv_block.register_forward_hook(decoder_hook_fn_model2)
    # ----------------------------------------------------------------------

    # 2. Projection Heads y Classifiers
    print("Cargando Projection Heads y Classifiers...")
    projection_head1 = ProjectionHead(input_dim=48).to(device)
    projection_head1.load_state_dict(torch.load(os.path.join(models_dir, "contrastive_projection_head_final_new_pipe1_v01_m1.pth"), map_location=device, weights_only=False))
    
    projection_head2 = ProjectionHead(input_dim=48).to(device)
    projection_head2.load_state_dict(torch.load(os.path.join(models_dir, "contrastive_projection_head_final_new_pipe2_m1_1dhzmigz.pth"), map_location=device, weights_only=False))
    
    classifier1 = Classifier(input_dim=128, num_classes=3).to(device)
    classifier1.load_state_dict(torch.load(os.path.join(models_dir, "supervised_classifier_final_pipe1_v01_m1.pth"), map_location=device, weights_only=False))
    
    classifier2 = Classifier(input_dim=128, num_classes=3).to(device)
    classifier2.load_state_dict(torch.load(os.path.join(models_dir, "supervised_classifier_final_pipe2_m1_1dhzmigz.pth"), map_location=device, weights_only=False))
    
    projection_head1.eval()
    projection_head2.eval()
    classifier1.eval()
    classifier2.eval()
    
    models_loaded = True
    print(f"Todos los modelos cargados exitosamente en {device}.")


def generate_probability_maps(embeddings, projection_head, classifier, device, batch_size=8192):
    projection_head.eval()
    classifier.eval()
    with torch.no_grad():
        # Extraer las dimensiones reales del parche que nos entregó el hook
        _, _, H, W, D = embeddings.shape
        
        embeddings = embeddings.squeeze(0).permute(1, 2, 3, 0)  # [H, W, D, 48]
        embeddings_flat = embeddings.reshape(-1, 48)  # [H*W*D, 48]
        probs_flat = []
        
        for i in range(0, embeddings_flat.shape[0], batch_size):
            batch = embeddings_flat[i:i+batch_size].to(device)
            z = projection_head(batch)
            z = F.normalize(z, dim=1)
            logits = classifier(z)
            probs = F.softmax(logits, dim=1)
            probs_flat.append(probs.cpu())
        
        probs_flat = torch.cat(probs_flat, dim=0)
        # Usamos H, W, D dinámicos en lugar de 128, 128, 128 fijo
        probs = probs_flat.reshape(H, W, D, 3)
        probs = probs.permute(3, 0, 1, 2)  # [3, H, W, D]
        
    return probs.to(device) # Regresamos a la GPU para que MONAI pueda ensamblar

# Preprocesamiento BraTS: Corrección N4, Registro al Atlas SRI, Skull Stripping y Co-registro de Modalidades
@celery_app.task(bind=True)
def run_preprocessing_task(self, case_id: str):
    """Tarea dedicada exclusivamente a convertir DICOM y estandarizar a BraTS."""
    try:
        upload_dir = os.getenv("UPLOAD_DIR", "/app/shared_data/uploads")
        case_path = os.path.join(upload_dir, case_id)
        
        self.update_state(state='PROGRESS', meta={'message': 'Verificando y convirtiendo DICOM a NIfTI...'})
        convert_dicom_to_nifti_for_case(case_path)
        
        self.update_state(state='PROGRESS', meta={'message': 'Ejecutando pipeline de preprocesamiento (ANTs)...'})
        atlas_path = "/app/shared_data/atlas/sri24_t1.nii.gz"
        atlas_mask_path = "/app/shared_data/atlas/sri24_mask.nii.gz" # <-- NUEVO
        
        run_brats_pipeline(case_path, atlas_path, atlas_mask_path) # <-- NUEVO
        
        return {"status": "success", "message": "Preprocesamiento completado", "case_id": case_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Inferencia Principal: Carga modelos, ensambla canales, corre inferencia por ventana deslizante y guarda resultados
@celery_app.task(bind=True)
def run_inference_task(self, case_id: str):
    try:
        self.update_state(state='PROGRESS', meta={'message': 'Cargando modelos...'})
        load_all_models()
        
        upload_dir = os.getenv("UPLOAD_DIR", "/app/shared_data/uploads")
        results_dir = os.getenv("RESULTS_DIR", "/app/shared_data/results")
        case_path = os.path.join(upload_dir, case_id)
        case_results_dir = os.path.join(results_dir, case_id)
        os.makedirs(case_results_dir, exist_ok=True)

        self.update_state(state='PROGRESS', meta={'message': 'Ensamblando volumen de 11 canales...'})
        
        # 1. Escanear todos los archivos del caso
        case_files = []
        for root, _, files in os.walk(case_path):
            for file in files:
                if file.endswith('.nii.gz'):
                    case_files.append(os.path.join(root, file))
                    
        if not case_files:
            raise ValueError("No se encontraron imágenes NIfTI para este caso.")

        # 2. Ordenar y detectar faltantes
        ordered_paths = [None] * 11
        for i, pattern in enumerate(MODALITY_ORDER_LIST):
            for filepath in case_files:
                if pattern in filepath:
                    ordered_paths[i] = filepath
                    break
        
        # 3. Extraer metadata de referencia del primer archivo válido encontrado
        reference_shape = (128, 128, 128)
        affine = np.eye(4)
        for path in ordered_paths:
            if path is not None:
                img_obj = nib.load(path)
                reference_shape = img_obj.shape
                affine = img_obj.affine
                break

        # 4. Cargar imágenes y rellenar faltantes con ceros
        stacked_images = []
        for path in ordered_paths:
            if path is not None:
                data = nib.load(path).get_fdata(dtype=np.float32)
                stacked_images.append(data)
            else:
                # Si no existe, creamos un volumen vacío con la misma forma
                stacked_images.append(np.zeros(reference_shape, dtype=np.float32))
        
        # Array final: [11, H, W, D]
        stacked_numpy = np.stack(stacked_images, axis=0)
        
        # 5. Transformaciones MONAI para inferencia (SOLO Normalización, cero recortes)
        val_transform = transforms.Compose([
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
            transforms.ToTensor()
        ])
        
        # El tensor ahora mantiene su tamaño original gigante (ej. [1, 11, 240, 240, 155])
        image_tensor = val_transform(stacked_numpy).unsqueeze(0).to(device)

        self.update_state(state='PROGRESS', meta={'message': 'Ejecutando Inferencia por Ventana Deslizante (SwinUNETR)...'})
        
        # with torch.no_grad():
            
        #     def predictor_fn(patch):
        #         _ = model1(patch)
        #         _ = model2(patch)
                
        #         emb1 = decoder_features_model1
        #         emb2 = decoder_features_model2
                
        #         p1 = generate_probability_maps(emb1, projection_head1, classifier1, device)
        #         p2 = generate_probability_maps(emb2, projection_head2, classifier2, device)

        #         comb = torch.zeros_like(p1)
        #         comb[0] = p1[0]
        #         comb[1] = torch.max(p1[1], p2[1])
        #         comb[2] = p2[2]
        #         comb = comb / (comb.sum(dim=0, keepdim=True) + 1e-6)
                
        #         # TRUCO DE RENDIMIENTO: Apilamos los 3 tensores de [3, H, W, D] 
        #         # en un super-tensor de [9, H, W, D] para extraerlo todo en una sola pasada.
        #         stacked_maps = torch.cat([comb, p1, p2], dim=0)
        #         return stacked_maps.unsqueeze(0)

        #     roi_size = (128, 128, 128)
        #     sw_batch_size = 1 
        
        with torch.no_grad():
            
            def predictor_fn(patch):
                _ = model1(patch)
                _ = model2(patch)
                
                emb1 = decoder_features_model1
                emb2 = decoder_features_model2
                
                p1 = generate_probability_maps(emb1, projection_head1, classifier1, device)
                p2 = generate_probability_maps(emb2, projection_head2, classifier2, device)

                # --- NUEVA OPERACIÓN: REFINAMIENTO DE INFILTRACIÓN ---
                # 1. Restamos el Tumor Core (p1[1]) a la Infiltración (p2[1]) y limitamos a 0.0
                p2_infil_refined = torch.clamp(p2[1] - p1[1], min=0.0)
                
                # 2. Calculamos la masa de probabilidad que fue removida por el solapamiento
                mass_removed = p2[1] - p2_infil_refined
                
                # 3. Sumamos esa masa a la clase "Resto/Fondo" (p2[0]) para que el mapa siga sumando 1.0
                p2_0_refined = p2[0] + mass_removed
                
                # 4. Reconstruimos el tensor de probabilidades del Pipeline 2 con la infiltración limpia
                p2_refined = torch.stack([p2_0_refined, p2_infil_refined, p2[2]], dim=0)
                # -----------------------------------------------------

                comb = torch.zeros_like(p1)
                comb[0] = p1[0]
                # Para la combinación de la Zona de Tratamiento, usamos la infiltración refinada
                comb[1] = torch.max(p1[1], p2_refined[1]) 
                comb[2] = p2_refined[2]
                
                # Normalización de seguridad
                comb = comb / (comb.sum(dim=0, keepdim=True) + 1e-6)
                
                # Apilamos los tres modelos usando p2_refined en lugar del p2 original
                stacked_maps = torch.cat([comb, p1, p2_refined], dim=0)
                return stacked_maps.unsqueeze(0)

            roi_size = (128, 128, 128)
            sw_batch_size = 1 
            # ... (el resto de tu código sliding_window_inference continúa igual)    
            # full_prob_maps ahora tiene tamaño [1, 9, 240, 240, 155]
            full_prob_maps = sliding_window_inference(
                inputs=image_tensor,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor_fn,
                overlap=0.25 
            )

            # Extraemos el volumen y lo dividimos
            full_prob_np = full_prob_maps.squeeze(0).cpu().numpy() # [9, H, W, D]
            
            prob_comb_np = full_prob_np[0:3]
            prob_p1_np   = full_prob_np[3:6]
            prob_p2_np   = full_prob_np[6:9]

            # --- SEGMENTACIONES DISCRETAS ---
            # 1. Combinado
            seg_comb = np.argmax(prob_comb_np, axis=0).astype(np.int16)
            seg_comb_orig = np.zeros_like(seg_comb, dtype=np.int16)
            seg_comb_orig[seg_comb == 1] = 6 # Zona de Tratamiento
            seg_comb_orig[seg_comb == 2] = 2 # Edema Vasogénico
            
            # 2. Pipeline 1 (0=Fondo, 1=Tumor Core, 2=Edema Total)
            seg_p1 = np.argmax(prob_p1_np, axis=0).astype(np.int16)
            
            # 3. Pipeline 2 (0=Resto, 1=Infiltración, 2=Edema Vasogénico)
            seg_p2 = np.argmax(prob_p2_np, axis=0).astype(np.int16)

            # Transponer todos para NIfTI [H, W, D, C]
            prob_comb_nifti = np.transpose(prob_comb_np, (1, 2, 3, 0))
            prob_p1_nifti   = np.transpose(prob_p1_np, (1, 2, 3, 0))
            prob_p2_nifti   = np.transpose(prob_p2_np, (1, 2, 3, 0))

            self.update_state(state='PROGRESS', meta={'message': 'Guardando múltiples volúmenes NIfTI...'})
            
            # Guardado Combinado
            nib.save(nib.Nifti1Image(prob_comb_nifti, affine), os.path.join(case_results_dir, f"prob_maps_{case_id}.nii.gz"))
            nib.save(nib.Nifti1Image(seg_comb_orig, affine), os.path.join(case_results_dir, f"segmentation_{case_id}.nii.gz"))
            
            # Guardado Pipeline 1
            nib.save(nib.Nifti1Image(prob_p1_nifti, affine), os.path.join(case_results_dir, f"prob_maps_p1_{case_id}.nii.gz"))
            nib.save(nib.Nifti1Image(seg_p1, affine), os.path.join(case_results_dir, f"segmentation_p1_{case_id}.nii.gz"))
            
            # Guardado Pipeline 2
            nib.save(nib.Nifti1Image(prob_p2_nifti, affine), os.path.join(case_results_dir, f"prob_maps_p2_{case_id}.nii.gz"))
            nib.save(nib.Nifti1Image(seg_p2, affine), os.path.join(case_results_dir, f"segmentation_p2_{case_id}.nii.gz"))

        return {"status": "success", "message": "Inferencia finalizada", "case_id": case_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()