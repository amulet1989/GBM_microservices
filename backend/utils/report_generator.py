import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF

def generate_clinical_report(case_id: str, results_dir: str) -> str:
    """Genera un informe PDF con métricas volumétricas y vistas multipanel."""
    
    case_results_dir = os.path.join(results_dir, case_id)
    seg_path = os.path.join(case_results_dir, f"segmentation_{case_id}.nii.gz")
    prob_path = os.path.join(case_results_dir, f"prob_maps_{case_id}.nii.gz")
    
    if not os.path.exists(seg_path) or not os.path.exists(prob_path):
        raise FileNotFoundError("Los resultados (segmentación o mapas) no existen para este caso.")

    # 1. Cargar Volúmenes NIfTI
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    
    prob_img = nib.load(prob_path)
    prob_data = prob_img.get_fdata() # [H, W, D, 3]

    # 2. Cálculos Volumétricos
    zooms = seg_img.header.get_zooms()
    voxel_volume_mm3 = np.prod(zooms[:3])
    
    # CORRECCIÓN DE CLASES: 
    # Clase 6 = Zona de Tratamiento (Tumor Core + Infiltración)
    # Clase 2 = Edema Vasogénico Puro
    voxels_tratamiento = np.sum(seg_data == 6)
    voxels_vasogenico = np.sum(seg_data == 2)
    
    vol_tratamiento_cm3 = (voxels_tratamiento * voxel_volume_mm3) / 1000.0
    vol_vasogenico_cm3 = (voxels_vasogenico * voxel_volume_mm3) / 1000.0
    vol_total_cm3 = vol_tratamiento_cm3 + vol_vasogenico_cm3

    # 3. Búsqueda del corte más representativo
    tumor_mask = (seg_data > 0).astype(int)
    slice_areas = np.sum(tumor_mask, axis=(0, 1))
    
    if np.sum(slice_areas) == 0:
        best_z = seg_data.shape[2] // 2
    else:
        best_z = np.argmax(slice_areas)

    # Extraer cortes 2D rotados
    seg_slice_rot = np.rot90(seg_data[:, :, best_z])
    
    # CORRECCIÓN DE CANALES DE PROBABILIDAD:
    # Canal 1 = Zona de Tratamiento
    # Canal 2 = Edema Vasogénico Puro
    prob_trat_rot = np.rot90(prob_data[:, :, best_z, 1])
    prob_vaso_rot = np.rot90(prob_data[:, :, best_z, 2])

    # 4. Generar la visualización Multipanel
    image_path = os.path.join(case_results_dir, f"snapshot_{case_id}.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('black')

    # Panel A: Segmentación Discreta
    # Norm: [0 a 1) -> Black, [1 a 5) -> Green (Clase 2), [5 a 7) -> Red (Clase 6)
    cmap_seg = matplotlib.colors.ListedColormap(['black', 'green', 'red'])
    norm_seg = matplotlib.colors.BoundaryNorm([0, 1, 5, 7], cmap_seg.N)
    
    axes[0].imshow(seg_slice_rot, cmap=cmap_seg, norm=norm_seg)
    axes[0].set_title("Segmentación del Modelo", color='white', pad=15, fontsize=14)
    axes[0].axis('off')

    # Panel B: Mapa de Probabilidad - Zona de Tratamiento
    im_trat = axes[1].imshow(prob_trat_rot, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Probabilidad: Zona de Tratamiento", color='white', pad=15, fontsize=14)
    axes[1].axis('off')

    # Panel C: Mapa de Probabilidad - Edema Vasogénico Puro
    im_vaso = axes[2].imshow(prob_vaso_rot, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Prob. Edema Vasogénico Puro", color='white', pad=15, fontsize=14)
    axes[2].axis('off')

    # Barra de color global
    cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.6]) 
    cbar = fig.colorbar(im_trat, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.savefig(image_path, bbox_inches='tight', facecolor='black', dpi=150)
    plt.close()

    # 5. Construcción del PDF
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Reporte de Segmentación Volumétrica y Probabilística", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"ID del Caso Clínico: {case_id}", ln=True)
    pdf.cell(0, 8, f"Corte Analizado (Plano Axial): Z = {best_z}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Métricas de Volumetría Tumoral", ln=True)
    pdf.set_font("Arial", '', 12)
    
    pdf.cell(85, 10, "Región de Interés (ROI)", border=1, align='C')
    pdf.cell(50, 10, "Volumen Calculado", border=1, align='C', ln=True)
    
    # Textos actualizados con las clases correctas
    pdf.cell(85, 10, "Zona de Tratamiento (Clase 6)", border=1)
    pdf.cell(50, 10, f"{vol_tratamiento_cm3:.2f} cm3", border=1, align='R', ln=True)
    
    pdf.cell(85, 10, "Edema Vasogenico Puro (Clase 2)", border=1)
    pdf.cell(50, 10, f"{vol_vasogenico_cm3:.2f} cm3", border=1, align='R', ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(85, 10, "Volumen Anormal Total", border=1)
    pdf.cell(50, 10, f"{vol_total_cm3:.2f} cm3", border=1, align='R', ln=True)
    
    pdf.ln(8)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Visualización de Segmentación y Mapas de Probabilidad", ln=True)
    pdf.ln(2)
    
    pdf.image(image_path, x=10, w=190) 
    
    pdf.ln(80)
    
    pdf.set_font("Arial", 'I', 9)
    disclaimer = ("Nota Clínica: La 'Zona de Tratamiento' incluye el núcleo tumoral activo, tejido necrótico "
                  "e infiltración circundante. Este reporte automatizado ha sido generado por redes neuronales "
                  "profundas (SwinUNETR) y no sustituye la lectura directa del radiólogo ni la decisión clínica final.")
    pdf.multi_cell(0, 5, disclaimer)
    
    pdf_output_path = os.path.join(case_results_dir, f"reporte_clinico_{case_id}.pdf")
    pdf.output(pdf_output_path)
    
    return pdf_output_path