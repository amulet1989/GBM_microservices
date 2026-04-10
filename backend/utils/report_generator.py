import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF

def generate_clinical_report(case_id: str, results_dir: str) -> str:
    case_results_dir = os.path.join(results_dir, case_id)
    
    # Rutas Combinadas
    seg_path = os.path.join(case_results_dir, f"segmentation_{case_id}.nii.gz")
    prob_path = os.path.join(case_results_dir, f"prob_maps_{case_id}.nii.gz")
    
    # Rutas Desglosadas
    seg_p1_path = os.path.join(case_results_dir, f"segmentation_p1_{case_id}.nii.gz")
    prob_p1_path = os.path.join(case_results_dir, f"prob_maps_p1_{case_id}.nii.gz")
    seg_p2_path = os.path.join(case_results_dir, f"segmentation_p2_{case_id}.nii.gz")
    prob_p2_path = os.path.join(case_results_dir, f"prob_maps_p2_{case_id}.nii.gz")
    
    if not os.path.exists(seg_path):
        raise FileNotFoundError("Los resultados no existen para este caso.")

    # 1. Cargar Volúmenes
    seg_data = nib.load(seg_path).get_fdata()
    prob_data = nib.load(prob_path).get_fdata()
    
    seg_p1_data = nib.load(seg_p1_path).get_fdata()
    prob_p1_data = nib.load(prob_p1_path).get_fdata()
    
    seg_p2_data = nib.load(seg_p2_path).get_fdata()
    prob_p2_data = nib.load(prob_p2_path).get_fdata()

    # 2. Cálculos Volumétricos
    zooms = nib.load(seg_path).header.get_zooms()
    voxel_volume_mm3 = np.prod(zooms[:3])
    
    vol_tratamiento_cm3 = (np.sum(seg_data == 6) * voxel_volume_mm3) / 1000.0
    
    # Desglose de P1 y P2
    vol_core_cm3 = (np.sum(seg_p1_data == 1) * voxel_volume_mm3) / 1000.0
    vol_infiltracion_cm3 = (np.sum(seg_p2_data == 1) * voxel_volume_mm3) / 1000.0

    # 3. Corte representativo
    slice_areas = np.sum((seg_data > 0).astype(int), axis=(0, 1))
    best_z = seg_data.shape[2] // 2 if np.sum(slice_areas) == 0 else np.argmax(slice_areas)

    # 4. Extraer cortes 2D
    seg_slice_rot = np.rot90(seg_data[:, :, best_z])
    prob_trat_rot = np.rot90(prob_data[:, :, best_z, 1])
    
    # Mapas específicos (Clase 1 de cada pipeline)
    prob_core_rot = np.rot90(prob_p1_data[:, :, best_z, 1])
    prob_infil_rot = np.rot90(prob_p2_data[:, :, best_z, 1])

    # 5. Generar Visualización (2 Filas x 2 Columnas)
    image_path = os.path.join(case_results_dir, f"snapshot_{case_id}.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12)) # Ajustado a 2x2
    fig.patch.set_facecolor('black')

    # Fila 1: Prob. Tumor Core y Prob. Infiltración
    im_core = axes[0,0].imshow(prob_core_rot, cmap='jet', vmin=0, vmax=1)
    axes[0,0].set_title("Prob. Tumor Core (P1)", color='white', pad=15, fontsize=14)
    axes[0,0].axis('off')

    axes[0,1].imshow(prob_infil_rot, cmap='jet', vmin=0, vmax=1)
    axes[0,1].set_title("Prob. Infiltración (P2)", color='white', pad=15, fontsize=14)
    axes[0,1].axis('off')

    # Fila 2: Segmentación Final y Prob. Zona Tratamiento
    cmap_seg = matplotlib.colors.ListedColormap(['black', 'green', 'red'])
    norm_seg = matplotlib.colors.BoundaryNorm([0, 1, 5, 7], cmap_seg.N)
    
    axes[1,0].imshow(seg_slice_rot, cmap=cmap_seg, norm=norm_seg)
    axes[1,0].set_title("Segmentación Final", color='white', pad=15, fontsize=14)
    axes[1,0].axis('off')

    im_trat = axes[1,1].imshow(prob_trat_rot, cmap='jet', vmin=0, vmax=1)
    axes[1,1].set_title("Prob. Zona Tratamiento", color='white', pad=15, fontsize=14)
    axes[1,1].axis('off')

    # Barra de color global (ajustada a la cuadrícula 2x2)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = fig.colorbar(im_trat, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.savefig(image_path, bbox_inches='tight', facecolor='black', dpi=150)
    plt.close()

    # 6. Construcción del PDF
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Reporte de Análisis Oncológico Multi-Pipeline", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"ID del Caso Clínico: {case_id}", ln=True)
    pdf.cell(0, 8, f"Corte Analizado (Plano Axial): Z = {best_z}", ln=True)
    pdf.ln(5)
    
    # KPIs Volumétricas
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Volumetría Detallada", ln=True)
    pdf.set_font("Arial", '', 11)
    
    # Ampliamos un poco la primera columna ya que no hay tantas filas
    pdf.cell(100, 8, "Región / Componente", border=1, align='C')
    pdf.cell(45, 8, "Volumen (cm3)", border=1, align='C', ln=True)
    
    # Desglose enfocado en la zona crítica
    pdf.cell(100, 8, "Tumor Core (Necrosis + Activo)", border=1)
    pdf.cell(45, 8, f"{vol_core_cm3:.2f} cm3", border=1, align='R', ln=True)
    
    pdf.cell(100, 8, "Infiltración Tumoral", border=1)
    pdf.cell(45, 8, f"{vol_infiltracion_cm3:.2f} cm3", border=1, align='R', ln=True)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(100, 8, "Zona de Tratamiento Total (Core + Infiltración)", border=1)
    pdf.cell(45, 8, f"{vol_tratamiento_cm3:.2f} cm3", border=1, align='R', ln=True)
    
    pdf.ln(10)
    
    # Gráficos
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Visualización Multipanel (Escala de Calor Jet)", ln=True)
    
    # Insertar la imagen (W más pequeño para compensar la forma cuadrada y centrarla)
    pdf.image(image_path, x=20, y=pdf.get_y(), w=170) 
    
    # Aseguramos espacio después de la imagen cuadrada
    pdf.set_y(pdf.get_y() + 160) 

    pdf.ln(5) 
    
    # Sección de Advertencia (Disclaimer)
    pdf.set_draw_color(200, 200, 200) 
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100) 
    advertencia = (
        "AVISO LEGAL: Este reporte ha sido generado de forma automatizada mediante modelos "
        "de segmentación por Inteligencia Artificial (SwinUNETR). Los resultados son de carácter "
        "experimental y para apoyo a la investigación. ESTE INFORME NO SUSTITUYE LA LECTURA "
        "DIRECTA DEL RADIÓLOGO NI LA DECISIÓN CLÍNICA FINAL DEL MÉDICO ESPECIALISTA."
    )
    pdf.multi_cell(0, 5, advertencia, align='J')

    # Generar Salida
    pdf_output_path = os.path.join(case_results_dir, f"reporte_clinico_{case_id}.pdf")
    pdf.output(pdf_output_path)
    
    return pdf_output_path