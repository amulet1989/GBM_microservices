import streamlit as st
import requests
import time

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="MRI Inferencia Pipeline", layout="wide")

MODALITY_ORDER = {
    "images_DSC": ["DSC_ap-rCBV", "DSC_PH", "DSC_PSR"],
    "images_DTI": ["DTI_AD", "DTI_FA", "DTI_RD", "DTI_TR"],
    "images_structural": ["FLAIR", "T1.", "T1GD", "T2"],
}

# ---------------------------------------------------------
# BARRA LATERAL: GESTIÓN DE CASOS
# ---------------------------------------------------------
st.sidebar.title("🗂️ Gestión de Casos")

# Obtener lista de casos del backend
def get_cases():
    try:
        res = requests.get(f"{BACKEND_URL}/cases/")
        if res.status_code == 200:
            return res.json().get("cases", [])
    except:
        return []
    return []

cases = get_cases()

# Inicializar el estado si no existe
if "active_case" not in st.session_state:
    st.session_state.active_case = ""

# 1. Crear un nuevo caso
new_case_id = st.sidebar.text_input("Crear Nuevo Caso (ID):", placeholder="Ej: UPENN-001")
if st.sidebar.button("➕ Iniciar Nuevo Caso"):
    if new_case_id:
        st.session_state.active_case = new_case_id
        st.rerun() # Recarga la app para aplicar el cambio

# 2. Forzar que el caso nuevo aparezca en la lista aunque la carpeta aún no exista en el backend
if st.session_state.active_case and st.session_state.active_case not in cases:
    cases.append(st.session_state.active_case)

# 3. Seleccionar caso activo (menú desplegable)
options = [""] + sorted(cases)
# Determinar el índice del caso activo actual
try:
    current_index = options.index(st.session_state.active_case)
except ValueError:
    current_index = 0

selected_case = st.sidebar.selectbox("Seleccionar Caso Activo:", options, index=current_index)

# Si el usuario cambia el selector manualmente, actualizamos el estado y recargamos
if selected_case != st.session_state.active_case:
    st.session_state.active_case = selected_case
    st.rerun()

active_case = st.session_state.active_case

# 4. Eliminar caso activo
if active_case:
    st.sidebar.divider()
    if st.sidebar.button(f"🗑️ Eliminar Caso {active_case}", type="primary"):
        res = requests.delete(f"{BACKEND_URL}/cases/{active_case}")
        # Lo eliminamos independientemente de si el backend dio error (ej. carpeta no existía)
        st.session_state.active_case = ""
        st.sidebar.success("Caso eliminado del espacio de trabajo.")
        st.rerun()

# Validar que haya un caso seleccionado antes de mostrar el resto de la app
if not active_case:
    st.title("Carga de Imágenes Médicas (MRI) 🧠")
    st.info("👈 Por favor, crea un nuevo caso o selecciona uno en la barra lateral para comenzar.")
    st.stop()

# ---------------------------------------------------------
# ÁREA PRINCIPAL: CARGA Y VERIFICACIÓN
# ---------------------------------------------------------
st.title(f"Paciente / Caso: `{active_case}`")
st.markdown("Sube los volúmenes para inferencia. Los archivos se guardarán aislados bajo este ID.")

tab1, tab2 = st.tabs(["⚡ Carga Inteligente (Masiva)", "🛠️ Carga Manual (Por Modalidad)"])

with tab1:
    st.header("Carga Masiva")
    bulk_files = st.file_uploader("Arrastra aquí todos los archivos del caso:", accept_multiple_files=True, key="bulk_uploader")
    
    if st.button("Subir y Organizar", type="primary", key="btn_bulk"):
        if bulk_files:
            files_to_send = [("files", (file.name, file.getvalue(), file.type)) for file in bulk_files]
            with st.spinner("Procesando..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/upload_bulk/{active_case}", files=files_to_send)
                    if res.status_code == 200:
                        st.success("¡Archivos subidos con éxito!")
                    else:
                        st.error("Error del servidor.")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.header("Carga Manual")
    for category, modalities in MODALITY_ORDER.items():
        st.subheader(f"{category}")
        cols = st.columns(len(modalities))
        for i, modality in enumerate(modalities):
            with cols[i]:
                uploaded_files = st.file_uploader(f"{modality}", accept_multiple_files=True, key=f"up_{modality}")
                if st.button(f"Subir {modality}", key=f"btn_{modality}"):
                    if uploaded_files:
                        files_to_send = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                        data = {"category": category, "modality": modality}
                        res = requests.post(f"{BACKEND_URL}/upload/{active_case}", data=data, files=files_to_send)
                        if res.status_code == 200:
                            st.success(f"Subido {modality}")

# ---------------------------------------------------------
# VERIFICACIÓN DE ESTADO Y ACCIÓN FINAL
# ---------------------------------------------------------
st.divider()
st.header("📋 Verificación de Archivos Cargados")
st.markdown("Revisa el estado de las modalidades. Puedes eliminar modalidades individuales si te equivocaste.")

# Obtener y renderizar el estado automáticamente
try:
    res = requests.get(f"{BACKEND_URL}/status/{active_case}")
    if res.status_code == 200:
        status_data = res.json()["status"]
        cols = st.columns(len(status_data))
        
        for idx, (category, modalities) in enumerate(status_data.items()):
            with cols[idx]:
                st.subheader(category)
                for mod in modalities:
                    mod_name = mod["modality"]
                    if mod["status"] == "Presente":
                        # Mostrar los archivos presentes
                        st.success(f"✅ **{mod_name}**\n\n`{', '.join(mod['files'])}`")
                        
                        # Botón para eliminar esta modalidad específica
                        if st.button(f"🗑️ Eliminar {mod_name}", key=f"del_{mod_name}_{active_case}"):
                            del_res = requests.delete(f"{BACKEND_URL}/cases/{active_case}/{category}/{mod_name}")
                            if del_res.status_code == 200:
                                st.rerun() # Recargar la interfaz al borrar
                            else:
                                st.error("No se pudo eliminar.")
                    else:
                        st.warning(f"⚠️ **{mod_name}** (Faltante)")
except Exception as e:
    st.error(f"Error al obtener el estado: {e}")

st.divider()
# Switch para decidir el flujo
needs_preprocessing = st.checkbox("⚙️ Ejecutar Preprocesamiento BraTS (Conversión DICOM, N4, Co-registro, Skull Stripping)", value=True, help="Desmarca esta opción si tus NIfTI ya pasaron por el pipeline de estandarización.")

if st.button(f"🚀 Iniciar Inferencia para {active_case}", type="secondary"):
    try:

        # 1. Enviar la petición para iniciar la tarea
        url = f"{BACKEND_URL}/process/{active_case}?run_preprocessing={str(needs_preprocessing).lower()}"
        res = requests.post(url)
        if res.status_code == 200:
            task_id = res.json()["task_id"]
            
            # Crear contenedores visuales dinámicos
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # 2. Polling (Consultar estado cada 2 segundos)
            with st.spinner("Conectando con el motor de inferencia..."):
                while True:
                    time.sleep(2)
                    task_res = requests.get(f"{BACKEND_URL}/task/{task_id}")
                    if task_res.status_code == 200:
                        task_data = task_res.json()
                        status = task_data["status"]
                        
                        if status == "PENDING":
                            status_text.info("⏳ En cola, esperando un Worker disponible...")
                        elif status == "PROGRESS":
                            msg = task_data.get("message", "Procesando...")
                            status_text.warning(f"⚙️ Procesando: {msg}")
                            # Simulamos avance visual
                            progress_bar.progress(50) 
                        elif status == "SUCCESS":
                            progress_bar.progress(100)
                            status_text.success("🎉 ¡Inferencia completada con éxito!")
                            st.balloons()
                            st.info("Se han generado los mapas de probabilidad y la segmentación.")
                            break
                        elif status == "FAILURE":
                            status_text.error(f"❌ Fallo en la inferencia: {task_data.get('error')}")
                            break
        else:
            st.error(f"Error al iniciar la tarea: {res.text}")
    except Exception as e:
        st.error(f"Error de conexión: {e}")

# ---------------------------------------------------------
# DESCARGA DE RESULTADOS
# ---------------------------------------------------------
st.divider()
st.header("📥 Resultados de Inferencia")

# Usamos caché para no descargar el mismo archivo del backend múltiples veces si recargamos la página
@st.cache_data(show_spinner=False)
def fetch_result_file(case_id, file_type):
    url = f"{BACKEND_URL}/download/{case_id}/{file_type}"
    res = requests.get(url)
    if res.status_code == 200:
        return res.content
    return None

try:
    # Preguntar al backend si hay resultados listos
    res_status = requests.get(f"{BACKEND_URL}/results/{active_case}")
    
    if res_status.status_code == 200:
        results_data = res_status.json()
        
        if results_data["segmentation"] or results_data["prob_maps"]:
            st.success(f"✅ Resultados listos y empaquetados para el paciente: `{active_case}`")
            cols = st.columns(2)
            
            # Tarjeta de Segmentación
            if results_data["segmentation"]:
                with cols[0]:
                    st.markdown("#### 🧠 Volumen de Segmentación")
                    st.info("Contiene las etiquetas de Infiltración Tumoral (2) y Edema Vasogénico (6).")
                    with st.spinner("Preparando archivo..."):
                        seg_data = fetch_result_file(active_case, "segmentation")
                        if seg_data:
                            st.download_button(
                                label="⬇️ Descargar Segmentación (.nii.gz)",
                                data=seg_data,
                                file_name=f"segmentation_{active_case}.nii.gz",
                                mime="application/gzip",
                                type="primary"
                            )
                            
            # Tarjeta de Mapas de Probabilidad
            if results_data["prob_maps"]:
                with cols[1]:
                    st.markdown("#### 📊 Mapas de Probabilidad")
                    st.info("Tensor en formato Float32 con las probabilidades brutas de cada vóxel.")
                    with st.spinner("Preparando archivo..."):
                        prob_data = fetch_result_file(active_case, "prob_maps")
                        if prob_data:
                            st.download_button(
                                label="⬇️ Descargar Mapas (.nii.gz)",
                                data=prob_data,
                                file_name=f"prob_maps_{active_case}.nii.gz",
                                mime="application/gzip"
                            )
        else:
            st.info("Aún no hay resultados de inferencia disponibles para este caso.")
except Exception as e:
    st.error(f"Error al verificar la disponibilidad de los resultados: {e}")