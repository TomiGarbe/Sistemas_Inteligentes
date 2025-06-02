import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
st.set_page_config(page_title="Detección de Personas y Chalecos", layout="wide", initial_sidebar_state="collapsed")

# -------------------------------------------------
# 1) ESTILOS CSS GLOBALES
# -------------------------------------------------
st.markdown("""
<style>
body, .stApp {
  background-color: #F0F0F0;
}    
                
.stTabs [data-baseweb="tab-list"] {
  display: flex;
  justify-content: center;
  margin-bottom: 1rem;
}

.stTabs [role="tab"] {
  background-color: rgba(0, 0, 0, 0.15);
  border-radius: 8px 8px 0 0;
  padding: 0.75rem 1.25rem;
  margin: 0 0.5rem;
  color: #222;
  font-weight: 600;
  transition: background-color 0.2s ease;
}
            
.stTabs [role="tab"]:hover {
  background-color: rgba(0, 0, 0, 0.25);
  color: #FFF;
}

.stTabs [role="tab"][aria-selected="true"] {
  background-color: #FFA500;
  color: #222;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
            
.stButton > button {
  width: 100%;
  padding: 0.75rem;
  background-color: rgba(0, 0, 0, 0.15);
  margin: 0 0.5rem;
  color: #222;
  font-weight: 600;
  transition: background-color 0.2s ease;
} 

.stButton > button:hover {
  background-color: rgba(0, 0, 0, 0.25);
  color: #FFF;
}
            
.stAlert div p {
  color: rgb(0, 0, 0);  /* aquí pones el color que quieras */
}
            
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 2) ENCABEZADO (TÍTULO + SUBTÍTULO)
# ------------------------------------------
st.markdown(
    "<div style='width:100%; height:10px;"
    "background: repeating-linear-gradient(45deg, #FFD700 0px, #FFD700 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; font-weight:800; color:#000000;'>PERSONAS Y CHALECOS DE SEGURIDAD</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#000000;'>Detección en Tiempo Real, en Imágenes y Videos</p>", unsafe_allow_html=True)

# Banda de seguridad (rayas amarillas y negras)
st.markdown(
    "<div style='width:100%; height:10px; margin-bottom: 20px; "
    "background: repeating-linear-gradient(45deg, #FFD700 0px, #FFD700 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)

# ------------------------------------------
# 3) DEFINICIÓN DE LAS PESTAÑAS
# ------------------------------------------
tab1, tab2, tab3 = st.tabs(["Análisis en Vivo", "Análisis de Imagen", "Análisis de Video"])

logo = Image.open("./assets/obrero.png")
with st.sidebar:
    st.image(logo, use_container_width =False, width=180)


# ------------------------------------------
# 4) TAB 1: CÁMARA EN VIVO
# ------------------------------------------
with tab1:
    st.markdown("<h3 style='text-align:left; color:#000000;'>Análisis en Vivo</h3>", unsafe_allow_html=True)

    st.write("") 
    st.write("") 
    # Inicializamos estado si no existe
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    # Tres columnas para centrar los botones
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Iniciar Cámara"):
                st.session_state.cam_on = True
        with col_b:
            if st.button("Detener Cámara"):
                st.session_state.cam_on = False

    u1, u2, u3 = st.columns([1, 2, 1])
    frame_placeholder = u2.empty()

    if st.session_state.cam_on:
        cap = cv2.VideoCapture(0)
        time.sleep(0.1)
        if not cap.isOpened():
            st.error("Error al abrir la cámara.")
            st.session_state.cam_on = False
        else:
            try:
                while st.session_state.cam_on:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error capturando frame.")
                        break
                    
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
                    time.sleep(0.05)
            except Exception:
                pass
            finally:
                cap.release()


# ------------------------------------------
# 5) TAB 2: ANALIZAR IMAGEN
# ------------------------------------------
with tab2:
    st.markdown("<h3 style='text-align:left; color:#000000;'>Análisis de Imagen Subida</h3>", unsafe_allow_html=True)

    cu1, cu2, cu3 = st.columns([1, 2, 1])
    with cu2:
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded:
        ci1, ci2, ci3 = st.columns([1, 2, 1])
        with ci2:
            st.image(uploaded, use_container_width=True)

        cb1, cb2, cb3 = st.columns([1, 2, 1])
        with cb2:
            if st.button("Analizar Imagen"):
                with st.spinner("Analizando imagen..."):
                    time.sleep(1)
                resultado = np.random.choice(["Chaleco Detectado", "Chaleco Ausente", "Sin Persona"])
                st.success(resultado)

# ------------------------------------------
# 6) TAB 2: ANALIZAR VIDEO
# ------------------------------------------
with tab3:
    st.markdown("<h3 style='text-align:left; color:#000000;'>Análisis de Video Subido</h3>", unsafe_allow_html=True)


    # Centrar el uploader en columnas [1,2,1]
    cu1, cu2, cu3 = st.columns([1, 2, 1])
    with cu2:
        uploaded_video = st.file_uploader( "", type=["mp4", "avi", "mov"]  )

    if uploaded_video:
        # Ahora centramos la reproducción del video, de nuevo en [1,2,1]
        v1, v2, v3 = st.columns([1, 2, 1])
        with v2:
            # st.video acepta directamente el archivo subido
            st.video(uploaded_video, format="video/mp4", start_time=0)
