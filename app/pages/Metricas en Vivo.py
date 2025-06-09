import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

st.set_page_config(page_title="Detección de Personas y Chalecos", layout="wide")

st.markdown("""
<style>
body, .stApp {
  background-color: #F0F0F0;
}
h1, h2, h3, h4, h5, h6, [data-testid="stMetricValue"] {
  color: #000000 !important;
}
            
div[data-testid="stMetric"], p  {
    color: #000000 !important;
    font-size: 1.15rem !important;
    line-height: 1.2 !important;
}
            
section[data-testid="stSidebar"] {
    background: #0e1117;  /* Cambiá el alpha a tu gusto */
    backdrop-filter: blur(2px);
} 
            
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='width:100%; height:10px;"
    "background: repeating-linear-gradient(45deg, #FEBA28 0px, #FEBA28 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align:center; font-weight:800; color:#000000;'>PERSONAS Y CHALECOS DE SEGURIDAD</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Métricas Generadas por Matplotlib en vivo</h4>", unsafe_allow_html=True)
st.markdown(
    "<div style='width:100%; height:10px; margin-bottom:20px; "
    "background: repeating-linear-gradient(45deg, #FEBA28 0px, #FEBA28 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)

try:
    logo = Image.open("./assets/obrero.png")
    with st.sidebar:
        st.image(logo, use_container_width=False, width=180)
except Exception:
    pass

# ------------- DATOS DE SESSION_STATE -------------
personas_x_frame = st.session_state.get("personas_x_frame", [])
chalecos_x_frame = st.session_state.get("chalecos_x_frame", [])
latencias = st.session_state.get("latencias", [])
frame_timestamps = st.session_state.get("frame_timestamps", [])

# --- Filtrar solo frames de los últimos 60 segundos ---
if frame_timestamps:
    ahora = time.time()
    idx_ultimos_60s = [i for i, t in enumerate(frame_timestamps) if ahora - t <= 60]

    # Función para acceder seguro (cero si falta)
    def safe_get(lista, i):
        try:
            return int(lista[i])
        except (IndexError, TypeError):
            return 0

    personas_60s = [safe_get(personas_x_frame, i) for i in idx_ultimos_60s]
    chalecos_60s = [safe_get(chalecos_x_frame, i) for i in idx_ultimos_60s]
    latencias_60s = [safe_get(latencias, i) for i in idx_ultimos_60s]

    if personas_60s:
        frames_analizados = len(personas_60s)
        max_personas = int(np.max(personas_60s))
        max_chalecos = int(np.max(chalecos_60s))
        fps_promedio = 1000 / np.mean(latencias_60s) if latencias_60s and np.mean(latencias_60s) > 0 else 0
        latencia_promedio = np.mean(latencias_60s) if latencias_60s else 0

        st.subheader("Resumen de los últimos 60 segundos de camara en vivo y video")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Máx. personas en un frame", max_personas)
        c2.metric("Máx. chalecos en un frame", max_chalecos)
        c3.metric("FPS promedio", f"{fps_promedio:.2f}")
        c4.metric("Latencia promedio (ms por frame)", f"{latencia_promedio:.1f}")

        st.markdown("---")

        # --------- GRAFICOS EN DOS COLUMNAS, MONTAÑA/ÁREA ---------
        gcol1, gcol2 = st.columns(2)

        with gcol1:
            st.markdown("**Personas detectadas por frame**")
            fig1, ax1 = plt.subplots(figsize=(5, 2.6))
            y_personas = personas_60s
            x_vals = range(1, len(y_personas) + 1)
            ax1.fill_between(x_vals, y_personas, step="mid", color="#2274A5", alpha=0.45)
            ax1.plot(x_vals, y_personas, linestyle='-', color="#2274A5", linewidth=2)  # ← sin marker
            ax1.set_xlabel("Frame (últimos 60s)")
            ax1.set_ylabel("Cantidad de personas")
            ax1.set_yticks(range(0, max(y_personas) + 2))
            ax1.set_title("Personas por frame", fontsize=11)
            st.pyplot(fig1, use_container_width=True)
        
        with gcol2:
            st.markdown("**Chalecos detectados por frame**")
            fig2, ax2 = plt.subplots(figsize=(5, 2.6))
            y_chalecos = chalecos_60s
            x_vals2 = range(1, len(y_chalecos) + 1)
            ax2.fill_between(x_vals2, y_chalecos, step="mid", color="#30c96b", alpha=0.45)
            ax2.plot(x_vals2, y_chalecos, linestyle='-', color="#30c96b", linewidth=2)  # ← sin marker
            ax2.set_xlabel("Frame (últimos 60s)")
            ax2.set_ylabel("Cantidad de chalecos")
            ax2.set_yticks(range(0, max(y_chalecos) + 2))
            ax2.set_title("Chalecos por frame", fontsize=11)
            st.pyplot(fig2, use_container_width=True)


    else:
        st.info("No hay datos de detección en los últimos 60 segundos.")
else:
    st.info("Aún no hay datos en vivo para graficar. Iniciá el reconocimiento desde la página principal para ver las métricas aquí.")
