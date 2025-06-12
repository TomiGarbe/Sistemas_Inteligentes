import streamlit as st
from PIL import Image
import os
st.set_page_config(page_title="Detección de Personas y Chalecos", layout="wide")

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
  background-color: #FEBA28;
  color: #222;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
            
h1, h2, h3, h4, h5, h6, p {
  color: #000000 !important;
}

section[data-testid="stSidebar"] {
    background: #0e1117;
    backdrop-filter: blur(2px);
} 
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# ENCABEZADO (TÍTULO + SUBTÍTULO)
# ------------------------------------------
st.markdown(
    "<div style='width:100%; height:10px;"
    "background: repeating-linear-gradient(45deg, #FEBA28 0px, #FEBA28 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align:center; font-weight:800;'>PERSONAS Y CHALECOS DE SEGURIDAD</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Métricas Generadas por YOLO</h4>", unsafe_allow_html=True)
st.markdown(
    "<div style='width:100%; height:10px; margin-bottom:20px; "
    "background: repeating-linear-gradient(45deg, #FEBA28 0px, #FEBA28 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)

logo = Image.open("./assets/obrero.png")
with st.sidebar:
    st.image(logo, use_container_width =False, width=180)

MAX_HEIGHT = 400  # altura máxima (px) para todas las imágenes

# ----------- NUEVO: Pestañas de métricas ---------
tab1, tab2, tab3 = st.tabs(["Métricas de Entrenamiento", "Métricas de Test", "Comparación"])

# ----------- MÉTRICAS DE ENTRENAMIENTO -----------
with tab1:
    st.header("Métricas de Entrenamiento")
    TRAIN_DIR = "../runs/detect/train3"
    train_files = [
        ("Resumen general", "results.png"),
        # ("Matriz de Confusión (sin normalizar)", "confusion_matrix.png"),
        ("Matriz de Confusión (normalizada)", "confusion_matrix_normalized.png"),
        ("Curva de Precision", "P_curve.png"),
        ("Curva de Recall", "R_curve.png"),
        ("Curva de Precision-Recall", "PR_curve.png"),
        ("Curva de F1 Score", "F1_curve.png"),
    ]

    train_available = []
    for title, fname in train_files:
        p = os.path.join(TRAIN_DIR, fname)
        if os.path.exists(p):
            train_available.append((title, p))
        else:
            st.warning(f"No hallé `{fname}` en `{TRAIN_DIR}`")

    for i in range(0, len(train_available), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, train_available[i : i + 2]):
            with col:
                st.subheader(title)
                img = Image.open(path)
                st.image(img)
        st.markdown("---")

# ----------- MÉTRICAS DE TEST -----------
with tab2:
    st.header("Métricas de Test")
    TEST_DIR = "../runs/detect/test_eval"

    test_raw = [
        # ("Matriz de Confusión (sin normalizar)", "confusion_matrix.png"),
        ("Matriz de Confusión (normalizada)", "confusion_matrix_normalized.png"),
        ("Curva de Precision", "P_curve.png"),
        ("Curva de Recall", "R_curve.png"),
        ("Curva de Precision-Recall", "PR_curve.png"),
        ("Curva de F1 Score", "F1_curve.png"),
    ]
    test_available = []
    for title, fname in test_raw:
        p = os.path.join(TEST_DIR, fname)
        if os.path.exists(p):
            test_available.append((title, p))
        else:
            st.warning(f"No hallé `{fname}` en `{TEST_DIR}`")
    for i in range(0, len(test_available), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, test_available[i : i + 2]):
            with col:
                st.subheader(title)
                img = Image.open(path)
                st.image(img)
        st.markdown("---")

# ----------- COMPARACIÓN VALIDACIÓN VS TEST -----------
with tab3:
    st.header("Comparación entre Validación y Test")
    TEST_VALID_DIR = "../runs/detect/test_eval/metricas"
    comp_files = [
        ("F1: Val vs Test", "F1_val_vs_test.png"),
        ("Precision: Val vs Test", "Precision_val_vs_test.png"),
        ("Recall: Val vs Test", "Recall_val_vs_test.png"),
        ("mAP50: Val vs Test", "mAP50_val_vs_test.png"),
        ("mAP50-95: Val vs Test", "mAP50-95_val_vs_test.png"),
    ]
    comp_available = []
    for title, fname in comp_files:
        p = os.path.join(TEST_VALID_DIR, fname)
        if os.path.exists(p):
            comp_available.append((title, p))
        else:
            st.warning(f"Comparación: No hallé `{fname}` en `{TEST_VALID_DIR}`")
    for i in range(0, len(comp_available), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, comp_available[i : i + 2]):
            with col:
                st.subheader(title)
                img = Image.open(path)
                st.image(img)
        st.markdown("---")
