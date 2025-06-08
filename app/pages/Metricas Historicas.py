import streamlit as st
from PIL import Image
import os

# -------------------------------------------------
# 1) ESTILOS CSS GLOBALES
# -------------------------------------------------
st.markdown("""
<style>
body, .stApp {
  background-color: #F0F0F0;
}    

h1, h2, h3, h4, h5, h6, p, span, div {
  color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# ENCABEZADO (TÍTULO + SUBTÍTULO)
# ------------------------------------------
st.markdown(
    "<div style='width:100%; height:10px;"
    "background: repeating-linear-gradient(45deg, #FFD700 0px, #FFD700 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align:center; font-weight:800;'>PERSONAS Y CHALECOS DE SEGURIDAD</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Métricas Generadas por YOLO</h4>", unsafe_allow_html=True)
st.markdown(
    "<div style='width:100%; height:10px; margin-bottom:20px; "
    "background: repeating-linear-gradient(45deg, #FFD700 0px, #FFD700 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Parámetros generales
# ------------------------------------------------------------
MAX_HEIGHT = 400  # altura máxima (px) para todas las imágenes

# ------------------------------------------------------------
# 2) MÉTRICAS DE ENTRENAMIENTO
# ------------------------------------------------------------
st.header("📈 Métricas de Entrenamiento")
TRAIN_DIR = "../runs/detect/train3"
train_files = [
    ("Resumen general", "results.png"),
    ("Matriz de Confusión (sin normalizar)", "confusion_matrix.png"),
    ("Matriz de Confusión (normalizada)", "confusion_matrix_normalized.png"),
    ("Curva Precision-Recall", "PR_curve.png"),
    ("Curva Precision vs Umbral", "P_curve.png"),
    ("Curva Recall vs Umbral", "R_curve.png"),
    ("Curva F1 vs Umbral", "F1_curve.png"),
]

# Filtrar los existentes
train_available = []
for title, fname in train_files:
    p = os.path.join(TRAIN_DIR, fname)
    if os.path.exists(p):
        train_available.append((title, p))
    else:
        st.warning(f"Entrenamiento ▶ No hallé `{fname}` en `{TRAIN_DIR}`")

# Mostrar en dos columnas
for i in range(0, len(train_available), 2):
    cols = st.columns(2)
    for col, (title, path) in zip(cols, train_available[i : i + 2]):
        with col:
            st.subheader(title)
            img = Image.open(path)
            w, h = img.size
            if h > MAX_HEIGHT:
                new_w = int(w * MAX_HEIGHT / h)
                img = img.resize((new_w, MAX_HEIGHT), resample=Image.LANCZOS)
            st.image(img)
    st.markdown("---")


# ------------------------------------------------------------
# 3) MÉTRICAS DE TEST
# ------------------------------------------------------------
st.header("🧪 Métricas de Test")
TEST_DIR = "../runs/detect/test_eval"
TEST_VALID_DIR = "../runs/detect/test_eval/metricas"

# 3.1 Resultados sobre Test
st.subheader("Resultados en Test")
test_raw = [
    ("Matriz de Confusión (sin normalizar)", "confusion_matrix.png"),
    ("Matriz de Confusión (normalizada)", "confusion_matrix_normalized.png"),
    ("Curva Precision-Recall", "PR_curve.png"),
    ("Curva Precision vs Umbral", "P_curve.png"),
    ("Curva Recall vs Umbral", "R_curve.png"),
    ("Curva F1 vs Umbral", "F1_curve.png"),
]
test_available = []
for title, fname in test_raw:
    p = os.path.join(TEST_DIR, fname)
    if os.path.exists(p):
        test_available.append((title, p))
    else:
        st.warning(f"Test ▶ No hallé `{fname}` en `{TEST_DIR}`")

for i in range(0, len(test_available), 2):
    cols = st.columns(2)
    for col, (title, path) in zip(cols, test_available[i : i + 2]):
        with col:
            st.subheader(title)
            img = Image.open(path)
            w, h = img.size
            if h > MAX_HEIGHT:
                new_w = int(w * MAX_HEIGHT / h)
                img = img.resize((new_w, MAX_HEIGHT), resample=Image.LANCZOS)
            st.image(img)
    st.markdown("---")

# 3.2 Comparación Validación vs Test
st.subheader("Comparación: Validación vs Test")
comp_files = [
    ("F1: Val vs Test", "F1_val_vs_test.png"),
    ("Precision: Val vs Test", "Precision_val_vs_test.png"),
    ("Recall: Val vs Test", "Recall_val_vs_test.png"),
    ("mAP50: Val vs Test", "mAP50_val_vs_test.png"),
    ("mAP50–95: Val vs Test", "mAP50-95_val_vs_test.png"),
]
comp_available = []
for title, fname in comp_files:
    p = os.path.join(TEST_VALID_DIR, fname)
    if os.path.exists(p):
        comp_available.append((title, p))
    else:
        st.warning(f"Comparación ▶ No hallé `{fname}` en `{TEST_DIR}`")

for i in range(0, len(comp_available), 2):
    cols = st.columns(2)
    for col, (title, path) in zip(cols, comp_available[i : i + 2]):
        with col:
            st.subheader(title)
            img = Image.open(path)
            w, h = img.size
            if h > MAX_HEIGHT:
                new_w = int(w * MAX_HEIGHT / h)
                img = img.resize((new_w, MAX_HEIGHT), resample=Image.LANCZOS)
            st.image(img)
    st.markdown("---")
