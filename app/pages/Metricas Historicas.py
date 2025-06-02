import streamlit as st
from PIL import Image
import os

st.title("Métricas Generadas por YOLO")

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


# ------------------------------------------------------------
# Ajusta esta ruta a donde YOLO guardó tus archivos PNG:
# ------------------------------------------------------------
METRICS_DIR = "../runs/detect/train"

# Lista de PNGs a mostrar (solo aquellas que resumen resultados)
png_files = [
    ("Resumen general", "results.png"),
    ("Matriz de Confusión (sin normalizar)", "confusion_matrix.png"),
    ("Matriz de Confusión (normalizada)", "confusion_matrix_normalized.png"),
    ("Curva Precision-Recall", "PR_curve.png"),
    ("Curva Precision vs Umbral", "P_curve.png"),
    ("Curva Recall vs Umbral", "R_curve.png"),
    ("Curva F1 vs Umbral", "F1_curve.png")
]

for title, filename in png_files:
    full_path = os.path.join(METRICS_DIR, filename)
    if os.path.exists(full_path):
        st.subheader(title)
        img = Image.open(full_path)
        st.image(img, use_container_width =True)
        st.markdown("---")
    else:
        st.warning(f"No se encontró `{filename}` en `{METRICS_DIR}`")