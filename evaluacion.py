# evaluacion_test_solometrics.py
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ----------------------------------------
# 1) Configuración de rutas y parámetros
# ----------------------------------------

# Ruta a tu modelo entrenado (best.pt)
MODEL_PATH = "./runs/detect/train2/weights/best.pt"

# Archivo YAML que contenga al menos la clave `test: ./dataset/test/images`
DATA_YAML = "./config.yaml"

# Tamaño de imagen para la validación (igual al que usaste en el entrenamiento)
IMG_SIZE = 640

# Batch size para la evaluación; si usas solo CPU, bájalo a 4 u 8
BATCH_SIZE = 8

# Dispositivo: "cpu" o "cuda:0" si dispones de GPU Nvidia
DEVICE = "cpu"

# Carpeta donde guardaremos los gráficos de cada métrica de TEST
output_dir = "metricas_test"
os.makedirs(output_dir, exist_ok=True)


# ----------------------------------------
# 2) Verificar que existan los archivos
# ----------------------------------------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"No encontré el modelo en: {MODEL_PATH}")
if not os.path.isfile(DATA_YAML):
    raise FileNotFoundError(f"No encontré el data.yaml en: {DATA_YAML}")


# ----------------------------------------
# 3) Cargar el modelo YOLOv8 entrenado
# ----------------------------------------
model = YOLO(MODEL_PATH)
print(f"✔️  Modelo cargado desde: {MODEL_PATH}\n")


# ----------------------------------------
# 4) Ejecutar evaluación sobre TEST
# ----------------------------------------
print("🔍 Ejecutando evaluación en TEST (solo sobre dataset/test)...\n")
results_test = model.val(
    data=DATA_YAML,
    split="test",     # Usa únicamente la sección `test:` de tu config.yaml
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE
)

# Extraer métricas globales de results_test.box:
#  • mp: Precision promedio sobre clases (@IoU=0.5)
#  • mr: Recall promedio sobre clases (@IoU=0.5)
#  • map50: mAP @ IoU=0.50
#  • map:   mAP @ IoU=0.50:0.95
#  • f1 (list): F1 por clase. Tomamos la media para obtener F1 global.
p50_test_global   = float(results_test.box.mp)
r50_test_global   = float(results_test.box.mr)
map50_test_global = float(results_test.box.map50)
map95_test_global = float(results_test.box.map)
f1_per_class      = results_test.box.f1  # lista de F1 por clase
if len(f1_per_class) > 0:
    f1_test_global = float(sum(f1_per_class) / len(f1_per_class))
else:
    f1_test_global = 0.0  # en caso de no haber clases

print(f"Precision @0.5  (TEST): {p50_test_global:.4f}")
print(f"Recall    @0.5  (TEST): {r50_test_global:.4f}")
print(f"mAP   @0.50     (TEST): {map50_test_global:.4f}")
print(f"mAP @0.50:0.95  (TEST): {map95_test_global:.4f}")
print(f"F1 score        (TEST): {f1_test_global:.4f}\n")


# ----------------------------------------
# 5) Preparar datos para generar cada gráfico
# ----------------------------------------
metric_names = ["Precision@0.5", "Recall@0.5", "mAP@0.50", "mAP@0.50:0.95", "F1"]
values_test = [
    p50_test_global,
    r50_test_global,
    map50_test_global,
    map95_test_global,
    f1_test_global
]

# ----------------------------------------
# 6) Generar y guardar un gráfico por cada métrica (solo TEST)
# ----------------------------------------
for i, name in enumerate(metric_names):
    test_value = values_test[i]

    # Crear la figura
    plt.figure(figsize=(4, 4))
    bar = plt.bar(
        [name],
        [test_value],
        color="tab:blue",
        width=0.4
    )

    # Etiquetas y título
    plt.ylim(0.0, 1.0)
    plt.ylabel(name)
    plt.title(f"{name} (TEST)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Mostrar el valor encima de la barra
    height = bar[0].get_height()
    plt.text(
        bar[0].get_x() + bar[0].get_width() / 2,
        height + 0.02,
        f"{test_value:.3f}",
        ha="center",
        va="bottom"
    )

    # Guardar la figura en un archivo PNG
    safe_name = name.replace("@", "").replace(":", "_").replace(".", "")
    filename = os.path.join(output_dir, f"{safe_name}_TEST.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"✔ Gráfico guardado en: {filename}")

print(f"\nListo: se generaron y guardaron 5 imágenes en la carpeta '{output_dir}'.")
