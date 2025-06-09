import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH      = "runs/detect/train4/weights/best.pt"
DATA_YAML       = "config.yaml"
VAL_RESULTS_CSV = "runs/detect/train4/results.csv"
IMG_SIZE        = 640
BATCH           = 4
DEVICE          = "cpu"
RLTS_DIR        = Path("runs/detect/test_eval")
OUT_DIR         = RLTS_DIR / "metricas"

def ensure_test_split():
    """Si no existe dataset/test/images, ejecuta dataset.py"""
    test_images = Path("dataset/test/images")
    if test_images.is_dir() and any(test_images.iterdir()):
        print("[OK] dataset/test/ ya existe.")
        return

    print("[INFO] dataset/test/ no encontrado; generando con dataset.py…")
    if not Path("dataset.py").is_file():
        sys.exit("❌ No encuentro dataset.py. Colócalo junto a este script.")

    subprocess.run([sys.executable, "dataset.py"], check=True)

    if not test_images.is_dir() or not any(test_images.iterdir()):
        sys.exit("❌ Después de correr dataset.py sigue sin existir dataset/test/. Revisa rutas en dataset.py.")


def eval_on_test():
    """Ejecuta la validación en split=test y devuelve el objeto Results"""
    print("\n[INFO] Evaluando modelo en TEST …")
    model = YOLO(MODEL_PATH)
    results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=str(RLTS_DIR.parent),  # "runs/detect"
        name=RLTS_DIR.name,            # "test_eval"
        exist_ok=True
    )
    return results


def load_val_metrics():
    """
    Carga la última fila de results.csv con las métricas de validación.
    Calcula F1 = 2·P·R / (P + R), ya que el CSV no incluye F1.
    """
    if not Path(VAL_RESULTS_CSV).is_file():
        sys.exit(f"❌ No encuentro {VAL_RESULTS_CSV}. Asegúrate de tenerlo en la misma carpeta.")
    df = pd.read_csv(VAL_RESULTS_CSV)
    last = df.iloc[-1]

    p_val   = float(last["metrics/precision(B)"])
    r_val   = float(last["metrics/recall(B)"])
    map50   = float(last["metrics/mAP50(B)"])
    map5095 = float(last["metrics/mAP50-95(B)"])
    if p_val + r_val > 0:
        f1_val = 2 * (p_val * r_val) / (p_val + r_val)
    else:
        f1_val = 0.0

    return {
        "Precision": round(p_val, 3),
        "Recall":    round(r_val, 3),
        "mAP50":     round(map50, 3),
        "mAP50-95":  round(map5095, 3),
        "F1":        round(f1_val, 3),
    }


def extract_test_metrics(results):
    """Devuelve métricas globales (promedio de clases) desde `results`."""
    precision = float(results.box.mp)
    recall    = float(results.box.mr)
    map50     = float(results.box.map50)
    map5095   = float(results.box.map)
    import numpy as np
    f1_array  = np.asarray(results.box.f1, dtype=float)
    f1_global = float(f1_array.mean())

    return {
        "Precision": round(precision, 3),
        "Recall":    round(recall, 3),
        "mAP50":     round(map50, 3),
        "mAP50-95":  round(map5095, 3),
        "F1":        round(f1_global, 3),
    }


def compare_and_plot(val_m, test_m):
    """Imprime tabla comparativa y genera gráficos con valores encima de cada barra."""
    print("\n========== COMPARACIÓN VAL vs TEST ==========")
    for k in val_m:
        delta = (test_m[k] - val_m[k]) * 100
        print(f"{k:10s}: val={val_m[k]*100:.2f}% | test={test_m[k]*100:.2f}% | Δ={delta:+.1f}%")

    # Nos aseguramos de que exista la carpeta de salida
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for k in val_m:
        plt.figure()
        bars = plt.bar(["Validación", "Test"], [val_m[k], test_m[k]], width=0.5)
        plt.title(f"{k} • Validación vs Test")
        plt.ylim(0, 1)
        plt.ylabel(f"{k} (%)")
   
        # Añadimos el valor encima de cada barra
        for idx, height in enumerate([val_m[k], test_m[k]]):
            plt.text(
                idx,
                height + 0.02,
                f"{height*100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

        filename = OUT_DIR / f"{k.replace(' ', '_')}_val_vs_test.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"\n[OK] PNG guardados en {OUT_DIR}/")


def main():
    ensure_test_split()
    results      = eval_on_test()
    test_metrics = extract_test_metrics(results)
    val_metrics  = load_val_metrics()
    compare_and_plot(val_metrics, test_metrics)


if __name__ == "__main__":
    main()
