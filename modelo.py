# Importa la clase YOLO de la libería ultralytics, para que puedas crear y entrenar modelos con facilidad.
from ultralytics import YOLO
import yaml

if __name__ == '__main__':
    # Se crea un nuevo moselo desde cero usando la configuración de yolov8n.yaml
    # yolov8n: versión muy pequeña y rápida, buena para pruebas o dispositivos con pocos recursos
    # .yaml: archivo que describe cómo es la estructura del modelo
    # model.to('cuda:0') pasa de usar el cpu a usar la gpu
    model = YOLO("yolov8n.yaml")
    model.to('cuda:0')

    # with open("hyp.yaml", "r") as f:
    #   hyp = yaml.safe_load(f)
      
    # Se entrena el modelo con tus propios datos
    # data="./config.yaml": usa el archivo que contiene rutas de entrenamiento, validación, test, cantidad de clases y sus nombres
    # epochs=100 y patience=5: establece un maximo de 100 epocas pero en caso de que no mejore durante 5 epocas seguidas se frena el entrnamiento.

    # results = model.train(data="./config.yaml", epochs=100, amp=False, workers=0, patience=5, augment=True, **hyp)
    results = model.train(data="./config.yaml", epochs=100, amp=False, workers=0, patience=5)