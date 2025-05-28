# Permite descargar datasets desde Kaggle fácilmente
import kagglehub
# Para interactuar con el sistema de archivos
import os
# Para abrir y manejar imágenes
from PIL import Image
# Para copiar archivos
import shutil
# Para manejar datos en forma de DataFrame
import pandas as pd
# Para operaciones numéricas
import numpy as np
# Para dividir datos en conjuntos de entrenamiento, validación y prueba
from sklearn.model_selection import train_test_split
# Sirve para leer y escribir archivos YAML, que son archivos de texto usados para guardar configuraciones de forma ordenada.
import yaml

# Descarga la última versión disponible del dataset especificado en Kaggle
kagglehub.dataset_download("tudorhirtopanu/yolo-highvis-and-person-detection-dataset")

# Define la ruta donde están almacenadas las imágenes y etiquetas del dataset
images_path = r"C:\Users\crist\.cache\kagglehub\datasets\tudorhirtopanu\yolo-highvis-and-person-detection-dataset\versions\1\YOLO-HiVis-Data\images"
labels_path = r"C:\Users\crist\.cache\kagglehub\datasets\tudorhirtopanu\yolo-highvis-and-person-detection-dataset\versions\1\YOLO-HiVis-Data\labels"

# Lista todos los archivos en esa carpeta que tengan extensión de imagen (insensible a mayúsculas)
# Hay 7937 imágenes en total
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Ruta base donde se guardarán los datos organizados
OUTPUT_PATH = './dataset'

# Función para crear un DataFrame con los IDs de las imágenes (sin extensión)
def create_df():
    # Lista los archivos de imagen y extrae su nombre sin extensión
    names = [f.split('.')[0] for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Retorna un DataFrame con una columna 'id'
    return pd.DataFrame({'id': names})

# Función para crear la estructura de carpetas necesarias para train, val y test
def create_dirs(base_path):
    # Itera por los tres subconjuntos
    for split in ['train', 'val', 'test']:
        # Crea carpeta de imágenes
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        # Crea carpeta de etiquetas
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)

# Función que copia las imágenes y etiquetas correspondientes a cada conjunto (train/val/test)
def copy_files(ids, split):
    # Itera por cada ID
    for id in ids:
        # Buscar imagen con extensión válida
        for ext in ['.jpg', '.jpeg', '.png']:
            # Ruta de la imagen original
            img_src = os.path.join(images_path, id + ext)
            # Verifica que exista
            if os.path.exists(img_src):
                # Ruta destino
                img_dst = os.path.join(OUTPUT_PATH, split, 'images', id + ext)
                # Copia la imagen
                shutil.copy(img_src, img_dst)
                # Sale del bucle al encontrar una imagen válida
                break

        # Copiar archivo de etiqueta correspondiente
        # Ruta de la etiqueta
        label_src = os.path.join(labels_path, id + '.txt')
        # Verifica que exista
        if os.path.exists(label_src):
            # Ruta destino
            label_dst = os.path.join(OUTPUT_PATH, split, 'labels', id + '.txt')
            # Copia la etiqueta
            shutil.copy(label_src, label_dst)

# Llama a la función para crear el DataFrame de IDs
df = create_df()
print('Total imágenes:', len(df))

# Si se encontraron imágenes
if len(df) > 0:
    # Divide el dataset: 2% para test y el resto para train + val
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.02, random_state=19)
    # Divide el conjunto restante en 80% train y 20% val
    X_train, X_val = train_test_split(X_trainval, test_size=0.2, random_state=19)

    # Crea las carpetas necesarias
    create_dirs(OUTPUT_PATH)

    # Copia los archivos a las carpetas correspondientes
    copy_files(X_train, 'train')
    copy_files(X_val, 'val')
    copy_files(X_test, 'test')

    print(f'Archivos organizados en "{OUTPUT_PATH}"')
    print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
else:
    print("No se encontraron imágenes.")

# Definir la configuración del archivo data.yaml
# Se crea un diccionario de configuración, una lista de datos que se van a guardar en el .yaml

# cambiar de probar a project
data_yaml = {
    # carpeta donde están las imágenes para entrenar el modelo
    'train': './dataset/train/images',
    # carpeta con imágenes pata validar si el modelo está aprendiendo bien
    'val': './dataset/val/images',
    # carpeta con immágenes para hacer pruebas finales del modelo
    'test': './dataset/test/images',
    # número de clases / objetos diferentes que puede detectar
    'nc': 2,
    # lista con los nombres de esas clases, en este caso son personas y chalecos reflectantes
    'names': ['person', 'hi-vis']
}

# Ruta donde se guarda el archivo
yaml_path = './config.yaml'

# Crear el archivo YAML
# Dentro de este archivo escribe los datos que definimos antes en data_yaml
with open(yaml_path, 'w') as f:
  # default_flow_style=False es para que el texto se vea ordenado
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"Archivo {yaml_path} creado exitosamente")