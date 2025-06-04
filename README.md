# Sistemas_Inteligentes
- Comandos para crear y activar el environment virtual: python -m venv venv && .\venv\Scripts\Activate.ps1
- Comandos para instalar dependencias: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt
- Comando para crear el dataset: python dataset.py
- Comando para entrenar el modelo: python modelo.py
- Comando para evaluar el modelo: python evaluacion.py
- Comando para prender la app (dentro de carpeta app): python -m streamlit run app.py 