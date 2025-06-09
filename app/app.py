import streamlit as st
import cv2, os
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image
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
            
div.st-emotion-cache-fis6aj {
    color: #FFFFFF;
    background-color: rgb(38, 39, 48);
    border-radius: 8px;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

div.stFileUploaderFile * {
    color: rgba(250, 250, 250, 0.6);
}
               
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 2) ENCABEZADO (TÍTULO + SUBTÍTULO)
# ------------------------------------------
st.markdown(
    "<div style='width:100%; height:10px;"
    "background: repeating-linear-gradient(45deg, #FEBA28 0px, #FEBA28 10px, #000000 10px, #000000 20px);'></div>",
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; font-weight:800; color:#000000;'>PERSONAS Y CHALECOS DE SEGURIDAD</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#000000;'>Detección en Tiempo Real, en Imágenes y Videos</h4>", unsafe_allow_html=True)

# Banda de seguridad (rayas amarillas y negras)
st.markdown(
    "<div style='width:100%; height:10px; margin-bottom: 20px; "
    "background: repeating-linear-gradient(45deg, #FEBA28 0px, #FEBA28 10px, #000000 10px, #000000 20px);'></div>",
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
        # Load the trained YOLO model
        model = YOLO("../runs/detect/train3/weights/best.pt")
        cap = cv2.VideoCapture(0)

        frame_times = []

        if not cap.isOpened():
            st.error("Error al abrir la cámara.")
            st.session_state.cam_on = False
        else:
            # Crear placeholder para cartel de personas
            person_count_placeholder = st.empty()
            latency_placeholder = st.empty()
            try:
                while st.session_state.cam_on:
                    start_time = time.time() 
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error capturando frame.")
                        break
                    
                    # Perform detection
                    frame = cv2.flip(frame, 1)
                    results = model(frame, conf=0.5)[0]  # conf=0.5 for minimum confidence
                    elapsed = (time.time() - start_time) * 1000  # en milisegundos
                    
                    person_count = sum(1 for box in results.boxes if int(box.cls[0]) == 0)
                    hivis_count = sum(1 for box in results.boxes if int(box.cls[0]) == 1)
                    personas_sin_chaleco = person_count - hivis_count
                    personas_con_chaleco = person_count - personas_sin_chaleco

                    # Guarda la info en session_state para la otra page
                    if "personas_x_frame" not in st.session_state:
                        st.session_state["personas_x_frame"] = []
                    if "chalecos_x_frame" not in st.session_state:
                        st.session_state["chalecos_x_frame"] = []
                    if "latencias" not in st.session_state:
                        st.session_state["latencias"] = []
                    
                    st.session_state["personas_x_frame"].append(person_count)
                    st.session_state["chalecos_x_frame"].append(hivis_count)
                    st.session_state["latencias"].append(elapsed)
                    frame_timestamp = time.time()
                    if "frame_timestamps" not in st.session_state:
                        st.session_state["frame_timestamps"] = []
                    st.session_state["frame_timestamps"].append(frame_timestamp)

                    annotated_frame = results.plot()
                    rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                    frame_times.append(elapsed)
                    if len(frame_times) > 30:  # promedio de los últimos 30 frames
                        frame_times = frame_times[-30:]
                    avg_latency = sum(frame_times) / len(frame_times)
                    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0

                    # Mostrar cartel estático abajo a la derecha
                    person_count_placeholder.markdown(
                        f"""
                        <div style="
                            position: fixed;
                            bottom: 75px;
                            right: 20px;
                            background-color: #FEBA28;
                            color: black;
                            padding: 10px 20px;
                            border-radius: 10px;
                            font-weight: bold;
                            font-size: 1rem;
                            box-shadow: 1px 1px 4px rgba(0,0,0,0.2);
                            z-index: 9999;
                        ">
                            Personas detectadas: {person_count}<br>
                            Chalecos detectados: {hivis_count}<br>
                            Personas sin chaleco: {personas_sin_chaleco}<br>
                            Personas sin chaleco: {personas_con_chaleco}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    latency_placeholder.markdown(
                        f"""
                        <div style="
                            position: fixed;
                            bottom: 20px;
                            right: 20px;
                            background-color: #FEBA28;
                            color: black;
                            padding: 10px 20px;
                            border-radius: 10px;
                            font-weight: bold;
                            font-size: 1rem;
                            box-shadow: 1px 1px 4px rgba(0,0,0,0.2);
                            z-index: 9999;
                        ">
                            Latencia promedio: {avg_latency:.1f} ms &nbsp;|&nbsp; FPS: {avg_fps:.1f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
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
        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Load the trained YOLO model
        model = YOLO("../runs/detect/train3/weights/best.pt")
        
        # Perform detection
        with st.spinner("Analizando imagen..."):
            results = model(image, conf=0.5)[0]

            # --- Contar personas y chalecos ---
            person_count = sum(1 for box in results.boxes if int(box.cls[0]) == 0)
            hivis_count = sum(1 for box in results.boxes if int(box.cls[0]) == 1)
            personas_sin_chaleco = person_count - hivis_count
            personas_con_chaleco = hivis_count

            # Annotate the image with detections
            annotated_image = results.plot()  # Draw bounding boxes and labels
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Display original and annotated images side by side
        ci1, ci2, ci3 = st.columns([1, 2, 1])
        with ci2:
            st.image(annotated_image_rgb, use_container_width=True)
            st.success("Análisis de imagen terminado")

        st.markdown(
            f"""
            <div style="
                position: fixed;
                bottom: 75px;
                right: 20px;
                background-color: #FEBA28;
                color: black;
                padding: 10px 20px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 1rem;
                box-shadow: 1px 1px 4px rgba(0,0,0,0.2);
                z-index: 9999;
            ">
                Personas detectadas: {person_count}<br>
                Chalecos detectados: {hivis_count}<br>
                Personas con chaleco: {personas_con_chaleco}<br>
                Personas sin chaleco: {personas_sin_chaleco}
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------
# 6) TAB 2: ANALIZAR VIDEO
# ------------------------------------------
with tab3:
    st.markdown("<h3 style='text-align:left; color:#000000;'>Análisis de Video Subido</h3>", unsafe_allow_html=True)

    # uploader centrado
    cu1, cu2, cu3 = st.columns([1, 2, 1])
    with cu2:
        uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov"])

    if uploaded_video:
        temp_in = "temp_video.mp4"
        with open(temp_in, "wb") as f:
            f.write(uploaded_video.getbuffer())

        model = YOLO("../runs/detect/train3/weights/best.pt")
        cap   = cv2.VideoCapture(temp_in)

        if not cap.isOpened():
            st.error("No se pudo abrir el video")
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25

            # ❱❱ NUEVO: placeholder en columna central
            v1, v2, v3 = st.columns([1, 2, 1])
            frame_box  = v2.empty()

            person_count_placeholder = st.empty()
            latency_placeholder = st.empty()

            bar = st.progress(0.0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            max_lado = max(w, h)
            scale = 1.0 if max_lado <= 720 else 720 / max_lado

            frame_idx = 0
            frame_times = []
            
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

                results   = model(frame, conf=0.5)[0]
                rgb       = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
                frame_box.image(rgb, channels="RGB", use_container_width=True)

                frame_idx += 1
                if frame_idx % 10 == 0:
                    bar.progress(min(frame_idx / total, 1.0))

                elapsed = (time.time() - start_time) * 1000  # en milisegundos
                person_count = sum(1 for box in results.boxes if int(box.cls[0]) == 0)
                hivis_count = sum(1 for box in results.boxes if int(box.cls[0]) == 1)
                personas_sin_chaleco = person_count - hivis_count
                personas_con_chaleco = person_count - personas_sin_chaleco    

                frame_timestamp = time.time()
                if "frame_timestamps" not in st.session_state:
                    st.session_state["frame_timestamps"] = []
                st.session_state["frame_timestamps"].append(frame_timestamp)

                # Guarda la info en session_state para la otra page
                if "personas_x_frame" not in st.session_state:
                    st.session_state["personas_x_frame"] = []
                if "chalecos_x_frame" not in st.session_state:
                    st.session_state["chalecos_x_frame"] = []
                if "latencias" not in st.session_state:
                    st.session_state["latencias"] = []
                
                st.session_state["personas_x_frame"].append(person_count)
                st.session_state["chalecos_x_frame"].append(hivis_count)
                st.session_state["latencias"].append(elapsed)

                frame_times.append(elapsed)
                if len(frame_times) > 30:
                    frame_times = frame_times[-30:]
                avg_latency = sum(frame_times) / len(frame_times)
                avg_fps = 1000 / avg_latency if avg_latency > 0 else 0

                person_count_placeholder.markdown(
                    f"""
                    <div style="
                        position: fixed;
                        bottom: 75px;
                        right: 20px;
                        background-color: #FEBA28;
                        color: black;
                        padding: 10px 20px;
                        border-radius: 10px;
                        font-weight: bold;
                        font-size: 1rem;
                        box-shadow: 1px 1px 4px rgba(0,0,0,0.2);
                        z-index: 9999;
                    ">
                        Personas detectadas: {person_count}<br>
                        Chalecos detectados: {hivis_count}<br>
                        Personas sin chaleco: {personas_sin_chaleco}<br>
                        Personas con chaleco: {personas_con_chaleco}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                latency_placeholder.markdown(
                    f"""
                    <div style="
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        background-color: #FEBA28;
                        color: black;
                        padding: 10px 20px;
                        border-radius: 10px;
                        font-weight: bold;
                        font-size: 1rem;
                        box-shadow: 1px 1px 4px rgba(0,0,0,0.2);
                        z-index: 9999;
                    ">
                        Latencia promedio: {avg_latency:.1f} ms &nbsp;|&nbsp; FPS: {avg_fps:.1f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            cap.release()
            bar.empty()
            st.success("Análisis de video terminado")
        os.remove(temp_in)
