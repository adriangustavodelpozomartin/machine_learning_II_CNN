import os, time
import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from cnn import CNN

# Configuración de la app
st.set_page_config(
    page_title="Clasificador de Entornos",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constantes
IMG_SIZE = 224
NUM_CLASSES = 15
CLASSES = [
    "Bedroom",
    "Coast",
    "Forest",
    "Highway",
    "Industrial",
    "Inside city",
    "Kitchen",
    "Living room",
    "Mountain",
    "Office",
    "Open country",
    "Store",
    "Street",
    "Suburb",
    "Tall building",
]

# Mapeo de nombres cortos a (archivo de pesos, constructor de modelo)
MODEL_OPTIONS = {
    "ResNet34": ("resnet34_3unfreeze.pt", torchvision.models.resnet34),
    "ResNet50": ("resnet50_3unfreeze_lr1e-4.pt", torchvision.models.resnet50),
}


# Detectar dispositivo: MPS (Mac) → CUDA → CPU
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Función para cargar el modelo (CPU/GPU/MPS)
@st.cache_resource
def get_model(model_key):
    weight_file, constructor = MODEL_OPTIONS[model_key]
    device = get_device()

    # Ruta completa al .pt
    path = os.path.join("models", weight_file)
    # Cargar pesos con map_location
    # state_dict = torch.load(path, map_location=device)
    state_dict = torch.load(path, map_location=device, weights_only=False)

    # Instanciar arquitectura y cargar pesos
    base_model = constructor(weights="DEFAULT")
    model = CNN(base_model, NUM_CLASSES)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, device


# Transformaciones
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)


# Predicción
def predict_image(image, model, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().tolist()
        pred_idx = int(torch.argmax(torch.tensor(probs)))
        return pred_idx, probs


# Barra lateral con instrucciones y selector de modelo
with st.sidebar:
    st.header("Instrucciones")
    st.write("1. Selecciona el modelo.")
    st.write("2. Sube una imagen.")
    st.write("3. Espera la predicción del modelo.")
    st.write("4. Observa las probabilidades.")
    st.markdown("---")
    model_choice = st.selectbox("Modelo", list(MODEL_OPTIONS.keys()))
    st.markdown("---")
    st.write(
        "Desarrollado por *Gonzalo Bobillo, Guillaume Guers, Adrián Gustavo del Pozo, Alberto Sáez-Royuela*"
    )

# Título principal
st.title("Clasificador de Entornos")
st.markdown("---")

# Carga de imagen
uploaded_file = st.file_uploader(
    "Selecciona una imagen para clasificar", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen seleccionada", use_container_width=True)
    st.markdown("---")

    # Obtener modelo y dispositivo
    model, device = get_model(model_choice)
    start_time = time.time()
    # Predicción
    pred_idx, probs = predict_image(image, model, device)
    elapsed_time = time.time() - start_time
    pred_class = CLASSES[pred_idx]

    # Mostrar resultado
    st.success(f"**Predicción:** {pred_class}")
    st.markdown(f"**Tiempo de respuesta:** {elapsed_time:.3f} segundos")
    st.markdown("**Probabilidades:**")

    # Barras horizontales con Streamlit y HTML (longitud proporcional)
    for cls, p in zip(CLASSES, probs):
        r = int((1 - p) * 255)
        g = int(p * 255)
        bar_width = int(p * 100)
        st.markdown(
            f"""
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
          <div style="width: 120px; font-size: 0.9rem;">{cls}</div>
          <div style="position: relative; background-color: #e0e0e0; width: 100%; height: 18px; border-radius: 3px; margin: 0 8px;">
            <div style="background-color: rgb({r},{g},0); width: {bar_width}%; height: 100%; border-radius: 3px;"></div>
          </div>
          <div style="width: 40px; font-size: 0.9rem; text-align: right;">{p:.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption(f"Arquitectura {model_choice} - MSc Big Data")
