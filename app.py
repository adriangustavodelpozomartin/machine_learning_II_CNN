import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from cnn import CNN, load_model_weights

# Configuración de la app
st.set_page_config(
    page_title="Clasificador de Entornos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
IMG_SIZE = 224
NUM_CLASSES = 15
CLASSES = [
    'Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'Inside city',
    'Kitchen', 'Living room', 'Mountain', 'Office', 'Open country', 'Store',
    'Street', 'Suburb', 'Tall building'
]

# Función para cargar el modelo (CPU/GPU)
@st.cache_resource
def get_model():
    model_weights = load_model_weights('resnet34_3unfreeze_overfitted')
    model = CNN(torchvision.models.resnet34(weights='DEFAULT'), NUM_CLASSES)
    model.load_state_dict(model_weights)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Predicción
def predict_image(image, model, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().tolist()
        pred_idx = int(torch.argmax(torch.tensor(probs)))
        return pred_idx, probs

# Barra lateral con instrucciones
with st.sidebar:
    st.header("Instrucciones")
    st.write("1. Sube una imagen.")
    st.write("2. Espera la predicción del modelo.")
    st.write("3. Observa las probabilidades.")
    st.markdown("---")
    st.write("Desarrollado por *Gonzalo Bobillo, Guillaume Guers, Adrián Gustavo del Pozo, Alberto Sáez-Royuela*")

# Título principal
st.title("Clasificador de Entornos ")
st.markdown("---")

# Carga de imagen
uploaded_file = st.file_uploader("Selecciona una imagen para clasificar", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen seleccionada", use_container_width=True)
    st.markdown("---")

    # Obtener modelo y dispositivo
    model, device = get_model()

    # Predicción
    pred_idx, probs = predict_image(image, model, device)
    pred_class = CLASSES[pred_idx]

    # Mostrar resultado
    st.success(f"**Predicción:** {pred_class}")
    st.markdown("**Probabilidades:**")

    # Barras horizontales con Streamlit y HTML (longitud proporcional)
    for cls, p in zip(CLASSES, probs):
        # Color gradiente rojo->verde
        r = int((1 - p) * 255)
        g = int(p * 255)
        bar_width = int(p * 100)
        # Render HTML
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
          <div style="width: 120px; font-size: 0.9rem;">{cls}</div>
          <div style="position: relative; background-color: #e0e0e0; width: 100%; height: 18px; border-radius: 3px; margin: 0 8px;">
            <div style="background-color: rgb({r},{g},0); width: {bar_width}%; height: 100%; border-radius: 3px;"></div>
          </div>
          <div style="width: 40px; font-size: 0.9rem; text-align: right;">{p:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Arquitectura ResNet34 - MSc Big Data")

