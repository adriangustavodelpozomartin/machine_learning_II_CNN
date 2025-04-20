# para correr la app, hay que escribir en el terminal: streamlit run app.py

import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from cnn import CNN, load_model_weights

img_size = 224  
num_clases = 15 
CLASSES = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'Inside city', 'Kitchen', 'Living room', 'Mountain', 'Office', 'Open country', 'Store', 'Street', 'Suburb', 'Tall building']
# Función para cargar el modelo
@st.cache_resource
def get_model():
    model_weights = load_model_weights('resnet34_3unfreeze_lrPequeño')
    model = CNN(torchvision.models.resnet34(weights='DEFAULT'), num_clases)
    model.load_state_dict(model_weights)
    model.eval()  # Cambia el modelo a modo evaluación
    return model

# Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Predicción para una sola imagen
def predict_image(image, model):
    img_tensor = transform(image).unsqueeze(0)  # Asegura el 4D
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return predicted_class, probs[0].tolist()

# Interfaz Streamlit
st.title("Clasificador de entornos")
st.write("Sube una imagen para clasificarla.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # convertimos a RGB por si viene en otro formato
    st.image(image, caption="Imagen subida", use_container_width=True)

    model = get_model()
    pred_idx, probs = predict_image(image, model)
    st.success(f"Predicción: {CLASSES[pred_idx]}")
    st.write("Probabilidades:")
    for i, prob in enumerate(probs):
        st.write(f"{CLASSES[i]}: {prob:.2f}")
