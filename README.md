# Canonist.ia

> **The go-to API for your real-estate portal image-classification.**

Canonist.ia es un servicio de clasificación de imágenes diseñado para inmobiliarias, que permite etiquetar automáticamente fotografías según su ubicación (interior, exterior, fachada, sala, cocina, etc.). Basado en aprendizaje por transferencia con PyTorch y Streamlit, ofrece una API sencilla y un web‑app para integración inmediata.

---

## 📂 Estructura del proyecto

```
.
├── __pycache__/  
├── dataset/  
│   ├── training/  
│   └── validation/  
├── models/  
├── wandb/  
├── 03Transfer_Learning_def.ipynb  
├── app.py  
├── cnn.py  
└── test.zip  
```

- **`__pycache__/`**  
  Carpeta generada automáticamente por Python para almacenar archivos compilados (`*.pyc`). Mejora el rendimiento al acelerar la carga de módulos.

- **`dataset/`**  
  - `training/`: Imágenes usadas para entrenar el modelo.  
  - `validation/`: Imágenes reservadas para validar y ajustar hiperparámetros.

- **`models/`**  
  Contiene los archivos `.pt` con los pesos de los mejores modelos guardados durante el entrenamiento.

- **`wandb/`**  
  Directorio generado por Weights & Biases para almacenar logs y métricas de experimentos.

- **`03Transfer_Learning_def.ipynb`**  
  Notebook principal donde se exploran los datos, se entrena el modelo (transfer learning) y se analizan los resultados.

- **`app.py`**  
  Script de Streamlit que levanta la interfaz web para consumir la API de Canonist.ia.

- **`cnn.py`**  
  Módulo con la clase `CNN`. Versión extendida para soportar múltiples arquitecturas (ResNet, EfficientNet, Swin, ViT, etc.).

- **`test.zip`**  
  Paquete con 5 imágenes de prueba usadas durante las demostraciones en clase.

---

## 🚀 Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/Canonist.ia.git
   cd Canonist.ia
   ```
2. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Uso

### Entrenamiento

En el notebook `03Transfer_Learning_def.ipynb` encontrarás paso a paso:
1. Preparación de datos con `torchvision.transforms`.  
2. Inicialización de modelos pre‑entrenados y ajuste de capas.  
3. Configuración de optimizador y scheduler.  
4. Monitorización con Weights & Biases.  

### Despliegue de la API

Ejecuta:
```bash
streamlit run app.py
```
Esto levantará un servidor local (por defecto en `http://localhost:8501`) con:
- **Subida de imagen**  
- **Predicción de ubicación**  
- **Visualización de métricas básicas**

---

## 📈 Resultados

- **Accuracy** final en validación: _XX.X%_  
- **Mejor modelo**: `efficientnet_b0` con fine‑tuning de las últimas 2 capas.

*(Consultad el notebook para detalle de curvas de entrenamiento y comparativa entre arquitecturas).*

---

## 🔗 Enlaces

- Proyecto en GitHub: https://github.com/tu-usuario/Canonist.ia  
- Weights & Biases dashboard: `[enlace a tu proyecto wandb]`

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT.  
Visita `LICENSE` para más detalles.
