# Canonist.ia

> **The go-to API for your real-estate portal image-classification.**

Canonist.ia es un servicio de clasificación de imágenes diseñado para inmobiliarias, que permite etiquetar automáticamente fotografías según su ubicación. Basado en aprendizaje por transferencia con PyTorch, ofrece una web‑app para integración inmediata creada con Streamlit.

---

## Estructura del proyecto

```
.
├── dataset/  
│   ├── training/  
│   └── validation/  
├── models/  
├── wandb/  
├── 03Transfer_Learning_def.ipynb  
├── app.py  
├── cnn.py
├── environment.yml  
└── test  
```

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

- **`test`**  
  Carpeta con 5 imágenes de prueba usadas durante las demostraciones en clase.

---

## Instalación

### Crea el entorno desde el .yml
conda env create -f environment.yml

### Actívalo
conda activate ML

---

## Uso

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
- **Visualización de probabilidades de predicción**

---

## Resultados

Los mejores resultados se obtuvieron realizando un fine‑tuning de la última etapa residual de ResNet. Concretamente, se descongelaron únicamente los tres bloques internos del layer4 (layer4.0, layer4.1 y layer4.2) y la cabeza de clasificación, dejando congeladas todas las capas previas.

### Resnet50

- **Accuracy** final en training: _98.69%_
- **Accuracy** final en validation: _94.87%_

### Resnet34

- **Accuracy** final en training: _99.16%_
- **Accuracy** final en validation: _94.47%_

---

## Enlaces

- Proyecto en GitHub: https://github.com/adriangustavodelpozomartin/machine_learning_II_CNN/tree/main
- Weights & Biases dashboard: https://wandb.ai/guillaume_-universidad-pontificia-comillas

---


