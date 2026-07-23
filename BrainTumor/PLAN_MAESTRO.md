# Plan Maestro — Clasificación de Tumores Cerebrales con Deep Learning

## Objetivo del documento

Organizar el trabajo del equipo para producir un **estudio comparativo riguroso** de 5 arquitecturas de Deep Learning aplicadas a la clasificación binaria de tumores cerebrales en MRI. El resultado final será un artículo científico con metodología reproducible, análisis multidimensional y discusión basada en evidencia.

---

## 1. Pregunta de Investigación

> ¿Cuál de las arquitecturas de Deep Learning (CNN, Transformer e híbrida) ofrece el **mejor equilibrio** entre **precisión**, **capacidad de generalización** y **costo computacional** para la clasificación binaria de tumores cerebrales en imágenes de resonancia magnética?

Esta pregunta guía todo el experimento y conecta directamente con la discusión del artículo.

---

## 2. Clasificación de los Modelos

| Familia | Modelo | Fundamento |
|---|---|---|
| CNN clásica | ResNet50 | Residual Learning, Skip Connections |
| CNN optimizada | EfficientNet-B0 | Compound Scaling (Depth × Width × Resolution) |
| Vision Transformer | ViT-B/16 | Patch Embedding, Self-Attention, CLS Token |
| Transformer jerárquico | Swin-T | Shifted Windows, Patch Merging, jerarquía |
| Arquitectura híbrida | CoAtNet | Convoluciones + Self-Attention unificados |

---

## 3. Metodología Experimental — Dos Fases

### Fase 1: Entrenamiento Base (comparación justa)

Todos los modelos se entrenan **exactamente igual**:

- Mismo dataset, mismo split train/val/test
- Mismo batch size (32)
- Mismo optimizador (Adam, lr=1e-4)
- Mismas épocas (30)
- Sin data augmentation (solo resize + normalize)
- Sin early stopping
- Sin scheduler
- Sin mixed precision

→ Permite responder: *¿Cuál arquitectura es intrinsicamente mejor?*

### Fase 2: Entrenamiento Optimizado

Mismas condiciones base, pero ahora se agregan:

- **Data Augmentation** (flip, rotation, color jitter, etc.)
- **Early Stopping** (paciencia = 7 épocas)
- **Learning Rate Scheduler** (ReduceLROnPlateau)
- **Mixed Precision** (torch.cuda.amp)
- **Model Checkpoint** (guardar best val loss)

→ Permite responder: *¿Las técnicas de optimización benefician por igual a CNN y Transformers?*

---

## 4. Pipeline Completo

```
Dataset
  ↓
Análisis Exploratorio (EDA)
  ├── Imágenes por clase
  ├── Resolución, tamaño promedio
  ├── Canales RGB
  └── Distribución train/val/test
  ↓
Limpieza (detección de corruptos y outliers)
  ↓
Particionamiento (train 64% / val 16% / test 20%)
  ↓
Data Augmentation (solo en train — después del split)
  ↓
Entrenamiento
  ├── Fase 1: Base (sin técnicas de optimización)
  └── Fase 2: Optimizado (con Early Stopping, Scheduler, AMP)
  ↓
Validación + Early Stopping
  ↓
Guardar mejor modelo (best_model.pth)
  ↓
Evaluación
  ├── Métricas de rendimiento (Acc, Prec, Rec, F1, ROC-AUC, etc.)
  └── Métricas computacionales (params, tiempo, memoria GPU)
  ↓
Comparación entre modelos
  ↓
Discusión y Conclusiones
```

---

## 5. División del Trabajo — Responsabilidades Científicas

| Integrante | Modelo | Responsable de | Entregable |
|---|---|---|---|
| **1** | ResNet50 | Arquitecturas CNN + experimentos Fase 1 | Entrenamiento, análisis y redacción de teoría CNN |
| **2** | EfficientNet-B0 | Optimización del entrenamiento (Early Stopping, Scheduler, Checkpoint, AMP) | Entrenamiento Fase 2 y análisis de estrategias de entrenamiento |
| **3** | ViT + Swin | Fundamentos de Transformers y comparación CNN vs Transformer | Entrenamientos (3 modelos) y sección teórica de Transformers |
| **4 (tú)** | CoAtNet | Integración del proyecto, análisis comparativo y redacción del artículo | Entrenamiento CoAtNet, tablas, figuras, discusión y conclusiones |

### Qué debe investigar cada uno:

**Integrante 1 — ResNet50:**
- Residual Learning, Skip Connections
- Max Pooling, Convolución
- Por qué las CNN profundas funcionan bien en imágenes médicas

**Integrante 2 — EfficientNet-B0:**
- Compound Scaling (Depth, Width, Resolution)
- Eficiencia computacional (FLOPs, parámetros)
- Cómo afecta el scaling al rendimiento

**Integrante 3 — ViT + Swin:**
- Patch Embedding, Self-Attention, Multi-Head Attention
- CLS Token, Positional Encoding
- Shifted Windows (Swin), Patch Merging, jerarquía
- Complejidad O(n²) vs O(n)

**Integrante 4 (tú) — CoAtNet:**
- Convoluciones + Self-Attention en una sola arquitectura
- Cómo CoAtNet unifica ambas y por qué mejora a ViT
- Integración del artículo completo

---

## 6. Experimentos

| Experimento | Augmentation | Early Stopping | Scheduler | Mixed Precision |
|---|---|---|---|---|
| 1 — Base | No | No | No | No |
| 2 — Solo Aug | Sí | No | No | No |
| 3 — Solo ES | No | Sí | No | No |
| 4 — Solo Sched | No | No | Sí | No |
| 5 — Full | Sí | Sí | Sí | Sí |

Cada experimento se ejecuta en los 5 modelos → 25 entrenamientos en total.

---

## 7. Métricas

### Rendimiento (todas calculadas en test)
| Métrica | Fórmula |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall (Sensitivity) | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| F1-Score | 2 × (Prec × Rec) / (Prec + Rec) |
| ROC-AUC | Área bajo la curva ROC |
| Balanced Accuracy | (Sensitivity + Specificity) / 2 |
| MCC | Matthews Correlation Coefficient |

### Costo Computacional
| Métrica | Descripción |
|---|---|
| Parámetros totales | # de pesos del modelo |
| Tiempo de entrenamiento | Por época y total |
| Tiempo de inferencia | Por batch en test |
| Memoria GPU | Pico durante entrenamiento |
| Épocas ejecutadas | Con early stopping |

---

## 8. Gráficas para el Artículo

1. Distribución del dataset (train/val/test por clase)
2. Ejemplos de imágenes por clase (con y sin tumor)
3. Curva de entrenamiento (Loss por época) — por modelo
4. Curva de validación (Loss por época) — por modelo
5. Accuracy por época — por modelo
6. Curva ROC — todos los modelos superpuestos
7. Matriz de confusión — por modelo
8. Comparación de Accuracy entre modelos (bar chart)
9. Comparación de F1 entre modelos (bar chart)
10. Comparación de tiempo de entrenamiento (bar chart)
11. Comparación de número de parámetros (bar chart)

---

## 9. Criterios de parada inteligente (convergencia)

El entrenamiento **no corre las 30 épocas fijas**. Se detiene automáticamente cuando el modelo converge, usando el primer criterio que se cumpla:

| Criterio | Detalle |
|---|---|
| 🎯 Precisión objetivo | Si `val_accuracy ≥ 99.5%` → el modelo ya es excelente, se detiene |
| 📉 Convergencia | Si el loss no mejora significativamente en 5 épocas → ya no aprende |
| ⏸ Early stopping | Si pasan 7 épocas sin superar el mejor loss |
| 📉 LR mínimo | Si el LR baja de `1e-6` → el optimizador ya no puede avanzar |
| 🛡 Máximo seguro | 30 épocas como límite absoluto (nunca se pasa) |

Esto permite que los modelos CNN (convergen rápido) terminen antes, mientras que Transformers (más lentos) tengan el tiempo necesario.

## 10. Cómo Ejecutar

```bash
# Fase 1: Entrenamiento Base (solo máximo de épocas)
python src/train.py --model resnet50 --phase 1

# Fase 2: Entrenamiento Optimizado (parada por convergencia)
python src/train.py --model resnet50 --phase 2

# Evaluar
python src/evaluate.py --model resnet50 --phase 2

# Ver todo el análisis en notebook
# Abrir notebooks/colab_setup.ipynb en Colab
```

## 11. Clave para ser el mejor del curso

Lo que diferencia este trabajo no es entrenar 5 modelos (varios equipos lo harán).

Lo que nos hace destacar es:

1. **Protocolo experimental idéntico** para todos los modelos
2. **Dos fases** (Base vs Optimizado) → permite aislar el efecto de las técnicas
3. **Particionamiento antes de augmentación** → sin Data Leakage (error que cometen muchos)
4. **Métricas multidimensionales** → no solo accuracy, también costo computacional
5. **Discusión basada en evidencia** → no "CoAtNet ganó", sino *"¿Por qué ganó?"*
6. **Reproducibilidad** → todo documentado, todo automático

Esto demuestra que **sabemos hacer ciencia**, no solo correr modelos.
