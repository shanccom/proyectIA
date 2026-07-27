# Guía para ejecutar en Google Colab

## 1. Subir los archivos a Google Drive

```
MiDrive/
├── BrainTumor/                  # ← todo el proyecto (menos dataset)
│   ├── src/
│   ├── notebooks/
│   ├── results/
│   ├── requirements.txt
│   └── ...
└── BrainTumor_Dataset/          # ← el dataset
    ├── train/{no, yes}/
    ├── val/{no, yes}/
    └── test/{no, yes}/
```

## 2. Abrir el notebook en Colab

```
https://colab.research.google.com/
```

Abrir: `BrainTumor/notebooks/colab_setup.ipynb`

## 3. Cambiar modelo y fase

En la **Celda 3**:

```python
MODELO = "coatnet"   # resnet50 | efficientnet_b0 | vit | swin | coatnet
FASE = 2              # 1 = Base, 2 = Optimizado
```

Ajustar rutas si es necesario:

```python
RUTA_DRIVE = "/content/drive/MyDrive/BrainTumor"
RUTA_DATASET = "/content/drive/MyDrive/BrainTumor_Dataset"
```

## 4. Ejecutar en orden

| Celda | Paso | Tiempo |
|---|---|---|
| 0 | Montar Drive | ~10s |
| 1 | Configurar | ~5s |
| 2 | Copiar proyecto y dataset | ~2min |
| 3 | Instalar dependencias | ~1min |
| 4 | Verificar GPU | ~5s |
| 5 | EDA (distribución + ejemplos) | ~10s |
| 6 | Limpieza de imágenes corruptas | ~30s |
| 7 | **Prueba rápida** (2 épocas) | ~2min |
| 8 | **Entrenamiento completo** | variable |
| 9 | Evaluación | ~1min |
| 10 | Guardar resultados en Drive | ~1min |

## 5. ¿Cuánto tarda el entrenamiento?

**No hay un tiempo fijo.** El entrenamiento se detiene automáticamente cuando el modelo **converge**, usando estos criterios (Fase 2):

| Criterio | Qué significa | Valor |
|---|---|---|
| Precisión objetivo | Si val_acc ≥ 99.5% → el modelo ya es excelente | `STOP_ACCURACY = 0.995` |
| Convergencia | Si el loss deja de mejorar por 5 épocas → ya no aprende más | `CONVERGENCE_WINDOW = 5` |
| Early stopping | Si pasan 7 épocas sin superar el mejor loss | `PATIENCE = 7` |
| LR mínimo | Si el learning rate baja de 1e-6 → no puede optimizar más | `MIN_LR = 1e-6` |
| Máximo seguro | Límite absoluto para no quedarse colgado | 30 épocas |

**En la práctica:**
- Modelos CNN (ResNet, EfficientNet): **10-20 épocas** (~15-40 min)
- Transformers (ViT, Swin, CoAtNet): **15-30 épocas** (~30-90 min)

El training se detiene apenas se cumple **el primer criterio**. No espera a terminar las 30 épocas si el modelo ya convergió.

## 6. Fases

**Fase 1 — Base** (solo máximo de épocas):
- Sin técnicas de optimización
- Corre las 30 épocas completas
- Útil para comparación justa entre modelos

**Fase 2 — Optimizado** (parada inteligente):
- Convergencia automática
- Data augmentation, Early Stopping, LR Scheduler, Mixed Precision

## 7. Distribución del equipo

Cada integrante ejecuta su modelo en ambas fases.

| Integrante | Modelo |
|---|---|
| 1 | `resnet50` |
| 2 | `efficientnet_b0` |
| 3 | `vit` y `swin` |
| 4 (tú) | `coatnet` |

## 8. Resultados guardados

```
results/
└── coatnet/
    ├── base/
    │   ├── best_model.pth
    │   ├── history.json        # loss, métricas, LR, tiempo por época
    │   ├── model_info.json     # parámetros, memoria GPU
    │   ├── metrics.json        # Acc, Prec, Rec, F1, ROC-AUC, MCC, etc.
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    └── optimized/
        └── ... (lo mismo)
```

## 9. Comandos manuales

```bash
python src/train.py --model coatnet --phase 2
python src/evaluate.py --model coatnet --phase 2
```
