# Tareas del Equipo

## Integrante 1 — ResNet50 + EfficientNet-B0

```bash
# Fase 1 (Base)
python src/train.py --model resnet50 --phase 1
python src/train.py --model efficientnet_b0 --phase 1

# Fase 2 (Optimizado)
python src/train.py --model resnet50 --phase 2
python src/train.py --model efficientnet_b0 --phase 2

# Evaluar
python src/evaluate.py --model resnet50 --phase 2
python src/evaluate.py --model efficientnet_b0 --phase 2
```

**Investigar y redactar:**
- Residual Learning, Skip Connections
- Compound Scaling (Depth, Width, Resolution)
- Por qué las CNN funcionan bien en imágenes médicas

---

## Integrante 2 — ViT

```bash
# Fase 1 (Base)
python src/train.py --model vit --phase 1

# Fase 2 (Optimizado)
python src/train.py --model vit --phase 2

# Evaluar
python src/evaluate.py --model vit --phase 2
```

**Investigar y redactar:**
- Patch Embedding, Self-Attention, Multi-Head Attention
- CLS Token, Positional Encoding
- Complejidad cuadrática O(n²) del Transformer

---

## Integrante 3 — Swin Transformer

```bash
# Fase 1 (Base)
python src/train.py --model swin --phase 1

# Fase 2 (Optimizado)
python src/train.py --model swin --phase 2

# Evaluar
python src/evaluate.py --model swin --phase 2
```

**Investigar y redactar:**
- Shifted Windows, Patch Merging
- Arquitectura jerárquica
- Complejidad lineal O(n) vs O(n²) de ViT

---

## Integrante 4 (tú) — CoAtNet

```bash
# Fase 1 (Base)
python src/train.py --model coatnet --phase 1

# Fase 2 (Optimizado)
python src/train.py --model coatnet --phase 2

# Evaluar
python src/evaluate.py --model coatnet --phase 2
```

**Investigar y redactar:**
- Cómo CoAtNet une convoluciones + self-attention
- Por qué mejora a ViT
- Integrar el artículo completo

---

## Procedimiento en Colab

Cada integrante sigue la **`COLAB_GUIDE.md`**:

1. Subir `BrainTumor/` y `BrainTumor_Dataset/` a su Drive
2. Abrir `notebooks/colab_setup.ipynb` en Colab
3. Cambiar `MODELO` por el suyo y `FASE=2`
4. Ejecutar todo en orden
5. Al terminar, copiar `results/<modelo>/optimized/` a carpeta compartida

## Recordar

- **Fase 1** = sin augmentation, sin early stopping (corre las 30 épocas)
- **Fase 2** = con todo (se detiene solo cuando converge)
