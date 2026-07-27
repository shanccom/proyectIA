# EfficientNet-B0 - Conceptos Fundamentales

## 1. Compound Scaling (Escalamiento Compuesto)

EfficientNet introduce la idea de que escalar una sola dimension de la red (profundidad, anchura o resolucion) produce mejoras limitadas. La clave es escalar **las tres dimensiones simultaneamente** de forma equilibrada mediante un coeficiente compuesto.

La formula de escalamiento es:

```
depth:      d = alpha^phi
width:      w = beta^phi
resolution: r = gamma^phi
```

Donde alpha * beta^2 * gamma^2 ≈ 2, y phi es el coeficiente de escalamiento.

## 2. Dimensiones de Escalamiento

**Profundidad (Depth):** Numero de capas en la red. Redes mas profundas pueden capturar caracteristicas mas complejas y abstractas. Sin embargo, las redes muy profundas sufren de vanishing gradient y dificultad de optimizacion. EfficientNet escala la profundidad moderadamente.

**Anchura (Width):** Numero de canales en cada capa. Redes mas anchas pueden capturar mas caracteristicas en cada nivel, pero aumentar la anchura incrementa el costo computacional cuadraticamente. EfficientNet escala la anchura de forma controlada.

**Resolucion (Resolution):** Tamano de la imagen de entrada. Mayores resoluciones permiten capturar detalles mas finos, pero aumentan el costo computacional. EfficientNet escala la resolucion gradualmente.

## 3. MBConv (Mobile Inverted Bottleneck Convolution)

El bloque basico de EfficientNet es MBConv, originalmente propuesto en MobileNetV2. Sus caracteristicas principales:

**Estructura del bloque MBConv:**
```
Entrada
  -> Convolucion 1x1 (expandir canales)
  -> Batch Normalization + Swish
  -> Depthwise Convolution 3x3 o 5x5
  -> Batch Normalization + Swish
  -> Convolucion 1x1 (comprimir canales)
  -> Batch Normalization
  -> Skip Connection (si las dimensiones lo permiten)
```

**Inverted Residual:** A diferencia de ResNet que primero reduce y luego expande, MBConv primero expande los canales (factor de 6), aplica depthwise convolution, y luego comprime. Esto permite que la depthwise convolution opere en un espacio de mayor dimensionalidad.

## 4. Depthwise Separable Convolution

Es una factorizacion de la convolucion estandar en dos pasos:

1. **Depthwise Convolution:** Aplica un filtro por cada canal de entrada de forma independiente
2. **Pointwise Convolution:** Convolucion 1x1 para combinar los canales

Esto reduce drasticamente el costo computacional: de (K^2 * C_in * C_out) a (K^2 * C_in + C_in * C_out).

## 5. Squeeze-and-Excitation (SE)

EfficientNet incorpora bloques SE (Squeeze-and-Excitation) que permiten a la red aprender cuales canales son mas importantes:

**Squeeze:** Comprime cada mapa de caracteristicas a un solo valor mediante Global Average Pooling.

**Excitation:** Pasa ese vector por dos capas Fully Connected:
- Primera capa: reduce la dimensionalidad (factor de 4)
- Segunda capa: restaura la dimensionalidad original con activacion Sigmoid

**Scale:** Multiplica cada canal original por su peso aprendido, realzando los canales relevantes y suprimiendo los irrelevantes.

## 6. Funcion de Activacion Swish

EfficientNet usa Swish en lugar de ReLU:

```
Swish(x) = x * sigmoid(x)
```

Swish es una funcion de activacion suave y no monotona que:
- Permite pequenos gradientes negativos (a diferencia de ReLU que los bloquea)
- Mejora el flujo del gradiente
- Ha demostrado superar a ReLU en redes profundas

## 7. Arquitectura Base de EfficientNet-B0

EfficientNet-B0 es la arquitectura base (phi=0) de la familia. Su estructura:

```
Stem: 3x3, 32 canales
Stage 1: MBConv1 3x3, 16 canales
Stage 2: MBConv6 3x3, 24 canales (2 bloques)
Stage 3: MBConv6 5x5, 40 canales (2 bloques)
Stage 4: MBConv6 3x3, 80 canales (3 bloques)
Stage 5: MBConv6 5x5, 112 canales (3 bloques)
Stage 6: MBConv6 5x5, 192 canales (4 bloques)
Stage 7: MBConv6 3x3, 320 canales (1 bloque)
Head: Conv 1x1, 1280 canales + Global Pool + FC
```

## 8. Eficiencia Computacional

EfficientNet-B0 logra la precision de ResNet50 con aproximadamente 10 veces menos parametros (5.3M vs 25.6M). Esto se debe a:

- Depthwise convolutions que reducen operaciones
- Compound scaling que optimiza el uso de recursos
- Bloques SE que mejoran la representacion sin costo excesivo

## 9. Familia EfficientNet (B0 a B7)

El coeficiente phi determina la version:

| Version | phi | Parametros | Precisión ImageNet |
|---|---|---|---|
| B0 | 0.0 | 5.3M | 76.3% |
| B1 | 1.0 | 7.8M | 78.8% |
| B2 | 2.0 | 9.2M | 79.6% |
| B3 | 3.0 | 12M | 81.1% |
| B4 | 4.0 | 19M | 82.6% |
| B5 | 5.0 | 30M | 83.3% |
| B6 | 6.0 | 43M | 84.0% |
| B7 | 7.0 | 66M | 84.4% |

## 10. Aplicacion a Imagenes Medicas

- La eficiencia computacional permite entrenar mas rapido en GPUs limitadas (Colab)
- El compound scaling ofrece buen equilibrio entre precision y velocidad
- Los bloques SE ayudan a enfocarse en caracteristicas relevantes de los tumores
- MBConv con depthwise convolution reduce el sobreajuste al tener menos parametros

## 11. Limitaciones

- MBConv puede perder informacion de posicion al comprimir canales
- El escalamiento compuesto fue disenado para ImageNet, puede no ser optimo en imagenes medicas
- Sensible a hiperparametros de regularizacion (drop connect, weight decay)
