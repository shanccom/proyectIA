# Swin Transformer - Conceptos Fundamentales

## 1. Motivacion: Limitaciones de ViT

ViT tiene dos problemas principales:

1. **Complejidad cuadratica O(n^2):** El self-attention global calcula relaciones entre todos los pares de parches. Para una imagen de 224x224 con parches de 4x4, serian (56 x 56)^2 = 9.8 millones de pares.

2. **Escalabilidad:** ViT no puede procesar imagenes de alta resolucion eficientemente debido a la complejidad O(n^2).

Swin Transformer resuelve ambos problemas mediante atencion en ventanas locales y una arquitectura jerarquica.

## 2. Shifted Window Attention (Atencion de Ventanas Desplazadas)

En lugar de calcular atencion global, Swin divide la imagen en ventanas no superpuestas y calcula atencion solo dentro de cada ventana.

**Ventana regular:** La imagen de 56x56 feature maps se divide en ventanas de MxM (tipicamente M=7), dando (56/7)^2 = 64 ventanas. Cada ventana calcula atencion solo entre sus 7x7 = 49 parches.

**Complejidad:** O(M^2 * N) donde M es el tamano de la ventana y N el numero total de parches. Esto es **lineal** en lugar de cuadratico.

## 3. Shifted Window Partition (Particion de Ventanas Desplazadas)

Para permitir la comunicacion entre ventanas vecinas, Swin alterna entre dos tipos de particion en capas consecutivas:

**Capa L (regular):** Divide en ventanas regulares desde la esquina superior-izquierda.

**Capa L+1 (desplazada):** Desplaza las ventanas en (M/2, M/2) pixeles. Esto conecta parches que antes estaban en diferentes ventanas.

```
Capa L (regular):      Capa L+1 (desplazada):
+---+---+---+          +---+---+---+
| 1 | 1 | 2 |          | 5 | 5 | 6 |
+---+---+---+          +---+---+---+
| 1 | 1 | 2 |          | 5 | 5 | 6 |
+---+---+---+          +---+---+---+
| 3 | 3 | 4 |          | 7 | 7 | 8 |
+---+---+---+          +---+---+---+
```

## 4. Masked Multi-Head Self-Attention

Con las ventanas desplazadas, algunas ventanas contienen parches que no son adyacentes en la imagen original. Swin aplica una mascara de atencion para evitar que parches de diferentes regiones interactuen dentro de la misma ventana desplazada.

Esto es puramente computacional (no afecta la calidad del modelo) y permite calcular todas las ventanas en paralelo.

## 5. Patch Merging (Fusion de Parches)

Swin construye una representacion jerarquica similar a las CNN:

**Ejemplo de Patch Merging:**
- Entrada: 4x4 parches (56x56 pixeles)
- Agrupa 2x2 parches vecinos (concatena sus caracteristicas)
- Aplica una capa lineal para reducir la dimensionalidad a la mitad
- Salida: 2x2 parches (28x28 pixeles), pero con el doble de profundidad (canales)

Este proceso se repite en cada etapa, creando una piramide de caracteristicas:
```
Stage 1: 56x56 (alta resolucion, caracteristicas locales)
Stage 2: 28x28 (resolucion media)
Stage 3: 14x14 (baja resolucion, caracteristicas semanticas)
Stage 4: 7x7   (mas baja resolucion, caracteristicas globales)
```

## 6. Arquitectura Jerarquica

A diferencia de ViT que tiene una unica resolucion durante todo el procesamiento, Swin construye mapas de caracteristicas a multiples escalas, similar a ResNet y otras CNN.

```
Entrada (224x224)
  -> Patch Partition (4x4 parches) -> 56x56
  -> Stage 1: 2 bloques Swin -> 56x56
  -> Patch Merging -> 28x28
  -> Stage 2: 2 bloques Swin -> 28x28
  -> Patch Merging -> 14x14
  -> Stage 3: 6 bloques Swin -> 14x14
  -> Patch Merging -> 7x7
  -> Stage 4: 2 bloques Swin -> 7x7
  -> Global Average Pooling -> FC -> Clasificacion
```

Cada bloque Swin consiste en:
1. Layer Norm + Window Multi-Head Self-Attention (W-MSA)
2. Skip Connection
3. Layer Norm + MLP (GELU)
4. Skip Connection
5. (En capas impares) Shifted Window MSA (SW-MSA) en lugar de W-MSA

## 7. Relative Position Bias (Sesgo de Posicion Relativa)

Swin usa un sesgo de posicion relativa en lugar de positional embeddings absolutos como ViT. Para cada par de parches dentro de una ventana, se anade un sesgo aprendible basado en su desplazamiento relativo:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d) + B) * V
```

Donde B es el sesgo de posicion relativa. Esto permite que el modelo generalice mejor a diferentes tamanos de imagen y posiciones.

## 8. Complejidad Computacional

| Modelo | Complejidad | Escalabilidad |
|---|---|---|
| ViT (global attention) | O(n^2 * d) | Pobre para altas resoluciones |
| Swin (window attention) | O(M^2 * n * d) | Lineal en n, escalable |

Donde n es el numero de parches, d la dimension del embedding y M el tamano de la ventana (tipicamente 7).

Para una imagen de 224x224:
- ViT: 196^2 = 38,416 pares de atencion
- Swin: 64 ventanas * 49^2 = 153,664 pares de atencion (pero con M constante, no crece con la imagen)

## 9. Ventajas del Swin en Imagenes Medicas

- La arquitectura jerarquica permite detectar tumores a diferentes escalas (pequenos y grandes)
- La ventana local reduce el costo computacional, permitiendo procesar imagenes de mayor resolucion
- El shifted window asegura conexiones entre regiones vecinas sin costo adicional
- Similar a las CNN en estructura jerarquica, pero con atencion en lugar de convolucion

## 10. Diferencias Clave con ViT

| Aspecto | ViT | Swin Transformer |
|---|---|---|
| Atencion | Global | Local (ventanas) + desplazada |
| Complejidad | O(n^2) | O(n) |
| Jerarquia | Un solo nivel | Multi-escala (como CNN) |
| Posicion | Absolute embedding | Relative position bias |
| Parches | Fijos durante toda la red | Patch merging reduce resolucion |
| Tipo de token | CLS token | Global Average Pooling |
| Ideal para | Clasificacion global | Segmentacion y deteccion |

## 11. Fine-tuning para Clasificacion Binaria

Swin-T (tiny) tiene 28M de parametros. Para clasificacion binaria:
- Se reemplaza la cabeza de clasificacion (originalmente 1000 clases) por una capa Fully Connected con 2 salidas
- El Global Average Pooling produce un vector de caracteristicas que alimenta la FC final

## 12. Parametros de Swin-T

| Componente | Valor |
|---|---|
| Dimension del embedding (C) | 96 |
| Capas en cada stage | 2, 2, 6, 2 |
| Tamano de ventana (M) | 7 |
| Numero de cabezas | 3, 6, 12, 24 |
| MLP ratio | 4 |
| Parametros totales | 28M |
