# Vision Transformer (ViT) - Conceptos Fundamentales

## 1. Transformador (Transformer)

Originalmente propuesto para procesamiento de lenguaje natural (NLP) en el articulo "Attention is All You Need". El Transformer se basa completamente en mecanismos de atencion, sin usar recurrencia ni convoluciones. ViT adapta esta arquitectura al dominio de la vision computacional.

## 2. Patch Embedding (Incrustacion de Parches)

A diferencia de las CNN que procesan pixeles individuales mediante convoluciones, ViT divide la imagen en parches (patches) cuadrados no superpuestos.

Para una imagen de 224x224 con parches de 16x16:

```
Numero de parches = (224/16)^2 = 196 parches
Dimension de cada parche = 16 * 16 * 3 = 768
```

Cada parche se "aplanan" en un vector unidimensional y se proyecta linealmente a una dimension de embedding fija (D). Este proceso reemplaza a la convolucion como metodo de extraccion de caracteristicas inicial.

## 3. Self-Attention (Autoatencion)

El mecanismo central del Transformer. Permite que cada posicion en la secuencia "preste atencion" a todas las otras posiciones.

Para cada parche, se calculan tres vectores:

```
Q (Query):  "Que estoy buscando?"
K (Key):    "Que ofrezco?"
V (Value):  "Que informacion tengo?"
```

La atencion se calcula como:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Donde la matriz Q * K^T contiene los "pesos de atencion" que indican cuanta relacion tiene cada parche con los demas.

## 4. Multi-Head Attention (Atencion Multi-Cabeza)

En lugar de calcular una sola atencion, ViT calcula "h" atenciones en paralelo (cabezas). Cada cabeza aprende diferentes relaciones entre los parches:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
```

Para ViT-B/16: h = 12 cabezas de atencion.

Esto permite que el modelo capture diferentes tipos de relaciones simultaneamente (por ejemplo, bordes, texturas, formas, contexto global).

## 5. CLS Token (Token de Clasificacion)

Al inicio de la secuencia de parches se agrega un token especial llamado [CLS] (class token). Este token no corresponde a ningun parche real, sino que es un vector aprendible.

Durante el entrenamiento, el [CLS] token agrega informacion global de toda la imagen a traves de las capas de atencion. Al final del Transformer, solo la salida del [CLS] token se usa para la clasificacion.

## 6. Positional Encoding (Codificacion Posicional)

Como el mecanismo de atencion es invariante a la posicion (permutacion), ViT necesita incorporar informacion sobre donde esta cada parche en la imagen.

ViT usa **positional embeddings aprendibles** que se suman a los patch embeddings:

```
Embedding final = Patch Embedding + Positional Embedding
```

Esto le indica al modelo la posicion relativa de cada parche (arriba-izquierda, centro, abajo-derecha, etc.).

## 7. Transformer Encoder (Codificador Transformer)

ViT utiliza solo el codificador del Transformer original. Cada bloque del codificador consiste en:

**Layer Normalization (LN):** Normaliza la entrada para estabilizar el entrenamiento, similar a Batch Normalization pero operando sobre la dimension de embedding.

**Multi-Head Self-Attention (MSA):** Calcula la atencion entre todos los parches.

**Skip Connection:** Suma la entrada original a la salida de la atencion (identico al concepto de ResNet).

**MLP (Multilayer Perceptron):** Dos capas Fully Connected con activacion GELU.

**Segunda Skip Connection:** Suma la entrada del MLP a su salida.

```
Entrada
  -> Layer Norm
  -> Multi-Head Attention
  -> Skip Connection (+)
  -> Layer Norm
  -> MLP (FC + GELU + FC)
  -> Skip Connection (+)
  -> Salida
```

## 8. Arquitectura ViT-B/16

ViT-B/16 (Base, parches de 16x16):

| Componente | Valor |
|---|---|
| Tamano del parche | 16x16 |
| Numero de parches | 196 |
| Dimension del embedding (D) | 768 |
| Bloques Transformer | 12 |
| Cabezas de atencion | 12 |
| MLP dimension | 3072 (4 * D) |
| Parametros totales | 86M |

## 9. GELU (Gaussian Error Linear Unit)

Funcion de activacion usada en el MLP del Transformer:

```
GELU(x) = x * Phi(x)
```

Donde Phi(x) es la funcion de distribucion acumulada de la Gaussiana. GELU es una aproximacion suave de ReLU que permite gradientes negativos suaves.

## 10. Ventajas de ViT en Imagenes Medicas

- El self-attention captura dependencias globales, ideal para tumores que pueden estar en cualquier region
- No tiene el campo receptivo limitado de las CNN
- Puede relacionar caracteristicas distantes en la imagen (ej. bordes del tumor con el contexto cerebral)
- Teoricamente puede detectar patrones que las CNN no pueden por su naturaleza local

## 11. Limitaciones

**Alto costo computacional:** La complejidad del self-attention es O(n^2) donde n es el numero de parches. Para 196 parches, se calculan 196 x 196 = 38,416 pares de atencion.

**Requiere muchos datos:** ViT necesita grandes volumenes de datos de entrenamiento para superar a las CNN. Con datasets pequenos, las CNN suelen rendir mejor. El preentrenamiento en ImageNet mitiga esto parcialmente.

**Pérdida de informacion local:** Al dividir la imagen en parches, se pierde la estructura de pixeles vecinos que las CNN capturan naturalmente.

**Posicional embeddings fijos:** El numero de parches es fijo, lo que limita la resolucion de entrada.

## 12. Fine-tuning para Clasificacion Binaria

Se reemplaza la cabeza de clasificacion (originalmente 1000 clases de ImageNet) por una nueva capa Fully Connected con 2 salidas (tumor / no tumor). El [CLS] token al final del Transformer se usa como entrada para esta capa.
