# CoAtNet - Conceptos Fundamentales

## 1. Motivacion: Unir lo mejor de CNN y Transformer

Las CNN y los Transformers tienen ventajas complementarias:

**CNN:**
- Inductivo de localidad: los pixeles cercanos estan mas relacionados
- Traduccion equivariancia: detecta patrones sin importar su posicion
- Eficiencia computacional con convoluciones depthwise
- Buen rendimiento con pocos datos

**Transformers (ViT):**
- Self-attention global: captura relaciones entre cualquier par de pixeles
- Mejor escalabilidad con mas datos
- Capacidad de modelar dependencias de largo alcance

CoAtNet busca combinar ambas mediante la **profunda integracion de convoluciones y atencion** en una sola arquitectura, no como modulos separados.

## 2. CoAtNet = Convolution + Attention

El nombre CoAtNet proviene de "Convolution and Attention Network". La idea central es construir bloques que mezclen operaciones convolucionales y de atencion dentro de la misma etapa de procesamiento.

## 3. Ecuacion de Convolucion vs Atencion

**Convolucion (depthwise):**

La convolucion depthwise se puede expresar como una suma sobre pixeles vecinos en una ventana local:

```
y_i = suma_{j en N(i)} w_{i-j} * x_j
```

Donde w_{i-j} son los pesos del kernel y N(i) es la ventana local alrededor del pixel i.

**Self-Attention:**

La atencion se expresa como una suma ponderada sobre todos los pixeles:

```
y_i = suma_{j} (softmax( Q_i * K_j / sqrt(d) )) * V_j
```

Donde los pesos de atencion se calculan dinamicamente a partir del contenido de los pixeles.

## 4. Atencion Relativa (Relative Attention)

CoAtNet utiliza una forma de atencion que incorpora un sesgo de posicion relativa, similar a Swin Transformer pero aplicado globalmente:

```
y_i = suma_{j} (softmax( Q_i * K_j / sqrt(d) + B_{i-j} )) * V_j
```

Donde B_{i-j} es un sesgo aprendible que depende de la posicion relativa entre i y j. Esto combina:

- La **adaptabilidad al contenido** del self-attention (a traves de Q*K)
- La **invarianza a la posicion** de las CNN (a traves de B)

## 5. MBConv con Atencion

CoAtNet no usa bloques convolucionales puros ni bloques de atencion puros. En su lugar, utiliza **MBConv (Mobile Inverted Bottleneck)** de EfficientNet como base y luego lo fusiona con atencion.

**Estructura tipica de un bloque CoAtNet:**
```
Entrada
  -> Layer Norm
  -> Depthwise Conv 3x3 (extracion local)
  -> Layer Norm
  -> Self-Attention Relativa (modelado global)
  -> Skip Connection (+)
  -> MLP (FC + GELU + FC)
  -> Skip Connection (+)
  -> Salida
```

## 6. Arquitectura de Etapas (Stages)

CoAtNet organiza las etapas en orden creciente de capacidad de atencion:

```
Stem: Conv 3x3, stride 2 (reduce resolucion rapido)
  -> Stage 1: Bloque Convolucional (MBConv puro)
  -> Stage 2: Bloque Hibrido (Conv + Attention)
  -> Stage 3: Bloque Transformer (Attention puro)
  -> Stage 4: Bloque Transformer (Attention puro)
```

**Logica del diseno:**
- Las primeras etapas procesan mayor resolucion, donde la convolucion es mas eficiente
- Las etapas finales procesan baja resolucion, donde el self-attention es mas eficiente y captura mejor las relaciones globales
- La transicion es gradual, no abrupta

## 7. Componentes Clave de CoAtNet

**Stem Convolucional:**
Dos capas convolucionales 3x3 con stride 2 que reducen rapidamente la resolucion de 224x224 a 56x56, similar a ResNet.

**MBConv Block:**
Bloques Mobile Inverted Bottleneck con Squeeze-and-Excitation, utilizados en las primeras etapas.

**Relative Attention Block:**
Bloques de atencion con sesgo de posicion relativa, utilizados en las etapas superiores.

**Transformer Block:**
Bloques clasicos de Transformer con atencion global y MLP, utilizados en las ultimas etapas.

## 8. CoAtNet-0 rw_224 (la variante que usamos)

Es la version mas pequena de la familia CoAtNet, con pesos entrenados por Ross Wightman (rw) a resolucion 224. Sus parametros:

| Componente | Valor |
|---|---|
| Dimension del embedding (C) | 64 |
| Numero de bloques por etapa | [2, 2, 6, 2] |
| Numero de cabezas de atencion | [2, 4, 8, 16] |
| Factor de expansion MLP | 4 |
| Tamano de kernel Conv | 3x3 |
| Parametros totales | ~25M |

La variante `rw_224` se refiere a los pesos publicados por Ross Wightman (autor de timm), entrenados con una receta mejorada que incluye regularizacion mas fuerte y aumentacion de datos mas agresiva durante el preentrenamiento en ImageNet.

## 9. Por que CoAtNet mejora a ViT

**Ventajas de CoAtNet sobre ViT:**
- Las convoluciones iniciales procesan eficientemente la alta resolucion, reduciendo el costo del self-attention posterior
- El sesgo de posicion relativa permite generalizar mejor a diferentes resoluciones
- La combinacion local+global captura tanto bordes finos (convolucion) como contexto global (atencion)
- Converge mas rapido que ViT puro porque las convoluciones proporcionan un buen punto de partida

**Ventajas de CoAtNet sobre CNN:**
- El self-attention en las ultimas etapas permite relaciones globales que las CNN no pueden capturar
- Mejor rendimiento en imagenes donde el contexto global es importante (como MRI donde el tumor puede estar en cualquier region)
- Mayor capacidad de generalizacion con conjuntos de datos grandes

## 10. La Familia CoAtNet

| Modelo | Parametros | Precisión ImageNet |
|---|---|---|
| CoAtNet-0 rw_224 | 25M | 81.6% |
| CoAtNet-1 | 42M | 83.3% |
| CoAtNet-2 | 75M | 84.1% |
| CoAtNet-3 | 168M | 84.5% |
| CoAtNet-4 | 275M | 84.9% |

Nosotros usamos CoAtNet-0 por ser la mas adecuada para nuestro dataset y recursos computacionales limitados (Colab).

## 11. Relevancia para este trabajo

CoAtNet es la arquitectura mas moderna de las cinco que comparamos. Representa la tendencia actual de **unificar CNN y Transformer** en lugar de tratarlos como enfoques separados. Su rendimiento en este estudio permitira responder:

- "Puede una arquitectura hibrida superar a las CNN puras (ResNet, EfficientNet) y a los Transformers puros (ViT, Swin) en clasificacion de tumores cerebrales?"
- "El costo computacional adicional de la atencion se justifica con una mejora en precision?"

## 12. Limitaciones

- Mayor consumo de memoria GPU que modelos puramente convolucionales
- Mas lento de entrenar que EfficientNet-B0 debido al self-attention
- Al ser un modelo relativamente reciente, hay menos documentacion y herramientas de debugging
- El preentrenamiento en ImageNet puede no optimizar completamente la parte de atencion
