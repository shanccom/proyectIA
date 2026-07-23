# ResNet50 - Conceptos Fundamentales

## 1. Problema del Gradiente Fugitivo (Vanishing Gradient)

A medida que las redes neuronales se vuelven mas profundas, el gradiente calculado durante la retropropagacion tiende a desvanecerse (volverse extremadamente pequeno) en las capas iniciales. Esto impide que las primeras capas aprendan efectivamente, limitando la profundidad util de la red. Este fenomeno se conoce como "vanishing gradient problem".

## 2. Residual Learning (Aprendizaje Residual)

ResNet introduce el concepto de "mapeo residual" para resolver el vanishing gradient. En lugar de aprender directamente una funcion H(x), la red aprende el residuo F(x) = H(x) - x.

```
Capa tradicional:     y = F(x)
Capa residual:        y = F(x) + x
```

La conexion de salto (skip connection) permite que el gradiente fluya directamente a traves de la red sin atenuarse, lo que permite entrenar redes con cientos de capas.

## 3. Skip Connections (Conexiones de Salto)

Tambien llamadas "identity shortcuts". Son conexiones que saltan una o mas capas y suman la entrada original a la salida de las capas convolucionales.

**Tipos de Skip Connections en ResNet50:**

- **Identity shortcut:** Cuando la dimension de entrada y salida coinciden, simplemente se suma x + F(x).
- **Projection shortcut:** Cuando cambia la dimension (por ejemplo, al reducir el tamaño espacial o aumentar canales), se usa una convolucion 1x1 para ajustar las dimensiones antes de sumar.

## 4. Bloque Bottleneck

ResNet50 usa bloques "bottleneck" que constan de 3 capas convolucionales:

1. Convolucion 1x1: Reduce la dimensionalidad (comprime los canales)
2. Convolucion 3x3: Extrae caracteristicas espaciales
3. Convolucion 1x1: Restaura la dimensionalidad original

Este diseno reduce significativamente el costo computacional en comparacion con usar directamente una convolucion 3x3 con muchos canales.

## 5. Arquitectura de ResNet50

ResNet50 tiene 50 capas de profundidad y sigue la siguiente estructura:

```
Conv1:     7x7, 64 canales, stride 2
MaxPool:   3x3, stride 2
Stage 1:   3 bloques bottleneck (64 -> 64 -> 256)
Stage 2:   4 bloques bottleneck (128 -> 128 -> 512)
Stage 3:   6 bloques bottleneck (256 -> 256 -> 1024)
Stage 4:   3 bloques bottleneck (512 -> 512 -> 2048)
Average Pool -> FC (1000) -> Softmax
```

## 6. Batch Normalization

Cada capa convolucional en ResNet va seguida de Batch Normalization antes de la funcion de activacion ReLU. Batch Normalization normaliza la salida de cada capa para que tenga media 0 y varianza 1, lo que:

- Acelera la convergencia del entrenamiento
- Permite usar learning rates mas altos
- Actua como regularizador, reduciendo la necesidad de Dropout
- Reduce la sensibilidad a la inicializacion de pesos

## 7. Global Average Pooling

En lugar de usar capas completamente conectadas (Fully Connected) al final, ResNet utiliza Global Average Pooling: calcula el promedio de cada mapa de caracteristicas, produciendo un vector de tamanio igual al numero de canales. Esto reduce drasticamente el numero de parametros y previene el sobreajuste.

## 8. Transfer Learning en ResNet50

El modelo preentrenado en ImageNet ha aprendido a reconocer bordes, texturas y formas en sus primeras capas. Al fine-tunearlo para clasificacion de tumores cerebrales:

- Las capas convolucionales (backbone) ya saben extraer caracteristicas visuales utiles
- Solo se reemplaza la ultima capa (Fully Connected) para adaptarla a 2 clases (tumor / no tumor)
- Durante el fine-tuning, todas las capas se actualizan con el nuevo dataset

## 9. Por que ResNet50 funciona bien en imagenes medicas

- Las skip connections permiten entrenar redes profundas incluso con datasets medicos limitados
- Los filtros convolucionales detectan bordes y texturas relevantes para tumores
- Batch Normalization estabiliza el entrenamiento con imagenes de diferente contraste
- Global Average Pooling reduce sobreajuste, critico en datasets medicos

## 10. Limitaciones

- Las convoluciones tienen un campo receptivo limitado (local), no capturan dependencias globales de la imagen
- Puede presentar sobreajuste si no se regulariza adecuadamente
- Menos eficiente computacionalmente que arquitecturas mas modernas como EfficientNet
