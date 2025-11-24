# Offside_Detection_System

#  Detector de Offside con IA

Sistema de visión artificial para detección automática de posiciones de offside en fútbol, utilizando YOLO para detección de jugadores y CLIP para clasificación de equipos.

##  Descripción

Este proyecto implementa un pipeline completo de visión por computadora que:

1. **Detecta jugadores** en imágenes de partidos de fútbol usando YOLOv8
2. **Clasifica equipos** automáticamente mediante CLIP (Vision Transformer)
3. **Calcula homografía** para proyectar posiciones de imagen a coordenadas reales del campo
4. **Permite selección manual** del jugador que marca la línea de offside
5. **Determina automáticamente** qué jugadores están en posición adelantada
6. **Genera visualizaciones** y métricas cuantitativas del análisis

##  Características

- Detección robusta de jugadores con modelo YOLO personalizado
- Clasificación automática de equipos sin entrenamiento previo (CLIP)
- Transformación perspectiva campo-imagen mediante homografía
- Identificación automática de porteros por posición
- Análisis de offside con selección interactiva
- Métricas cuantitativas: histogramas, distribuciones, heatmaps
- Preprocesamiento adaptativo de imágenes

## Tecnologías Utilizadas

- **YOLOv8** (Ultralytics) - Detección de objetos
- **CLIP** (OpenAI) - Clasificación de equipos por color de camiseta
- **OpenCV** - Procesamiento de imágenes y homografía
- **scikit-learn** - Clustering K-means
- **PyTorch** - Backend de CLIP
- **Matplotlib/Seaborn** - Visualizaciones y métricas

## Requisitos
```bash
ultralytics
opencv-python-headless
scikit-learn
pillow
matplotlib
seaborn
torch
clip (OpenAI)
numpy
```

## Cómo ejecutar

### En Google Colab (Recomendado)

1. **Sube tu modelo entrenado** `best.pt` a `/content/`

2. **Copia el código completo** en una celda de Colab

3. **Ejecuta la celda** - El sistema te pedirá:
   - Subir imágenes del partido
   - Seleccionar el jugador que marca la línea de offside

4. **Visualiza los resultados** - El sistema mostrará:
   - Imagen preprocesada
   - Jugadores detectados y numerados
   - Clasificación por equipos
   - Histograma de confianzas de detección
   - Distribución de equipos
   - Colores dominantes de camisetas
   - Mapa de calor de posiciones en el campo
   - Resultado final con línea de offside

## Pipeline de Procesamiento
```
1. Preprocesamiento
   └─> Mejora adaptativa de calidad de imagen

2. Homografía
   └─> Cálculo de transformación perspectiva campo-imagen

3. Detección YOLO
   └─> Identificación de todos los jugadores
   └─> Generación de histograma de confianzas

4. Visualización Inicial
   └─> Jugadores numerados sin clasificar

5. Clasificación CLIP
   └─> Agrupación automática en equipos
   └─> Análisis de distribución y balance
   └─> Visualización de colores dominantes

6. Cálculo de Posiciones
   └─> Proyección a coordenadas reales del campo
   └─> Identificación de porteros
   └─> Generación de heatmap de posiciones

7. Selección Manual
   └─> Usuario indica jugador que marca línea de offside

8. Análisis de Offside
   └─> Determinación automática de jugadores en posición adelantada
   └─> Análisis por dirección de ataque

9. Resultado Final
   └─> Visualización completa con línea de offside
   └─> Estadísticas y estado final
```

## Código de Colores

- **Verde** - Equipo 0
- **Rojo** - Equipo 1
- **Naranja** - Jugador que marca la línea de offside
- **Magenta** - Jugadores en posición de offside
- **Gris** - Jugadores sin clasificar

## Métricas Generadas

1. **Histograma de Confianzas** - Distribución de confianza de detecciones YOLO
2. **Distribución de Equipos** - Gráfico de barras con balance de jugadores
3. **Colores Dominantes** - Visualización del color promedio de cada equipo
5. **Estadísticas de Detección** - Media, mediana, desviación estándar

## Configuración

### Ajustar confianza de detección

Para ajustar la confianza simplemente debemos cambiar el valor

```python
persons = detector.detect(frame, conf=0.25)  
```

### Modificar dimensiones del campo
```python
homography = RobustHomography(field_length=105.0, field_width=68.0)
```

### Cambiar modelo YOLO
```python
detector = PlayerDetector(model_path="/ruta/a/tu/modelo.pt")
```

## Notas

- El modelo `best.pt` es una version re-entrenada de YOLO11 utilizada para este programa
- Las clases detectadas dependen de cómo fue entrenado el modelo utilizado
- La homografía funciona mejor con imágenes desde ángulos elevados
- La clasificación CLIP funciona mejor con camisetas de colores contrastantes






