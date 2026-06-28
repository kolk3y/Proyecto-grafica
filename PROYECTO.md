# Esculturas que Engañan a la Luz: Replicación y Evaluación de Shadow Art mediante Renderizado Diferenciable

Replicación y extensión del paper [Shadow Art Revisited: A Differentiable Rendering Based Approach (WACV 2022)](https://openaccess.thecvf.com/content/WACV2022/papers/Sadekar_Shadow_Art_Revisited_A_Differentiable_Rendering_Based_Approach_WACV_2022_paper.pdf) de Sadekar et al.

El objetivo es optimizar un objeto 3D mediante renderizado diferenciable para que sus proyecciones de sombra coincidan con un conjunto de siluetas binarias objetivo.

## Integrantes

| Nombre | Tarea principal |
|---|---|
| Erick Lara | Instalación del entorno, pipeline voxel con dataset original, recolección y binarización de siluetas propias |
| Nicolás Medel Correa | Pipeline mesh con dataset original, experimentos con siluetas propias, análisis de variación por cantidad de fotos |
| Vicente Retamal | Evaluación perceptual, cálculo de métricas (IoU, pixel accuracy, tiempo de convergencia) |

---

## Estructura del proyecto

```
Proyecto-grafica/
│
├── data/
│   ├── viewsdataset/          # 15 siluetas del paper original (512x512 PNG)
│   └── custom_silhouettes/
│       ├── raw/               # Imágenes originales descargadas (ignoradas por git)
│       ├── processed/         # Siluetas binarizadas listas para usar (512x512 PNG)
│       ├── binarize.py        # Script de preprocesamiento
│       └── generate_simple.py # Genera las 5 siluetas simples programáticamente
│
├── voxel/                     # Pipeline de optimización voxel
│   ├── val.py                 # Script principal
│   ├── models.py              # VolumeModel
│   ├── datasets.py            # Carga de datos
│   ├── losses.py              # Funciones de pérdida
│   └── utills.py              # Renderers y utilidades
│
├── mesh/                      # Pipeline de optimización mesh
│   ├── val.py                 # Script principal
│   ├── datasets.py            # Carga de datos
│   ├── losses.py              # Funciones de pérdida
│   └── utills.py              # Renderers y utilidades
│
├── media/                     # GIFs y figuras del paper original
├── voxel_results/             # Outputs del pipeline voxel (ignorado por git)
├── mesh_results/              # Outputs del pipeline mesh (ignorado por git)
│
├── PROYECTO.md                # Este archivo
├── README.md                  # README original del repositorio
└── requirements.txt           # Dependencias originales del paper
```

---

## Setup del entorno

El entorno requiere Python 3.12, PyTorch 2.4.1 con CUDA 12.1 y pytorch3d 0.7.8. Se usa Miniconda para evitar conflictos de dependencias.

### Prerrequisitos

- GPU NVIDIA con al menos 8 GB VRAM (probado en GTX 1080)
- Miniconda instalado en `~/miniconda3`
- WSL2 (Ubuntu) o Linux nativo

### 1. Crear el entorno conda

```bash
conda create -n shadowart python=3.12
```

### 2. Instalar PyTorch con CUDA

```bash
conda install pytorch=2.4.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

> **Nota WSL2:** Si aparece el error `undefined symbol: iJIT_NotifyEvent`, degradar MKL:
> ```bash
> conda install mkl=2023.1.0 -c defaults
> ```

### 3. Instalar pytorch3d

```bash
conda install pytorch3d=0.7.8 -c pytorch3d -c fvcore -c conda-forge
```

### 4. Instalar dependencias restantes

```bash
pip install imageio matplotlib scikit-image tqdm opencv-contrib-python
pip install --force-reinstall "scikit-image==0.26.0"  # compatibilidad NumPy 2.x
```

### Verificar instalación

```bash
conda run -n shadowart python -c "
import torch, pytorch3d
print('torch:', torch.__version__)
print('pytorch3d:', pytorch3d.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## Dataset

### Dataset original (paper)

15 siluetas en `data/viewsdataset/`, todas 512×512 PNG con fondo negro y figura blanca. Generadas manualmente con herramienta de dibujo digital por los autores del paper.

```
duck.png        mikey.png       batman.png      bunny_0.png     bunny_135.png
heart.png       hand.png        like.png        puma.png        Spider-Man.png
superman.png    teddy2.png      victory.png     armadillo_0.png yo.png
```

### Dataset propio

15 siluetas en `data/custom_silhouettes/processed/`, clasificadas en tres niveles de complejidad geométrica medida por **compacidad** (4π·área/perímetro², donde 1.0 = círculo perfecto):

| Nivel | Siluetas | Compacidad aprox. |
|---|---|---|
| Simple | s1_circulo, s2_rectangulo, s3_estrella, s4_flecha, s5_cruz | 0.26 – 0.90 |
| Media | fish, bird, house, tree, sitting_cat | 0.21 – 0.79 |
| Alta | running_person, guitar, butterfly, bike, snowflake | 0.08 – 0.22 |

#### Proceso de binarización

Las 5 siluetas simples se generaron programáticamente con OpenCV. Las 10 restantes se obtuvieron de [openclipart.org](https://openclipart.org) (licencia CC0) y se procesaron con `data/custom_silhouettes/binarize.py`:

1. Extracción de máscara por canal alfa (si disponible) o umbralización de Otsu sobre escala de grises
2. Inversión automática para imágenes con figura oscura sobre fondo claro
3. Limpieza morfológica (apertura + cierre, kernel 5×5) para eliminar artefactos
4. Recorte al bounding box con margen del 15% y reescalado a 512×512

```bash
# Reproducir el dataset propio
cd data/custom_silhouettes
conda run -n shadowart python generate_simple.py   # genera las 5 simples
# Poner imágenes descargadas en raw/ y luego:
conda run -n shadowart python binarize.py          # procesa las 10 restantes
```

---

## Correr los pipelines

Todos los comandos se ejecutan desde la carpeta del pipeline correspondiente.

### Pipeline Voxel

Optimiza una cuadrícula de vóxeles 128³ mediante ray marching diferenciable.

```bash
cd voxel/
conda run --no-capture-output -n shadowart python val.py \
    cuda:0 \          # dispositivo
    <exp_id> \        # nombre del experimento (carpeta en voxel_results/)
    600 \             # iteraciones
    0.01 \            # learning rate
    -swt 10.0 \       # peso pérdida silueta
    -l1wt 10.0 \      # peso pérdida L1
    -sdlist <img1.png> <img2.png> ...   # siluetas objetivo (rutas relativas a voxel/)
```

**Ejemplo con dataset original:**
```bash
cd voxel/
conda run --no-capture-output -n shadowart python val.py cuda:0 duck_mikey 600 0.01 -swt 10.0 -l1wt 10.0 -sdlist duck.png mikey.png
```

**Tiempo estimado:** ~9 min con 2 vistas en GTX 1080.  
**Output:** `voxel_results/<exp_id>/` con `.obj`, `.npy`, `.gif` y `log.txt` con métricas.

### Pipeline Mesh

Optimiza los vértices de una esfera ico deformada mediante rasterización diferenciable.

```bash
cd mesh/
conda run --no-capture-output -n shadowart python val.py \
    cuda:0 \          # dispositivo
    <exp_id> \        # nombre del experimento (carpeta en mesh_results/)
    2000 \            # iteraciones
    0.15 \            # learning rate
    0 \               # model_id (siempre 0)
    -swt 1.6 \        # peso pérdida silueta
    -l1wt 1.6 \       # peso pérdida L1
    -mwt 0.0 \        # peso MS-SSIM
    -ewt 1.6 \        # peso pérdida de aristas
    -nwt 0.6 \        # peso pérdida de normales
    -lwt 1.2 \        # peso suavizado laplaciano
    -sdlist <img1.png> <img2.png> ...
```

**Ejemplo con dataset original:**
```bash
cd mesh/
conda run --no-capture-output -n shadowart python val.py cuda:0 duck_mikey 2000 0.15 0 -swt 1.6 -l1wt 1.6 -mwt 0.0 -ewt 1.6 -nwt 0.6 -lwt 1.2 -sdlist duck.png mikey.png
```

**Tiempo estimado:** ~11 min con 2 vistas en GTX 1080.  
**Output:** `mesh_results/<exp_id>/` con `.obj`, `.gif` y `log.txt` con métricas.

> **Importante:** las siluetas se pasan con ruta relativa al directorio donde se corre el script. Para usar imágenes de otro directorio, usar rutas absolutas:
> ```bash
> -sdlist /ruta/absoluta/a/img1.png /ruta/absoluta/a/img2.png
> ```

---

## Plan de experimentos

### Experimento A — Replicación baseline (dataset original)

Verificar que el método reproduce los resultados del paper con la configuración por defecto.

| Corrida | Siluetas | Pipeline | Iter |
|---|---|---|---|
| A-V1 | duck + mikey | Voxel | 600 |
| A-V2 | batman + heart | Voxel | 600 |
| A-V3 | bunny_0 + superman | Voxel | 600 |
| A-M1 | duck + mikey | Mesh | 2000 |
| A-M2 | batman + heart | Mesh | 2000 |
| A-M3 | bunny_0 + superman | Mesh | 2000 |

### Experimento B — Variación por cantidad de fotos

Mismo conjunto de siluetas base, variando el número de vistas N ∈ {2, 5, 10}.

| Corrida | Vistas | Siluetas | Pipeline |
|---|---|---|---|
| B-V-2 | 2 | duck, mikey | Voxel |
| B-V-5 | 5 | duck, mikey, batman, heart, like | Voxel |
| B-V-10 | 10 | duck, mikey, batman, heart, like, superman, bunny_0, teddy2, puma, hand | Voxel |
| B-M-2 | 2 | duck, mikey | Mesh |
| B-M-5 | 5 | duck, mikey, batman, heart, like | Mesh |
| B-M-10 | 10 | duck, mikey, batman, heart, like, superman, bunny_0, teddy2, puma, hand | Mesh |

### Experimento C — Siluetas propias por complejidad

Evaluar generalización del método fuera del dataset original.

| Corrida | Complejidad | Ejemplo de siluetas | Pipeline |
|---|---|---|---|
| C-simple | Baja | s1_circulo + s3_estrella | Ambos |
| C-media | Media | fish + sitting_cat | Ambos |
| C-alta | Alta | bike + snowflake | Ambos |

---

## Métricas

Cada corrida genera un `log.txt` con:

| Métrica | Descripción |
|---|---|
| **IoU** | Intersection over Union entre sombra predicha y silueta objetivo |
| **Dice** | Coeficiente Dice (similar a IoU, más sensible a regiones pequeñas) |
| **MS-SSIM** | Multi-Scale Structural Similarity |
| **Edge loss** | Regularización de aristas (solo mesh) |
| **Laplacian loss** | Suavidad de la superficie (solo mesh) |
| **Normal loss** | Consistencia de normales (solo mesh) |

---

## Preguntas de investigación

1. ¿Qué características geométricas de una silueta (compacidad, huecos, aspect ratio) correlacionan con el IoU final?
2. ¿Existe un punto de saturación al agregar más vistas? ¿O más siempre es mejor?
3. ¿El modo óptimo (voxel vs. mesh) depende de la complejidad de la silueta?
4. ¿Hay correlación entre IoU y reconocimiento humano de las sombras?
