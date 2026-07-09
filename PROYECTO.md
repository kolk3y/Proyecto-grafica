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
│   ├── val.py                 # Script principal (línea de comandos)
│   ├── Voxel_method.ipynb     # Notebook con automatización de experimentos y métricas
│   ├── models.py              # VolumeModel
│   ├── datasets.py            # Carga de datos
│   ├── losses.py              # Funciones de pérdida
│   └── utills.py              # Renderers y utilidades
│
├── mesh/                      # Pipeline de optimización mesh
│   ├── val.py                 # Script principal (línea de comandos)
│   ├── Mesh_method.ipynb      # Notebook con automatización de experimentos, métricas y visualización
│   ├── datasets.py            # Carga de datos
│   ├── losses.py              # Funciones de pérdida
│   └── utills.py              # Renderers y utilidades
│
├── media/                     # GIFs y figuras del paper original
├── voxel_results/             # Outputs del pipeline voxel (subcarpetas por experimento ignoradas por git; los .csv de métricas sí se versionan)
├── mesh_results/              # Outputs del pipeline mesh (subcarpetas por experimento ignoradas por git; los .csv de métricas sí se versionan)
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

15 siluetas en `data/custom_silhouettes/processed/`, clasificadas por complejidad geométrica medida con **compacidad** (4π·área/perímetro², donde 1.0 = círculo perfecto):

| Nivel | Archivo | Compacidad |
|---|---|---|
| Simple | s1_circulo.png | 0.91 |
| Simple | s2_rectangulo.png | 0.76 |
| Simple | s3_cruz.png | 0.51 |
| Simple | s4_flecha.png | 0.49 |
| Simple | s5_house.png | 0.48 |
| Media | m1_bird.png | 0.42 |
| Media | m2_tree.png | 0.39 |
| Media | m3_fish.png | 0.34 |
| Media | m4_star.png | 0.28 |
| Media | m5_guitar.png | 0.24 |
| Alta | h1_sitting_cat.png | 0.22 |
| Alta | h2_running_person.png | 0.20 |
| Alta | h3_butterfly.png | 0.11 |
| Alta | h4_snowflake.png | 0.08 |
| Alta | h5_bike.png | 0.02 |

> Valores recalculados tras el fix de `binarize.py` (commit `4f804d1`), que corrigió la binarización de PNGs con canal alfa engañoso (100% opaco). Esto reclasificó `h3`/`h4`: la mariposa (antes `h4_butterfly`) pasó a `h3_butterfly` y el copo de nieve (antes `h3_snowflake`) pasó a `h4_snowflake`, ya que la nueva medición da mayor compacidad a la mariposa.

#### Proceso de binarización

Las 5 siluetas simples se generaron programáticamente con OpenCV (`generate_simple.py`). Las 10 restantes se obtuvieron de [openclipart.org](https://openclipart.org) (licencia CC0) y se procesaron con `binarize.py`:

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

**Importante:** todos los experimentos usan `mirror_mode=0` (desactivado) para que los resultados sean comparables entre voxel y mesh.

### Pipeline Voxel

Ver `voxel/Voxel_method.ipynb` para la versión automatizada con todos los experimentos.

```bash
cd voxel/
conda run --no-capture-output -n shadowart python val.py \
    cuda:0 <exp_id> 600 0.01 \
    -mr 0 -swt 10.0 -l1wt 10.0 \
    -sdlist <img1.png> <img2.png> ...
```

**Tiempo estimado:** ~9 min con 2 vistas en GTX 1080.
**Output:** `voxel_results/<exp_id>/` con `.obj`, `.npy`, `.gif` y `log.txt`.

### Pipeline Mesh

Ver `mesh/Mesh_method.ipynb` para la versión automatizada con todos los experimentos (kernel `shadowart`). También existe `val.py` para correr un experimento suelto por línea de comandos:

```bash
cd mesh/
conda run --no-capture-output -n shadowart python val.py \
    cuda:0 <exp_id> 2000 0.15 0 \
    -mr 0 -swt 1.6 -l1wt 1.6 -mwt 0.0 -ewt 1.6 -nwt 0.6 -lwt 1.2 \
    -sdlist <img1.png> <img2.png> ...
```

**Tiempo estimado:** ~4-4.5 min por vista con `NITER=2000` en GTX 1080 (ej. un experimento de 3 vistas tarda ~12-13 min).
**Output:** `mesh_results/<exp_id>/` con `.obj`, `.gif`, `log.txt` y las siluetas GT/predicción por vista (`sample_0view_N.png`, `sample_0_pred_view_N.png`).

> Las siluetas se pasan con ruta relativa al directorio donde se corre el script.
> Para siluetas propias usar ruta absoluta, por ejemplo:
> `/mnt/c/Users/.../data/custom_silhouettes/processed/s1_circulo.png`

### Métricas y visualización (notebooks)

Ambos notebooks (`voxel/Voxel_method.ipynb`, `mesh/Mesh_method.ipynb`) calculan IoU, Dice, Precision, Recall y ROI Pixel Accuracy por vista a partir de las imágenes ya guardadas (no requieren reentrenar) y exportan un CSV consolidado con las columnas `experimento, view, IoU, Dice, Precision, Recall, ROI_PA` (`view` = nombre de la silueta, no índice):

- `voxel_results/metricas_voxel.csv`
- `mesh_results/metricas_consolidadas.csv` (experimentos base) y `mesh_results/metricas_espejo_paper.csv` (experimentos espejo)

`Mesh_method.ipynb` incluye además `visualize_silhouettes(exp_id)`, que genera una figura por experimento con tres filas (Silueta original / Sombra obtenida / Diferencia) y una columna por vista, con el Recall de cada una — mismo formato usado en las figuras de Voxel.

---

## Plan de experimentos

Todos los experimentos se corren en **ambos pipelines** (voxel y mesh) para habilitar comparación directa.

### Experimentos con dataset propio (siluetas propias)

| Exp ID | Vistas | Siluetas |
|---|---|---|
| 2-simple-simple | 2 | s1_circulo, s3_cruz |
| 2-simple-alta | 2 | s2_rectangulo, h2_running_person |
| 2-alta-alta | 2 | h4_snowflake, h5_bike |
| 5-simple | 5 | s1_circulo, s2_rectangulo, s3_cruz, s4_flecha, s5_house |
| 5-media | 5 | m1_bird, m2_tree, m3_fish, m4_star, m5_guitar |
| 5-alta | 5 | h1_sitting_cat, h2_running_person, h4_snowflake, h3_butterfly, h5_bike |
| 5-2simple-2media-1alta | 5 | s1_circulo, s5_house, m1_bird, m5_guitar, h4_snowflake |
| 10-5simple-5media | 10 | s1–s5 + m1–m5 |
| 10-5media-5alta | 10 | m1–m5 + h1–h5 |

### Experimentos baseline (dataset original del paper)

| Exp ID | Vistas | Siluetas |
|---|---|---|
| 2-mikey-puma | 2 | mikey, puma |
| 3-heroes | 3 | Spider-Man, superman, batman |
| 3-bunny-teddy-duck | 3 | bunny_0, teddy2, duck |
| 3-mikey-puma-heart | 3 | mikey, puma, heart |

### Experimentos espejo de generalización

Prueban combinaciones de complejidad no vistas en el resto del plan (mismos nombres/imágenes en ambos pipelines).

| Exp ID | Vistas | Siluetas |
|---|---|---|
| 2a-media-alta | 2 | m1_bird, h2_running_person |
| 3b-1media-2alta | 3 | m5_guitar, h1_sitting_cat, h3_butterfly |
| 3c-2simple-1media | 3 | s5_house, s4_flecha, m3_fish |
| 3d-simple-media-alta | 3 | s2_rectangulo, m1_bird, h2_running_person |

---

