"""
Convierte imágenes crudas a siluetas binarias 512x512 para el pipeline de Shadow Art.

Uso:
    python binarize.py                        # procesa todo lo que haya en raw/
    python binarize.py raw/mi_imagen.png      # procesa una imagen específica

La imagen de entrada puede ser:
  - PNG con fondo transparente (alfa = 0)   → usa el canal alfa como máscara
  - PNG/JPG con fondo blanco sobre negro    → aplica threshold OTSU
  - PNG/JPG con fondo claro                 → invierte y aplica threshold

Salida: processed/<nombre>.png (512x512, fondo negro, figura blanca)
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path


OUTPUT_SIZE = 512
PADDING = 0.15  # porcentaje de margen alrededor de la figura


def binarize_image(img_path: str) -> np.ndarray:
    """Convierte una imagen a silueta binaria 512x512."""
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer: {img_path}")

    # --- Paso 1: obtener máscara binaria ---
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        alpha = img_bgr[:, :, 3]
        # VERIFICACIÓN: ¿Realmente hay transparencia en esta imagen?
        if alpha.min() < 255: 
            # Sí, hay píxeles transparentes. Confiamos en el alfa.
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        else:
            # Es un PNG engañoso: tiene canal alfa pero es 100% opaco.
            # Ignoramos el alfa y usamos los canales de color (BGR).
            gray = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(mask) > 127:
                mask = cv2.bitwise_not(mask)
    else:
        # Sin alfa (JPG, BMP, o PNG de 24 bits): convertir a gris y aplicar OTSU
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask) > 127:
            mask = cv2.bitwise_not(mask)

    # --- Paso 2: limpiar ruido pequeño ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Paso 3: recortar al bounding box de la figura ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se encontró figura en la imagen")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Ampliar un poco el recorte
    pad_x = int(w * PADDING)
    pad_y = int(h * PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(mask.shape[1], x + w + pad_x)
    y2 = min(mask.shape[0], y + h + pad_y)
    crop = mask[y1:y2, x1:x2]

    # --- Paso 4: redimensionar a 512x512 manteniendo aspect ratio ---
    h_crop, w_crop = crop.shape
    scale = (OUTPUT_SIZE * (1 - 2 * PADDING)) / max(h_crop, w_crop)
    new_w = int(w_crop * scale)
    new_h = int(h_crop * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centrar en canvas negro
    canvas = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    y_off = (OUTPUT_SIZE - new_h) // 2
    x_off = (OUTPUT_SIZE - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Rebinarizar para asegurar que sea estrictamente 0/255
    _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)

    return canvas


def compute_complexity(mask: np.ndarray) -> dict:
    """Calcula métricas de complejidad geométrica para el análisis."""
    binary = (mask > 127).astype(np.uint8) * 255

    area = (binary > 0).sum()
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    # Compacidad: 4π·área / perímetro² (1 = círculo perfecto, < 1 = más complejo)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Componentes conexas (huecos y partes separadas)
    _, labels = cv2.connectedComponents((binary > 0).astype(np.uint8))
    n_components = labels.max()

    # Relación de aspecto
    if contours:
        ext_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if ext_contours:
            x, y, w, h = cv2.boundingRect(max(ext_contours, key=cv2.contourArea))
            aspect_ratio = max(w, h) / max(min(w, h), 1)
        else:
            aspect_ratio = 1.0
    else:
        aspect_ratio = 1.0

    return {
        "area_pct": round(area / (mask.shape[0] * mask.shape[1]) * 100, 1),
        "compactness": round(compactness, 4),
        "n_components": int(n_components),
        "aspect_ratio": round(aspect_ratio, 2),
    }


def process_file(src_path: str, out_dir: str) -> None:
    name = Path(src_path).stem + ".png"

    try:
        result = binarize_image(src_path)
        if out_dir != "":
            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, result)
        metrics = compute_complexity(result)
        print(f"  OK  {name:30s} area={metrics['area_pct']:5.1f}%  "
              f"compactness={metrics['compactness']:.4f}  "
              f"components={metrics['n_components']}  "
              f"aspect={metrics['aspect_ratio']:.2f}")
    except Exception as e:
        print(f"  ERR {name}: {e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #raw_dir = os.path.join(script_dir, "raw") # Preprocesar imagenes para binarizarlas y ajustar su tamaño a 512x512
    #raw_dir = "data/viewsdataset" # Evaluar compacidad de las imagenes del dataset
    raw_dir = "data/custom_silhouettes/processed" # Evaluar compacidad de las imagenes nuevas
    # out_dir = os.path.join(script_dir, "processed") # Dejar como out_dir = "" para obtener solo las metricas
    out_dir = ""
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    if len(sys.argv) > 1:
        # Procesar archivos específicos
        files = sys.argv[1:]
    else:
        # Procesar todo en raw/
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                 if Path(f).suffix.lower() in exts]
        files.sort()

    if not files:
        print(f"No hay imágenes en {raw_dir}/")
        print("Pon tus imágenes ahí y vuelve a correr el script.")
        return

    print(f"\nProcesando {len(files)} imagen(es)...\n")
    for f in files:
        process_file(f, out_dir)

if __name__ == "__main__":
    main()
