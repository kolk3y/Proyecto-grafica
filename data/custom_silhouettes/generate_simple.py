"""
Genera las 5 siluetas 'simples' programáticamente (no requieren descarga).
Guarda directo en processed/.
"""

import cv2
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
SIZE = 512
os.makedirs(OUT_DIR, exist_ok=True)


def save(name, img):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, img)
    binary = (img > 127).astype(np.uint8)
    area_pct = binary.sum() / (SIZE * SIZE) * 100
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) if contours else 1
    compactness = (4 * np.pi * binary.sum()) / (perimeter ** 2) if perimeter > 0 else 0
    print(f"  {name:25s} area={area_pct:.1f}%  compactness={compactness:.4f}")


def canvas():
    return np.zeros((SIZE, SIZE), dtype=np.uint8)


cx, cy = SIZE // 2, SIZE // 2  # centro


# 1. Círculo — compacidad ≈ 1.0 (máxima, forma más simple)
img = canvas()
cv2.circle(img, (cx, cy), 190, 255, -1)
save("s1_circulo.png", img)


# 2. Rectángulo — compacidad moderada, ángulos rectos
img = canvas()
cv2.rectangle(img, (cx - 170, cy - 110), (cx + 170, cy + 110), 255, -1)
save("s2_rectangulo.png", img)


# 3. Estrella de 5 puntas — compacidad baja, muchos vértices
def star_points(cx, cy, r_outer, r_inner, n=5):
    pts = []
    for i in range(n * 2):
        angle = np.radians(-90 + i * 180 / n)
        r = r_outer if i % 2 == 0 else r_inner
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    return np.array(pts, dtype=np.int32)

img = canvas()
pts = star_points(cx, cy, 200, 80)
cv2.fillPoly(img, [pts], 255)
save("s3_estrella.png", img)


# 4. Flecha — asimétrica, compacidad media-baja
img = canvas()
arrow = np.array([
    [cx - 160, cy - 55], [cx + 40, cy - 55],
    [cx + 40, cy - 120], [cx + 170, cy],
    [cx + 40, cy + 120], [cx + 40, cy + 55],
    [cx - 160, cy + 55],
], dtype=np.int32)
cv2.fillPoly(img, [arrow], 255)
save("s4_flecha.png", img)


# 5. Cruz / más — forma con concavidades, compacidad media
img = canvas()
arm = 70
length = 175
pts = np.array([
    [cx - arm, cy - length], [cx + arm, cy - length],
    [cx + arm, cy - arm],    [cx + length, cy - arm],
    [cx + length, cy + arm], [cx + arm, cy + arm],
    [cx + arm, cy + length], [cx - arm, cy + length],
    [cx - arm, cy + arm],    [cx - length, cy + arm],
    [cx - length, cy - arm], [cx - arm, cy - arm],
], dtype=np.int32)
cv2.fillPoly(img, [pts], 255)
save("s5_cruz.png", img)


print("\nSimples generadas en:", OUT_DIR)
print("Ahora descarga las 10 restantes (medias y complejas) desde openclipart.org")
