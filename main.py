import io
import base64

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# IMAGE LOADING
# ---------------------------------------------------------------------------

def load_image_from_bytes(file_bytes: bytes, max_size: int = 220) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    width, height = img.size
    largest = max(width, height)

    if max_size and largest > max_size:
        scale = max_size / largest
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)

    return np.array(img)


# ---------------------------------------------------------------------------
# K-MEANS CLUSTERING
# ---------------------------------------------------------------------------

def _kmeans_plusplus_init(pixels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = pixels.shape[0]
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = pixels[rng.integers(n)]

    closest_dist_sq = np.sum((pixels - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = closest_dist_sq.sum()
        probs = closest_dist_sq / total if total > 0 else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centers[c] = pixels[idx]
        new_dist_sq = np.sum((pixels - centers[c]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centers


def _assign(pixels: np.ndarray, px_sq: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = px_sq[:, None] - 2.0 * (pixels @ centers.T) + np.sum(centers ** 2, axis=1)[None, :]
    return np.argmin(d, axis=1)


def kmeans(pixels: np.ndarray, k: int, iterations: int = 25, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = _kmeans_plusplus_init(pixels, k, rng)
    px_sq = np.sum(pixels ** 2, axis=1)
    labels = np.zeros(pixels.shape[0], dtype=np.int64)

    for _ in range(iterations):
        labels = _assign(pixels, px_sq, centers)

        counts = np.bincount(labels, minlength=k).astype(np.float64)
        new_centers = centers.copy()
        nonempty = counts > 0
        for ch in range(3):
            sums = np.bincount(labels, weights=pixels[:, ch], minlength=k)
            new_centers[nonempty, ch] = sums[nonempty] / counts[nonempty]

        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    labels = _assign(pixels, px_sq, centers)
    inertia = float(np.sum((pixels - centers[labels]) ** 2))

    return labels, centers, inertia


def segment_image(image_rgb: np.ndarray, k: int, iterations: int = 25):
    height, width, _ = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3).astype(np.float64) / 255.0

    labels, centers, inertia = kmeans(pixels, k, iterations)

    centers_255 = np.clip(centers * 255, 0, 255).astype(np.uint8)
    segmented_image = centers_255[labels].reshape(height, width, 3)

    return segmented_image, centers_255, inertia, pixels.shape[0]


# ---------------------------------------------------------------------------
# GRID OVERLAY
# ---------------------------------------------------------------------------

def add_grid(image_rgb: np.ndarray, spacing: int) -> np.ndarray:
    if spacing <= 0:
        return image_rgb

    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    for x in range(0, width + 1, spacing):
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255), width=1)
    for y in range(0, height + 1, spacing):
        draw.line([(0, y), (width, y)], fill=(255, 255, 255), width=1)

    return np.array(img)


# ---------------------------------------------------------------------------
# CLUSTER LEGEND AND ERROR STATS
# ---------------------------------------------------------------------------

def cluster_legend(centers_255: np.ndarray) -> list:
    legend = []
    for idx, (r, g, b) in enumerate(centers_255.tolist()):
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        legend.append({"cluster": idx, "r": r, "g": g, "b": b, "luminance": round(luminance, 1)})
    return legend


def error_stats(inertia: float, pixel_count: int) -> dict:
    mean_error = inertia / pixel_count
    rmse_255 = float(np.sqrt(mean_error) * 255)
    return {
        "inertia": round(inertia, 6),
        "mean_error": round(mean_error, 8),
        "rmse_255": round(rmse_255, 2),
    }


# ---------------------------------------------------------------------------
# ELBOW GRAPH
# ---------------------------------------------------------------------------

def compute_elbow(image_rgb: np.ndarray, k_min: int, k_max: int,
                   sample_pixels: int = 3000, iterations: int = 15, seed: int = 42):
    all_pixels = image_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    n = all_pixels.shape[0]

    rng = np.random.default_rng(seed)
    if sample_pixels < n:
        idx = rng.choice(n, sample_pixels, replace=False)
        sample = all_pixels[idx]
    else:
        sample = all_pixels

    k_values, mean_errors = [], []
    for k in range(k_min, k_max + 1):
        if k > sample.shape[0]:
            break
        _, _, inertia = kmeans(sample, k, iterations, seed)
        k_values.append(k)
        mean_errors.append(inertia / sample.shape[0])

    return k_values, mean_errors


# ---------------------------------------------------------------------------
# ENCODING FOR HTTP RESPONSES
# ---------------------------------------------------------------------------

def image_to_base64_png(image_rgb: np.ndarray) -> str:
    img = Image.fromarray(image_rgb.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
