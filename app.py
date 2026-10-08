"""
app.py — Flask web server.

This file only handles HTTP routing (serving the frontend, and exposing
/api/segment and /api/elbow). All of the actual image-segmentation work
happens in main.py.

A small in-memory cache keeps live slider updates snappy: changing only the
grid size re-uses the already-computed K-Means result, and the (unchanged)
original image is only sent back when the client asks for it.
"""
import os
import hashlib
from collections import OrderedDict

from flask import Flask, request, jsonify, send_from_directory

import main as segmentation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

CACHE_LIMIT = 32
_cache = OrderedDict()


def _cache_get(key):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key, value):
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_LIMIT:
        _cache.popitem(last=False)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/segment", methods=["POST"])
def api_segment():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file_bytes = request.files["image"].read()
    k = int(request.form.get("k", 4))
    max_size = int(request.form.get("max_size", 220))
    grid_size = int(request.form.get("grid_size", 30))
    need_original = request.form.get("need_original", "1") == "1"

    if k < 2:
        return jsonify({"error": "k must be at least 2."}), 400

    digest = hashlib.md5(file_bytes).hexdigest()

    img_key = ("img", digest, max_size)
    img_entry = _cache_get(img_key)
    if img_entry is None:
        image_rgb = segmentation.load_image_from_bytes(file_bytes, max_size)
        img_entry = {"rgb": image_rgb, "png": segmentation.image_to_base64_png(image_rgb)}
        _cache_put(img_key, img_entry)
    image_rgb = img_entry["rgb"]

    seg_key = ("seg", digest, max_size, k)
    seg_entry = _cache_get(seg_key)
    if seg_entry is None:
        segmented_image, centers_255, inertia, pixel_count = segmentation.segment_image(image_rgb, k)
        seg_entry = {
            "image": segmented_image,
            "legend": segmentation.cluster_legend(centers_255),
            "stats": segmentation.error_stats(inertia, pixel_count),
        }
        _cache_put(seg_key, seg_entry)

    segmented_with_grid = segmentation.add_grid(seg_entry["image"], grid_size)

    payload = {
        "width": int(image_rgb.shape[1]),
        "height": int(image_rgb.shape[0]),
        "segmented_png": segmentation.image_to_base64_png(segmented_with_grid),
        "legend": seg_entry["legend"],
        "error_stats": seg_entry["stats"],
    }
    if need_original:
        payload["original_png"] = img_entry["png"]
    return jsonify(payload)


@app.route("/api/elbow", methods=["POST"])
def api_elbow():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    max_size = int(request.form.get("max_size", 220))
    k_min = int(request.form.get("k_min", 2))
    k_max = int(request.form.get("k_max", 9))

    if k_max < k_min:
        return jsonify({"error": "k_max must be >= k_min."}), 400

    image_rgb = segmentation.load_image_from_bytes(image_file.read(), max_size)
    k_values, mean_errors = segmentation.compute_elbow(image_rgb, k_min, k_max)

    return jsonify({"k_values": k_values, "mean_errors": mean_errors})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
