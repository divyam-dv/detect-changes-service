
from flask import Flask, request, jsonify
import cv2
import numpy as np
import urllib.request
import tempfile
import os

app = Flask(__name__)

def load_image_from_url(url):
    try:
        with urllib.request.urlopen(url) as resp:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            return cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None

def create_polygon_mask(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

def detect_changes(baseline_image, latest_image, polygon_points):
    mask = create_polygon_mask(baseline_image, polygon_points)
    diff = cv2.absdiff(baseline_image, latest_image)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_masked = cv2.bitwise_and(diff_gray, diff_gray, mask=mask)
    _, thresh = cv2.threshold(diff_masked, 30, 255, cv2.THRESH_BINARY)
    return thresh

@app.route('/detect_changes', methods=['POST'])
def detect_changes_api():
    try:
        data = request.json
        baseline_url = data.get('baseline_url')
        latest_url = data.get('latest_url')
        polygon_points = data.get('polygon_points', [])

        if not baseline_url or not latest_url:
            return jsonify({"error": "Both baseline_url and latest_url are required."}), 400

        baseline_image = load_image_from_url(baseline_url)
        latest_image = load_image_from_url(latest_url)

        if baseline_image is None or latest_image is None:
            return jsonify({"error": "Failed to load one or both images from URLs."}), 400

        result = detect_changes(baseline_image, latest_image, polygon_points)

        _, temp_file_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(temp_file_path, result)

        return jsonify({
            "message": "Change detection completed",
            "overlay_image_path": temp_file_path
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
