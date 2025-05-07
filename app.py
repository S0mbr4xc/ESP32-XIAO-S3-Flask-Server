# Author: vlarobbyk
# Version: 1.1
# Date: 2025/05/07
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.
# Esteban Cordova

from flask import Flask, render_template, Response
from io import BytesIO
import time
import cv2
import numpy as np
import requests

app = Flask(__name__)

# Dirección del stream de la ESP32-CAM
_URL = 'http://192.168.101.47'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])

# Detector de fondo global
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)

def get_stream():
    return requests.get(stream_url, stream=True)

def decode_frame(chunk):
    if len(chunk) > 100:
        img_data = BytesIO(chunk)
        return cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
    return None

def calculate_fps(start_time, frame_count):
    elapsed = time.time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0

def detect_motion(gray):
    return background_subtractor.apply(gray)

def overlay_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def apply_histogram_equalization(gray):
    return cv2.equalizeHist(gray)

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_gamma_correction(gray, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(gray, table)

def extract_moving_regions(frame, mask):
    return cv2.bitwise_and(frame, frame, mask=mask)

def add_title(image, title):
    """Añade un título sobre la imagen."""
    output = image.copy()
    cv2.putText(output, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return output

def generate_video_stream():
    stream = get_stream()
    frame_count = 0
    start_time = time.time()

    for chunk in stream.iter_content(chunk_size=100000):
        frame = decode_frame(chunk)
        if frame is None:
            continue

        frame_count += 1
        fps = calculate_fps(start_time, frame_count)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = detect_motion(gray)
        hist_eq = apply_histogram_equalization(gray)
        clahe = apply_clahe(gray)
        gamma = apply_gamma_correction(gray)
        fusion = cv2.addWeighted(hist_eq, 0.33, clahe, 0.33, 0)
        fusion = cv2.addWeighted(fusion, 0.66, gamma, 0.34, 0)
        moving_region = extract_moving_regions(frame, mask)

        overlay_fps(frame, fps)

        # Títulos
        imgs = [
            add_title(frame, "Original + FPS"),
            add_title(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "Movimiento (Mascara)"),
            add_title(cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR), "Ecualización Hist."),
            add_title(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), "CLAHE"),
            add_title(cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR), "Correccion Gamma"),
            add_title(cv2.cvtColor(fusion, cv2.COLOR_GRAY2BGR), "Combinado Final"),
            add_title(moving_region, "Solo Movimiento (AND)")
        ]

        # Redimensionar imágenes para rejilla
        resized_imgs = [cv2.resize(img, (300, 220)) for img in imgs]

        # Armar rejilla 2x4 (rellenamos con imagen negra si falta una)
        empty = np.zeros_like(resized_imgs[0])
        while len(resized_imgs) < 8:
            resized_imgs.append(add_title(empty, "Vacio"))

        fila1 = np.hstack(resized_imgs[:4])
        fila2 = np.hstack(resized_imgs[4:])
        grid = np.vstack([fila1, fila2])

        flag, encoded = cv2.imencode(".jpg", grid)
        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
