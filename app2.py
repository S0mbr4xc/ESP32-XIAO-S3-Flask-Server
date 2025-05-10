import numpy as np
import cv2
import os
from flask import Flask, render_template, Response

app = Flask(__name__)

# Directorio donde se encuentran las imágenes médicas
IMAGE_DIR = 'img/'

# Lista de imágenes médicas para mostrar
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.dcm'))]

# Cargar una imagen médica
def load_image(image_path):
    return cv2.imread(image_path)

# Operaciones morfológicas

def apply_erosion(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_dilation(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_top_hat(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def apply_black_hat(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

def add_title(image, title):
    """Añade un título sobre la imagen."""
    output = image.copy()
    cv2.putText(output, title, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (0, 255, 255), 3, cv2.LINE_AA)  # Cambio la posición vertical a 50
    return output

def generate_video_stream_with_morphological_operations(image_file):
    # Cargar la imagen médica
    image_path = os.path.join(IMAGE_DIR, image_file)
    frame = load_image(image_path)

    # Aplicar operaciones morfológicas con diferentes tamaños de máscara
    kernel_sizes = [(3, 3), (15, 15), (37, 37)]
    
    results = []
    
    # Agregar las imágenes procesadas para cada kernel
    for kernel_size in kernel_sizes:
        # Erosión
        erosion = apply_erosion(frame, kernel_size)
        results.append((f'Erosion_{kernel_size}', erosion))
        
        # Dilatación
        dilation = apply_dilation(frame, kernel_size)
        results.append((f'Dilation_{kernel_size}', dilation))
        
        # Top Hat
        top_hat = apply_top_hat(frame, kernel_size)
        results.append((f'TopHat_{kernel_size}', top_hat))
        
        # Black Hat
        black_hat = apply_black_hat(frame, kernel_size)
        results.append((f'BlackHat_{kernel_size}', black_hat))

        # Imagen Original + (Top Hat - Black Hat)
        top_hat_minus_black_hat = cv2.subtract(top_hat, black_hat)
        results.append((f'Original_plus_TopHat_minus_BlackHat_{kernel_size}', cv2.add(frame, top_hat_minus_black_hat)))

    # Preparar las imágenes para visualización con el título
    imgs = []
    for title, img in results:
        imgs.append(add_title(img, title))

    # Redimensionar imágenes para la rejilla
    resized_imgs = [cv2.resize(img, (300, 220)) for img in imgs]

    # Armar la rejilla 5x3 (5 filas, 3 columnas)
    # Si faltan imágenes, añadir imágenes vacías
    empty = np.zeros_like(resized_imgs[0])
    while len(resized_imgs) < 15:
        resized_imgs.append(add_title(empty, "Vacio"))

    fila1 = np.hstack(resized_imgs[0:3])
    fila2 = np.hstack(resized_imgs[3:6])
    fila3 = np.hstack(resized_imgs[6:9])
    fila4 = np.hstack(resized_imgs[9:12])
    fila5 = np.hstack(resized_imgs[12:15])
    
    grid = np.vstack([fila1, fila2, fila3, fila4, fila5])

    flag, encoded = cv2.imencode(".jpg", grid)
    if not flag:
        return None

    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded) + b'\r\n')


@app.route("/")
def index():
    return render_template("index2.html", image_files=image_files)

@app.route("/video_stream/<image_file>")
def video_stream(image_file):
    return Response(generate_video_stream_with_morphological_operations(image_file),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
