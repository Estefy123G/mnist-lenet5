"""
App web - Reconocimiento de dígitos MNIST (0-9)
Big Data - Unidad 2 (Diapositiva 22)
Sirve el mejor modelo de la comparación (LeNet-5) exportado a ONNX.
Como referencia también se incluye la CNN Keras (mayor accuracy, 27x más parámetros).
"""
import base64
import io
import os
import time

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
MODELOS = {
    "lenet5": ("LeNet-5", os.path.join(BASE, "modelo", "lenet5_mnist.onnx")),
    "cnn_keras": ("CNN Keras", os.path.join(BASE, "modelo", "cnn_keras_mnist.onnx")),
}
MODELO_POR_DEFECTO = "lenet5"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB
SESIONES = {
    clave: ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    for clave, (_, ruta) in MODELOS.items()
    if os.path.exists(ruta)
}


def preprocesar(img: Image.Image):
    """Convierte cualquier imagen a 28x28 en escala de grises, fondo negro y dígito blanco (formato MNIST)."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fondo, img.convert("RGBA"))
    img = img.convert("L")
    tamano_original = img.size
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    invertida = False
    # MNIST tiene fondo negro; si el fondo es claro (papel blanco), se invierte
    if arr.mean() > 127:
        arr = 255.0 - arr
        invertida = True
    return arr, tamano_original, invertida


def a_base64(arr: np.ndarray, escala: int = 10) -> str:
    img = Image.fromarray(arr.astype(np.uint8)).resize((28 * escala, 28 * escala), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    modelos = [(k, MODELOS[k][0]) for k in MODELOS if k in SESIONES]
    return render_template("index.html", modelos=modelos, por_defecto=MODELO_POR_DEFECTO)


@app.route("/predecir", methods=["POST"])
def predecir():
    archivo = request.files.get("imagen")
    if archivo is None or archivo.filename == "":
        return jsonify(error="No se recibió ninguna imagen."), 400
    try:
        img = Image.open(archivo.stream)
        img.load()
    except Exception:
        return jsonify(error="El archivo no es una imagen válida (use PNG, JPG, BMP...)."), 400

    clave = request.form.get("modelo", MODELO_POR_DEFECTO)
    if clave not in SESIONES:
        clave = MODELO_POR_DEFECTO
    sesion = SESIONES[clave]

    arr, tamano, invertida = preprocesar(img)
    x = (arr / 255.0).reshape(1, 28, 28, 1).astype(np.float32)
    t0 = time.perf_counter()
    prob = sesion.run(None, {sesion.get_inputs()[0].name: x})[0][0]
    t_ms = (time.perf_counter() - t0) * 1000
    digito = int(np.argmax(prob))
    return jsonify(
        digito=digito,
        confianza=round(float(prob[digito]) * 100, 2),
        probabilidades=[round(float(p) * 100, 2) for p in prob],
        imagen_28=a_base64(arr),
        tamano_original=f"{tamano[0]}×{tamano[1]}",
        invertida=invertida,
        modelo=MODELOS[clave][0],
        tiempo_ms=round(t_ms, 2),
    )


@app.route("/salud")
def salud():
    return jsonify(estado="ok", modelos=[MODELOS[k][0] for k in SESIONES])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
