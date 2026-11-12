# App web — Reconocedor de dígitos MNIST (LeNet-5)

Aplicación Flask que permite **cargar y mostrar una imagen 28×28** y **predecir el dígito (0-9)** con el mejor modelo de la comparación (**LeNet-5**, por defecto), exportado a ONNX. En la interfaz se puede elegir también la CNN Keras para contrastar.

```
app_web/
├── app.py                 # Backend Flask (rutas /, /predecir, /salud)
├── templates/index.html   # Interfaz: cargar imagen, vista previa, predicción y probabilidades
├── modelo/lenet5_mnist.onnx     # modelo desplegado (el mejor)
├── modelo/cnn_keras_mnist.onnx  # alternativa seleccionable en la interfaz
├── requirements.txt       # flask, gunicorn, onnxruntime, numpy, pillow
├── Procfile               # Heroku / Railway
├── render.yaml            # Render (Blueprint)
└── .python-version        # 3.11
```

## Ejecutar localmente

```bash
cd app_web
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Abrir http://localhost:5000 y subir una imagen de la carpeta `../imagenes_prueba/`.

## Desplegar en Render.com (gratis)

1. Crear un repositorio en GitHub y subir **el contenido de la carpeta `app_web`** (que `app.py` quede en la raíz del repo).
   ```bash
   cd app_web
   git init
   git add .
   git commit -m "App MNIST LeNet-5"
   git branch -M main
   git remote add origin https://github.com/<usuario>/mnist-lenet5.git
   git push -u origin main
   ```
2. Entrar a https://render.com → **New + → Web Service** → conectar el repositorio.
3. Configuración (Render la detecta sola gracias a `render.yaml`):
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1`
   - **Instance type:** Free
4. **Create Web Service** y esperar el build (~2-3 min). La URL queda como `https://mnist-lenet5.onrender.com`.

> En el plan gratuito el servicio "se duerme" tras 15 min sin uso; la primera petición posterior tarda ~30-50 s en despertar.

## Desplegar en Railway.app (alternativa)

1. https://railway.app → **New Project → Deploy from GitHub repo** → elegir el repositorio.
2. Railway usa el `Procfile` automáticamente. En **Settings → Networking → Generate Domain** para obtener la URL pública.

## ¿Por qué ONNX y no TensorFlow en el servidor?

TensorFlow instalado ocupa ~500 MB y supera la RAM de los planes gratuitos (512 MB). `onnxruntime` ocupa ~15 MB y da **exactamente las mismas predicciones** que el modelo Keras (verificado en el notebook, sección 9). El modelo Keras original está en `../modelos/lenet5_mnist.keras`.
