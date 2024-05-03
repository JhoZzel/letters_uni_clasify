import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from skimage import io, transform
import numpy as np

app = Flask(__name__)

# Ruta al modelo .h5 (relativa al directorio actual)
model_path = os.path.join(os.path.dirname(__file__), 'modelo.h5')
model = load_model(model_path)  # Carga el modelo previamente entrenado

@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = io.imread(file)
            img = transform.resize(img, (28, 28))  # Ajusta el tamaño de la imagen según lo que espera tu modelo
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            classes = ['U', 'N', 'I']
            result = classes[predicted_class]
            return render_template("predict.html", prediction=result)
        else:
            return "No se ha enviado ninguna imagen."

if __name__ == "__main__":
    app.run(debug=True)
