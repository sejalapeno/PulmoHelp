from __future__ import division, print_function

# coding=utf-8
import os
import tensorflow as tf

import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

print("tensorflow ver")
print(tf.__version__)
# Flask utils
from flask import request, render_template
from flask import Flask
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)

@app.route("/")
def Index():
    return render_template('index.html')

@app.route("/Home")
def Home():
    return render_template('index.html')

@app.route("/About")
def About():
    return render_template('about.html')


@app.route("/Services")
def Services():
    return render_template('services.html')


@app.route("/Blog")
def Blog():
    return render_template('blog.html')


@app.route("/Contact")
def Contact():
    return render_template('index_profile.html')

@app.route("/Predict")
def Predict():
    return render_template('predict.html')


MODEL_PATH = 'D:\LDFlask\model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The Disease is Bacterial Pneumonia"
    elif preds == 1:
        preds = "The Disease is Corona virus"
    elif preds == 2:
        preds = "Your lungs are healthy!"
    elif preds == 3:
        preds = "The Disease is Tuberculosis"
    elif preds == 4:
        preds = "The Disease is Viral pneumonia"

    return preds



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


#app.run(debug=True)
if __name__ == '__main__':
    app.run(port=5001,debug=True)