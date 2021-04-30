import tensorflow as tf

from flask import Flask, request, render_template
# from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
import random
import os
# from tensorflow.python.keras.layers.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

model = tf.keras.models.load_model("./model/model_keras.h5")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


def predict_single(img_path):
    img = np.array(cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), (150,150), interpolation=cv2.INTER_CUBIC)).reshape((1, 150, 150, 3))
    return model.predict(img)

def decode_labels(target, thresh=0.5):
    labels = {
        0: ['crimp', 'crimpy boi', 'rip fingers'],
        1: ['jug', 'victory jug'],
        2: ['sloper', 'sloppy', 'sloppy boi', 'slop'],
        3: ['pinch', 'piiiinch', 'pinchy boi'],
        4: ['pocket', 'not a sloper'],
        5: ['edge']
    }
    return random.choice(labels[np.argmax(target)])



"""
ENDPOINTS BELOW
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict', methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        print(request.files)
        f = request.files["file"]

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", f.filename)
        f.save(file_path)

        preds = predict_single(file_path)
        result = decode_labels(preds)
        print(result)
        return result
    return None


if __name__ == '__main__':
    app.debug = True
    app.run()