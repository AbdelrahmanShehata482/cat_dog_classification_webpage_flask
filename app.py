from __future__ import division, print_function
import cv2
import os
import numpy as np
import tensorflow as tf



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Load your trained model
model= tf.keras.models.load_model("C:\\Users\\AL-alamia\\Documents\\VS Code\\Projects\\flask coding\\AI3\\ai.h5")


#function for processing the input image abd prediction
def model_predict(img_path, model):
    test_img = cv2.imread(img_path)
    test_img = cv2.resize(test_img,(256,256))
    test_input = test_img.reshape((1,256,256,3))
    y=model.predict(test_input)
    print(y)
    return y[0,0]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
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
        

        # Process your result for human
        dic= {0:"Cat", 1:"Dog"}
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = dic[preds]   # ImageNet Decode
        # return the result
        return pred_class
    return None


if __name__ == '__main__':
    app.run(debug=True, port=5005)

