
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import pandas as pd
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

#################################################
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload():
        file = request.files['file']
        filename = secure_filename(file.filename)    
        up="UPLOAD_FOLDER"
        if up not in os.listdir('.'):
            os.mkdir("UPLOAD_FOLDER")
        save_location = os.path.join('UPLOAD_FOLDER', filename)
        file.save(save_location)
        image = Image.open(save_location)
#resize image to 28x28 pixels
        img = image.resize((28,28))
#convert rgb to grayscale
        img = img.convert('L')
        img = np.invert(img)
        img = np.array(img)
        X = img.reshape(1, 784) 
#Normalization of data
#from 0-255 to 0-1 
        X = X / 255
# Reshape image in 3D
        X = X.reshape(-1,28,28,1)
# load the model
        model = load_model('model_1.h5')
# use model to predict
        y_pred = model.predict(X)
        y_predicted = np.argmax(y_pred, 1)
        y=int(y_predicted[0])
        return jsonify({'prediction': y})  
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)