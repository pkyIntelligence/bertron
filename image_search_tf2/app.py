import os
import numpy as np
import pickle
import config as cfg
from inference import simple_inference_with_color_filters
from flask import Flask, request, render_template, send_from_directory

import tensorflow as tf

#Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

with tf.device('CPU:0'):
    #Define our model
    model = tf.keras.models.load_model(os.path.join('saver', 'model_epoch_10'))

#Load training set vectors
with open('hamming_train_vectors.pickle', 'rb') as f:
    train_vectors = pickle.load(f)
    
#Load color vectors
with open('color_vectors.pickle', 'rb') as f:
    color_vectors = pickle.load(f)

#Load training set paths
with open('train_images_pickle.pickle', 'rb') as f:
    train_paths = pickle.load(f)

#Define Flask app
app = Flask(__name__, static_url_path='/static')

#Define apps home page
@app.route('/')
def index():
    return render_template('index.html')

#Define upload function
@app.route('/upload', methods=['POST'])
def upload():

    upload_dir = os.path.join(APP_ROOT, 'uploads/')

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist('file'):
        img_name = img.filename
        destination = '/'.join([upload_dir, img_name])

        img.save(destination)

    with tf.device('CPU:0'):
        #Inference
        result = np.array(train_paths)[simple_inference_with_color_filters(model, train_vectors,
                                            os.path.join(upload_dir, img_name), color_vectors, cfg.IMAGE_SIZE)]

    result_final = []

    for img in result:
        result_final.append('images/'+img.split('/')[-1]) #Just grabbing the image file name

    return render_template('result.html', image_name=img_name, result_paths=result_final)

#Define helper function for finding image paths
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('uploads', filename)

#Start the application

if __name__ == '__main__':
    app.run(port=5000, debug=True)