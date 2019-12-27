import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf

#Allow GPU memory allocation to grow - CUDNN wasn't working otherwise
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tqdm import tqdm
from scipy.spatial.distance import hamming, cosine, euclidean


def image_loader(image_path, image_size):
    '''
    Load an image from a disk.

    :param image_path: String, path to the image
    :param image_size: tuple, size of an output image Example: image_size=(32, 32)
    '''

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, cv2.INTER_CUBIC)
    return image


def dataset_preprocessing(dataset_path, labels_file_path, image_size, image_paths_pickle, dtype=None):
    '''
    Loads images and labels from dataset folder.

    :param dataset_path: String, path to the train/test dataset folder
    :param labels_file_path: String, path to the .txt file where classes names are written
    :param image_size: tuple, single image size
    :param image_paths_pickle: String, name of a pickle file where all image paths will be saved
    '''

    with open(labels_file_path, 'r') as f:
        classes = f.read().split('\n')[:-1]

    images = []
    labels = []
    image_paths = []

    for image_name in os.listdir(dataset_path):
        try:
            image_path = os.path.join(dataset_path, image_name)
            images.append(image_loader(image_path, image_size))
            image_paths.append(image_path)
            for idx in range(len(classes)):
                if classes[idx] in image_name:  # Example: 0_frog.png
                    labels.append(idx)
        except:
            pass

    with open(image_paths_pickle + ".pickle", 'wb') as f:
        pickle.dump(image_paths, f)

    assert len(images) == len(labels)
    return np.array(images, dtype=dtype), np.array(labels, dtype=dtype)


def cosine_distance(training_set_vectors, query_vector, top_n=50):
    '''
    Calculates cosine distances between query image (vector) and all training set images (vectors).

    :param training_set_vectors: numpy Matrix, vectors for all images in the training set
    :param query_vector: numpy vector, query image (new image) vector
    :param top_n: integer, number of closest images to return
    '''

    distances = []

    for i in range(len(training_set_vectors)):  # For Cifar 10 -> 50k images
        distances.append(cosine(training_set_vectors[i], query_vector[0]))

    return np.argsort(distances)[:top_n]


def hamming_distance(training_set_vectors, query_vector, top_n=50):
    '''
    Calculates hamming distances between query image (vector) and all training set images (vectors).

    :param training_set_vectors: numpy Matrix, vectors for all images in the training set
    :param query_vector: numpy vector, query image (new image) vector
    :param top_n: Integer, number of closest images to return
    '''

    distances = []

    for i in range(len(training_set_vectors)):  # For Cifar 10 -> 50k images
        distances.append(hamming(training_set_vectors[i], query_vector[0]))

    return np.argsort(distances)[:top_n]


def sparse_accuracy(true_labels, predicted_labels):
    '''
    Calculates accuracy of a model based on softmax outputs.

    :param true_labels: numpy array, real labels of each sample. Example: [1, 2, 1, 0, 0]
    :param predicted_labels: numpy matrix, softmax probabilities. Example [[0.2, 0.1, 0.7], [0.9, 0.05, 0.05]]
    '''

    assert len(true_labels) == len(predicted_labels)

    correct = 0

    for i in range(len(true_labels)):

        if np.argmax(predicted_labels[i]) == true_labels[i]:
            correct += 1

    return correct / len(true_labels)


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, number_of_filters, kernel_size, strides=(1, 1),
                 padding='SAME', activation=tf.nn.relu,
                 max_pool=True, batch_norm=True):
        '''
        Defines convolutional block layer.

        :param number_of_filters: integer, number of conv filters
        :param kernel_size: tuple, size of conv layer kernel
        :param padding: string, type of padding technique: SAME or VALID
        :param activation: tf.object, activation function used on the layer
        :param max_pool: boolean, if true the conv block will use max_pool
        :param batch_norm: boolean, if true the conv block will use batch normalization
        '''
        super(ConvBlock, self).__init__()

        self._conv_layer = tf.keras.layers.Conv2D(filters=number_of_filters,
                                                  kernel_size=kernel_size,
                                                  strides=strides,
                                                  padding=padding,
                                                  activation=activation)

        self._max_pool = max_pool
        if max_pool:
            self._mp_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                       strides=(2, 2),
                                                       padding='SAME')

        self._batch_norm = batch_norm
        if batch_norm:
            self._bn_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):

        conv_features = x = self._conv_layer(inputs)
        if self._max_pool:
            x = self._mp_layer(x)
        if self._batch_norm:
            x = self._bn_layer(x, training)
        return x, conv_features


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, units, activation=tf.nn.relu,
                 dropout_rate=None, batch_norm=True):
        '''
        Defines dense block layer.

        :param units: integer, number of neurons/units for a dense layer
        :param activation: tf.object, activation function used on the layer
        :param dropout_rate: dropout rate used in this dense block
        :param batch_norm: boolean, if true the conv block will use batch normalization
        '''
        super(DenseBlock, self).__init__()

        self._dense_layer = tf.keras.layers.Dense(units, activation=activation)

        self._dropout_rate = dropout_rate
        if dropout_rate is not None:
            self._dr_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self._batch_norm = batch_norm
        if batch_norm:
            self._bn_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):

        dense_features = x = self._dense_layer(inputs)
        if self._dropout_rate is not None:
            x = self._dr_layer(x, training)
        if self._batch_norm:
            x = self._bn_layer(x, training)
        return x, dense_features


class ImageSearchModel(tf.keras.Model):

    def __init__(self, dropout_rate, image_size, number_of_classes=10):
        '''
        Defines CNN model.

        :param dropout_rate: dropout_rate
        :param learning_rate: learning_rate
        :param image_size: tuple, (height, width) of an image
        :param number_of_classes: integer, number of classes in a dataset.
        '''
        super(ImageSearchModel, self).__init__()

        self._bn_layer = tf.keras.layers.BatchNormalization()

        self._conv_block_1 = ConvBlock(number_of_filters=64,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._conv_block_2 = ConvBlock(number_of_filters=128,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._conv_block_3 = ConvBlock(number_of_filters=256,
                                       kernel_size=(5, 5),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._conv_block_4 = ConvBlock(number_of_filters=512,
                                       kernel_size=(5, 5),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._flatten_layer = tf.keras.layers.Flatten()

        self._dense_block_1 = DenseBlock(units=128,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._dense_block_2 = DenseBlock(units=256,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._dense_block_3 = DenseBlock(units=512,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._dense_block_4 = DenseBlock(units=1024,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._final_dense = tf.keras.layers.Dense(units=number_of_classes,
                                                  activation=None)

        self._final_softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training):
        x = self._bn_layer(inputs, training)
        x, conv_1_features = self._conv_block_1(x, training)
        x, conv_2_features = self._conv_block_2(x, training)
        x, conv_3_features = self._conv_block_3(x, training)
        x, conv_4_features = self._conv_block_4(x, training)
        x = self._flatten_layer(x)
        x, dense_1_features = self._dense_block_1(x, training)
        x, dense_2_features = self._dense_block_2(x, training)
        x, dense_3_features = self._dense_block_3(x, training)
        x, dense_4_features = self._dense_block_4(x, training)
        x = self._final_dense(x) # x are logits
        x = self._final_softmax(x) # x are class probs
        return x, dense_2_features, dense_4_features


def train(model, epochs, batch_size, learning_rate,
          data, save_dir, saver_delta=0.15, patience=2):
    '''
    The core training function, use this function to train a model.

    :param model: CNN model
    :param epochs: integer, number of epochs
    :param drop_rate: float, dropout_rate
    :param batch_size: integer, number of samples to put through the model at once
    :param data: tuple, train-test data Example(X_train, y_train, X_test, y_test)
    :param save_dir: string, path to a folder where model checkpoints will be saved
    :param saver_delta: float, used to prevent overfitted model to be saved
    :param patience: int, used for early stopping, number of consecutive epochs without improvement
    '''

    X_train, y_train, X_valid, y_valid = data

    best_test_accuracy = 0.0

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    epoch_val_accuracy = []
    counter = 0

    for epoch in range(epochs):

        train_accuracy = []
        train_loss = []

        for ii in tqdm(range(len(X_train) // batch_size)):
            start_id = ii * batch_size
            end_id = start_id + batch_size

            X_batch = X_train[start_id:end_id]
            y_batch = y_train[start_id:end_id]

            with tf.GradientTape() as tape:
                class_probs, _, _ = model(X_batch, training=True)
                loss_value = loss_fn(y_batch, class_probs)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_accuracy.append(sparse_accuracy(y_batch, class_probs))
            train_loss.append(loss_value)

        test_accuracy = []

        for ii in tqdm(range(len(X_valid) // batch_size)):
            start_id = ii*batch_size
            end_id = start_id + batch_size

            X_valid_batch = X_valid[start_id:end_id]
            y_valid_batch = y_valid[start_id:end_id]

            test_class_probs, _, _ = model(X_valid_batch, training=False)

            test_accuracy.append(sparse_accuracy(y_valid_batch, test_class_probs))

        print(f"Epoch: {epoch+1}/{epochs}",
              f" | Training accuracy: {np.mean(train_accuracy)}"
              f" | Training loss: {np.mean(train_loss)}")

        mean_test_acc = np.mean(test_accuracy)
        print(f"Test accuracy: {mean_test_acc}")

        epoch_val_accuracy.append(mean_test_acc)

        if mean_test_acc <= best_test_accuracy:
            counter = counter + 1
            if counter >= patience:
                print("Early stopping. Exiting training.")
                return
        else:
            counter = 0
            best_test_accuracy = mean_test_acc

        if np.mean(train_accuracy) > np.mean(test_accuracy):  # to prevent underfitting
            if np.abs(np.mean(train_accuracy) - np.mean(test_accuracy)) <= saver_delta:  # to prevent overfit
                if np.mean(test_accuracy) >= best_test_accuracy:
                    best_test_accuracy = np.mean(test_accuracy)
                    model.save_weights(f"{save_dir}/model_epoch_{epoch}", save_format='tf')


def create_training_set_vectors_with_colors(model,
                                            X_train,
                                            y_train,
                                            batch_size,
                                            checkpoint_path,
                                            image_size,
                                            distance='hamming'):
    '''
    Creates training set vectors and saves them in a pickle file.

    :param model: CNN model
    :param X_train: numpy array, loaded training set images
    :param y_train: numpy array,loaded training set labels
    :param batch_size: integer, number of samples to put trhough the model at once
    :param checkpoint_path: string, path to the model checkpoint
    :param image_size: tuple, single image (height, width)
    :param distance: string, type of distance to be used,
                             this parameter is used to choose a way how to prepare and save training set vectors
    '''

    model.load_weights(checkpoint_path)

    dense_2_features = []
    dense_4_features = []

    ##########################################################################
    ### Calculate color feature vectors for each image in the training set ###
    color_features = []
    for img in X_train:
        channels = cv2.split(img)
        features = []
        for chan in channels:
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.append(hist)

        color_features.append(np.vstack(features).squeeze())
    ##########################################################################

    # iterate through training set
    for ii in tqdm(range(len(X_train) // batch_size)):
        start_id = ii * batch_size
        end_id = start_id + batch_size

        X_batch = X_train[start_id:end_id]

        _, dense_2, dense_4 = model(X_batch, training=False)

        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)

    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    # hamming distance - vectors processing
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)  # binarize vectors
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        training_vectors = np.hstack((dense_2_features, dense_4_features))
        with open('hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    # cosine distance - vectors processing
    elif distance == 'cosine':
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        training_vectors = np.hstack((training_vectors, color_features[:len(training_vectors)]))
        with open('cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    #########################################################################
    ### Save training set color feature vectors to a separate pickle file ###
    with open('color_vectors.pickle', 'wb') as f:
        pickle.dump(color_features[:len(training_vectors)], f)
    #########################################################################


def compare_color(color_vectors,
                  uploaded_image_colors,
                  ids):
    '''
    Comparing color vectors of closest images from the training set with a color vector of a uploaded image (query image).

    :param color_vectors: color features vectors of closest training set images to the uploaded image
    :param uploaded_image_colors: color vector of the uploaded image
    :param ids: indices of training images being closest to the uploaded image (output from a distance function)
    '''
    color_distances = []

    for i in range(len(color_vectors)):
        color_distances.append(euclidean(color_vectors[i], uploaded_image_colors))

    # The 15 is just an random number that I have choosen, you can return as many as you need/want
    return ids[np.argsort(color_distances)[:15]]


def simple_inference_with_color_filters(model,
                                        train_set_vectors,
                                        uploaded_image_path,
                                        color_vectors,
                                        image_size,
                                        distance='hamming'):
    '''
    Doing simple inference for single uploaded image.

    :param model: CNN model
    :param session: tf.Session, restored session
    :param train_set_vectors: loaded training set vectors
    :param uploaded_image_path: string, path to the uploaded image
    :param color_vectors: loaded training set color features vectors
    :param image_size: tuple, single image (height, width)
    :param dsitance: string, type of distance to be used,
                             this parameter is used to choose a way how to prepare vectors
    '''

    image = image_loader(uploaded_image_path, image_size)

    ####################################################
    ## Calculating color histogram of the query image ##
    channels = cv2.split(image)
    features = []
    for chan in channels:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.append(hist)

    color_features = np.vstack(features).T
    ####################################################

    import ipdb; ipdb.set_trace()
    img_input = np.expand_dims(image, axis=0).astype('float32')
    _, dense_2_features, dense_4_features = model(img_input, training=False)

    closest_ids = None
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = hamming_distance(train_set_vectors, uploaded_image_vector)

        # Comparing color features between query image and closest images selected by the model
        closest_ids = compare_color(np.array(color_vectors)[closest_ids], color_features, closest_ids)

    elif distance == 'cosine':
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = cosine_distance(train_set_vectors, uploaded_image_vector)

        # Comparing color features between query image and closest images selected by the model
        closest_ids = compare_color(np.array(color_vectors)[closest_ids], color_features, closest_ids)

    return closest_ids

epochs = 20
batch_size = 128
learning_rate = 0.001
dropout_rate = 0.6
image_size = (32, 32)

#Using float16 for GPU acceleration
X_train, y_train = dataset_preprocessing('dataset/train/', 'dataset/labels.txt', image_size=image_size,
                                         image_paths_pickle="train_images_pickle", dtype='float32')
X_valid, y_valid = dataset_preprocessing('dataset/test/', 'dataset/labels.txt', image_size=image_size,
                                         image_paths_pickle="test_images_pickle", dtype='float32')

model = ImageSearchModel(dropout_rate, image_size)

data = (X_train, y_train, X_valid, y_valid)

train(model=model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, data=data, save_dir='saver')

create_training_set_vectors_with_colors(model, X_train, y_train, batch_size, 'saver/model_epoch_5', image_size)

with open('train_images_pickle.pickle', 'rb') as f:
    train_image_paths = pickle.load(f)

with open('hamming_train_vectors.pickle', 'rb') as f:
    train_set_vectors = pickle.load(f)

with open('color_vectors.pickle', 'rb') as f:
    color_vectors = pickle.load(f)

test_image = 'dataset/test/1052_airplane.png'
result_ids = simple_inference_with_color_filters(model, train_set_vectors, test_image, color_vectors, image_size,
                                                 distance='hamming')