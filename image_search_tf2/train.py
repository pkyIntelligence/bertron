import tensorflow as tf
import pickle
import tqdm
import cv2

from utils.utils import *


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