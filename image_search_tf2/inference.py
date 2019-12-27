import cv2

from utils.utils import *
from utils.dataset import image_loader


def simple_inference(model,
                     train_set_vectors,
                     uploaded_image_path,
                     image_size,
                     distance='hamming'):
    '''
    Doing simple inference for single uploaded image.

    :param model: CNN model
    :param session: tf.Session, restored session
    :param train_set_vectors: loaded training set vectors
    :param uploaded_image_path: string, path to the uploaded image
    :param image_size: tuple, single image (height, width)
    :param dsitance: string, type of distance to be used,
                             this parameter is used to choose a way how to prepare vectors
    '''

    image = image_loader(uploaded_image_path, image_size)
    channels = cv2.split(image)
    features = []

    for chan in channels:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.append(hist)

    color_features = np.vstack(features).T

    _, dense_2_features, dense_4_features = model(image, training=False)

    closest_ids = None
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = hamming_distance(train_set_vectors, uploaded_image_vector)

    elif distance == 'cosine':
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = cosine_distance(train_set_vectors, uploaded_image_vector)

    return closest_ids


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