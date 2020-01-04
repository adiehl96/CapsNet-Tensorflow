import os
import scipy
import numpy as np
import tensorflow as tf
import pickle
from skimage.transform import rotate, resize
from skimage import exposure
import skimage.io as io

from config import cfg


def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_face_set(batch_size, is_training=True):
    path = os.path.join('data', 'faceSet')
    if is_training:
        fd = open(os.path.join(path, 'imgset'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = pickle.load(fd)
        trainX = loaded.reshape((57575, 86, 86, 3)).astype(np.float32)

        fd = open(os.path.join(path, 'categories'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = pickle.load(fd)
        trainY = loaded.reshape((57575)).astype(np.int32)

        data_set = list(zip(trainX,trainY))
        np.random.shuffle(data_set)
        trainX, trainY = list(zip(*data_set))
        trainX = np.asarray(trainX).reshape((57575, 86, 86, 3)).astype(np.float32)
        trainY = np.asarray(trainY).reshape((57575)).astype(np.int32)
        trX = trainX[:52000] / 255.
        trY = trainY[:52000]

        valX = trainX[52000:, ] / 255.
        valY = trainY[52000:]

        num_tr_batch = 52000 // batch_size
        num_val_batch = 5575 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 'faceseteval'), 'rb')
        loaded = pickle.load(fd)
        trainX = loaded.reshape((10000, 86, 86, 3)).astype(np.float32)

        fd = open(os.path.join(path, 'facesetevalcat'), 'rb')
        loaded = pickle.load(fd)
        trainY = loaded.reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return trainX / 255., trainY, num_te_batch


def load_facegreyredux_set(batch_size, is_training=True):
    path = os.path.join('data', 'facegreyredux')
    if is_training:
        fd = open(os.path.join(path, 'facegreyredux'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = pickle.load(fd)
        loaded = np.asarray(loaded)
        trainX = loaded.reshape((50000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'facegreyreduxcat'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = pickle.load(fd)
        loaded = np.asarray(loaded)
        trainY = loaded.reshape((50000)).astype(np.int32)

        data_set = list(zip(trainX,trainY))
        np.random.shuffle(data_set)
        trainX, trainY = list(zip(*data_set))
        trainX = np.asarray(trainX).reshape((50000, 28, 28, 1)).astype(np.float32)
        trainY = np.asarray(trainY).reshape((50000)).astype(np.int32)
        trX = trainX[:40000] / 255.
        trY = trainY[:40000]

        valX = trainX[40000:, ] / 255.
        valY = trainY[40000:]

        num_tr_batch = 40000 // batch_size
        num_val_batch = 10000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 'facegreyreduxeval'), 'rb')
        loaded = pickle.load(fd)
        loaded = np.asarray(loaded)
        trainX = loaded.reshape((10000, 28, 28, 1)).astype(np.float32) / 255.

        fd = open(os.path.join(path, 'facegreyreduxevalcat'), 'rb')
        loaded = pickle.load(fd)
        trainY = loaded.reshape((10000)).astype(np.int32)

        rotatedlist = []
        for image in trainX:
            image = rotate(image, cfg.rotation, preserve_range=True)
            if(cfg.mooney):
                v_min, v_max = np.percentile(image, (49.99999999, 51))
                image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
            rotatedlist.append(image)
            if(len(rotatedlist)==1000):
                I = resize(image.reshape(28, 28), (128, 128))
                io.imsave("rotate" + str(cfg.rotation) +  "example.jpg", I, cmap='gray')
        rotatedlist = np.asarray(rotatedlist)
        trainX = rotatedlist.reshape((10000, 28, 28, 1)).astype(np.float32)

        num_te_batch = 10000 // batch_size
        return trainX, trainY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'faceset':
        return load_face_set(batch_size, is_training)
    elif dataset == 'facegreyredux':
        return load_facegreyredux_set(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'faceset':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_face_set(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == 'facegreyredux':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_facegreyredux_set(batch_size, is_training=True)

    def generator():
        for e1, e2 in zip(trX, trY):
            yield e1, e2

    tf_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(list(trX[0].shape)), ())).repeat().shuffle(batch_size * 32).batch(batch_size=batch_size, drop_remainder=True)
    # tf_dataset = tf.data.Dataset.from_tensor_slices((trX, trY)).repeat().shuffle(batch_size * 32).batch(batch_size, drop_remainder=True)

    # iterator = tf_dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(tf_dataset)
    (X, Y) = iterator.get_next()
    # Y=tf.reshape(Y, tf.TensorShape(batch_size))

    return X, Y


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
