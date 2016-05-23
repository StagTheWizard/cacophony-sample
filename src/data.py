import os
import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from PIL import Image
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH

"""
Brent Martin 22/4/2016
Routines for loading images files into the appropriate data structures
for processing by the colvolutional network code.
"""


def load_images(train_folder, valid_folder, test_folder, train_targets, valid_targets, test_targets):
    """
    Loads folder of images for use by convolution_mlp.
    train_set, valid_set, test_set format: tuple(input, target)
    input is a numpy.ndarray of 2 dimensions (a matrix)
    where each row corresponds to an example. target is a
    numpy.ndarray of 1 dimension (vector) that has the same length as
    the number of rows in the input. It should give the target
    to the example with the same index in the input.
    """
    return load_folder(train_folder, train_targets), \
           load_folder(valid_folder, valid_targets), \
           load_folder(test_folder, test_targets),


def load_folder(folder, targets):
    """
    Loads a single folder of images
    - load each image into a vector of greyscale values, making a matrix of <image , pixels>
    - wrap the matrix in a theano shared tensor variable
    - wrap the target vector in a tensor shared variable
    - return as a tuple (images, targets)
    """
    print("*** Loading folder " + folder + " ***")
    images = []
    for file in os.listdir(folder):
        print("...loading " + file)
        filename = folder + "/" + file
        images.append(load_image(filename))
    images_tensor = theano.shared(numpy.asarray(images))
    targets_tensor = theano.shared(numpy.asarray(targets))
    return images_tensor, targets_tensor


def load_image(filename):
    """
    Returns the image as a 1d vector of greyscale values
    """
    img = Image.open(filename).convert('L')
    # extract the image as a vector of greyscale values between 0 and 1
    raster = (numpy.asarray(img, dtype='float64') / 256.).reshape(IMAGE_HEIGHT * IMAGE_WIDTH)
    # print raster # DEBUG - shows the vector of greyscale values
    return raster


def load_data():
    # Change these to point to your training, validation and test set image directories
    resources_dir = '../resources/halfsize images'

    train_dir = resources_dir + '/train'
    valid_dir = resources_dir + '/valid'
    test_dir = resources_dir + '/test'

    # Number of classes per
    classes = 5

    n_train_per_class = 14
    n_valid_per_class = 8
    n_test_per_class = 8

    train_target = [i // n_train_per_class for i in range(n_train_per_class*classes)]
    valid_target = [i // n_valid_per_class for i in range(n_valid_per_class*classes)]
    test_target = [i // n_test_per_class for i in range(n_test_per_class*classes)]

    return load_images(train_dir, valid_dir, test_dir, train_target, valid_target, test_target)
