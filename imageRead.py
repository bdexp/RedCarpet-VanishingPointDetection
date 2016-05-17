import imageflow
import numpy as np
import pandas as pd
import random

from skimage import exposure, filters
from scipy.misc import imresize
import matplotlib.pyplot as plt

DATA_PATH = '../Data/Clean_images/'
DATA_FOLDERS = [
    'dashedlinesmissing',
    'fulltrack1',
    'fulltrack2',
    'leftcurve',
    'rightcurve',
    'rightlanemissing',
    'roadnear',
    'startbox',
    'straightroad'
]

# Class for keeping a dataset together with labels.
class DataSet(object):

    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            self._images_original = images

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def images_original(self):
        return self._images_original

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed


    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:

            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

# Read images from given folders.
def read_images(folders, resize=False, newHeight=0, newWidth=0):

    images = np.zeros([0, 480, 752], dtype=np.uint8)

    for name in folders:

        print('Loading images in: ' + name)
        temp = imageflow.reader._read_pngs_from(DATA_PATH + name)
        images = np.append(images, temp, axis=0)

    if resize:

        # Resize the images
        images_resized = np.zeros([images.shape[0], newHeight, newWidth], dtype=np.uint8)
        for image in range(images.shape[0]):
            images_resized[image] = imresize(images[image], [newHeight, newWidth], 'bilinear')

        return (images_resized)


    return (images)

# Read labels from given folders.
def read_labels(folders, newHeight, newWidth):

    labels = np.zeros([0,2])
    for name in folders:
        temp = pd.read_csv(DATA_PATH + name + '/labels.csv')
        labels = np.append(labels, temp[['VP_x', 'VP_y']], axis=0)

    # Divide to match the resize operation
    # labels[:,0] = np.round(labels[:,0] / (480 / newHeight))
    # labels[:,0] = np.round(labels[:,0] / (752 / newWidth))

    return(labels)


def combine_data(resize=False, newHeight=0, newWidth=0, training_ratio=0.8, distortionRate=0.0, carOriginPos=[376.0, 480.0], addFlipped=True):

    class DataSets(object):
        pass

    data_sets = DataSets()

    # Read images and labels
    images = read_images(DATA_FOLDERS, resize, newHeight, newWidth)
    labels = read_labels(DATA_FOLDERS, newHeight, newWidth)

    # Delete images and labels for which the labels are infinite
    mask = labels[:,0] > -1000
    labels = labels[mask]
    images = images[mask]

    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 256.0)
    #images = np.multiply(images, 1.0 / np.float(np.max(images)))

    # Batch randomisation
    nr_of_splits = 40
    nrsplits = images.shape[0] / nr_of_splits
    nrrows = int(training_ratio * nrsplits)
    trainidx = np.sort(np.random.choice(nrsplits, nrrows, replace=False))
    testidx = [x for x in range(nrsplits) if x not in trainidx]
    #validx = np.sort(np.random.choice(testidx, np.ceil(((1.0-training_ratio)/2.0)*nrsplits).astype(int), replace=False))
    #testidx = list(set(testidx) - set(validx))


    # Training Set
    images_split = np.array_split(images, nrsplits)
    labels_split = np.array_split(labels, nrsplits)
    train_images = images_split[trainidx[0]]
    train_labels = labels_split[trainidx[0]]
    for idx in trainidx[1:]:
        train_images = np.append(train_images, images_split[idx], axis=0)
        train_labels = np.append(train_labels, labels_split[idx], axis=0)

    # Flipped Images
    if addFlipped:
        print('\nFlips Images..')

        flipped_images = [np.fliplr(i) for i in images]
        flipped_labels = np.copy(labels)

        for lidx in range(flipped_labels.shape[0]):
            if flipped_labels[lidx][0] >= carOriginPos[0]:
                flipped_labels[lidx][0] = carOriginPos[0] - (flipped_labels[lidx][0] - carOriginPos[0])
            else:
                flipped_labels[lidx][0] = carOriginPos[0] + (carOriginPos[0] - flipped_labels[lidx][0])

        flipped_images_split = np.array_split(flipped_images, nrsplits)
        flipped_labels_split = np.array_split(flipped_labels, nrsplits)

        '''
        IMAGE = 600
        print(labels[IMAGE])
        print(flipped_labels[IMAGE])
        plt.figure()
        plt.imshow(imresize(images[IMAGE], [480, 752], 'bilinear'), cmap='gray')
        plt.plot(labels[IMAGE][0], labels[IMAGE][1], "ro")
        plt.figure()
        plt.imshow(imresize(flipped_images[IMAGE], [480, 752], 'bilinear'), cmap='gray')
        plt.plot(flipped_labels[IMAGE][0], flipped_labels[IMAGE][1], "ro")
        plt.figure()
        plt.imshow(exposure.adjust_gamma(images[IMAGE], random.uniform(1.0, 3.0)), cmap='gray')
        plt.figure()
        plt.imshow(filters.gaussian(images[IMAGE], random.uniform(0.5, 2.0)), cmap='gray')
        plt.figure()
        plt.imshow(exposure.equalize_hist(images[IMAGE]), cmap='gray')
        plt.show()
        '''

        del flipped_images
        del flipped_labels

        for idx in trainidx:
            train_images = np.append(train_images, flipped_images_split[idx], axis=0)
            train_labels = np.append(train_labels, flipped_labels_split[idx], axis=0)

        del flipped_images_split
        del flipped_labels_split



    # Distorted Images
    print('\nDistorting Images..')

    distorted_images = np.copy(images)

    del images
    del labels

    for i in range(distorted_images.shape[0]):
        distortionType = random.randrange(0,3)
        if (distortionType == 0): # Gamma Correction
            distorted_images[i] = exposure.adjust_gamma(distorted_images[i], random.uniform(1.0, 3.0))
        elif (distortionType == 1): # Gaussian Blur
            distorted_images[i] = filters.gaussian(distorted_images[i], random.uniform(0.5, 2.0))
        else: # Histogram Equalization
            distorted_images[i] = exposure.equalize_hist(distorted_images[i])

    distorted_images_split = np.array_split(distorted_images, nrsplits)

    for idx in trainidx:
        if (random.uniform(0.0, 1.0) < distortionRate):
            train_images = np.append(train_images, distorted_images_split[idx], axis=0)
            train_labels = np.append(train_labels, labels_split[idx], axis=0)

    del distorted_images
    del distorted_images_split


    # Test set
    test_images = images_split[testidx[0]]
    test_labels = labels_split[testidx[0]]
    for idx in testidx[1:]:
        test_images = np.append(test_images, images_split[idx], axis=0)
        test_labels = np.append(test_labels, labels_split[idx], axis=0)


    # Put the training, validation, and test set into a dataset
    data_sets.train = DataSet(np.expand_dims(train_images, axis=3), train_labels)
    data_sets.test = DataSet(np.expand_dims(test_images, axis=3), test_labels)


    print('')
    print('Training Set Shape: ' + str(data_sets.train.images.shape))
    print('Test Set Shape: ' + str(data_sets.test.images.shape))



    return (data_sets)