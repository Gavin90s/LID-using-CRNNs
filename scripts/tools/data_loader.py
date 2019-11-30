import abc
import csv

import numpy as np
from keras.utils.np_utils import to_categorical

from tools.audio_to_image import SpectrogramGenerator


class CSVLoader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir):
        self.images_label_pairs = []
        self.input_shape = (129, 500, 1)
        self.num_classes = 3
        self.batch_size = 128

        with open(data_dir, "r") as csv_file:
            for (file_path, label) in list(csv.reader(csv_file)):
                self.images_label_pairs.append((file_path, int(label)))

    def get_data(self, should_shuffle=True, is_prediction=False):
        start = 0

        while True:
            data_batch = np.zeros((self.batch_size,) + self.input_shape)  # (batch_size, cols, rows, channels)
            label_batch = np.zeros(
                (self.batch_size, self.num_classes))  # (batch_size,  num_classes)

            for i, (file_path, label) in enumerate(self.images_label_pairs[start:start + self.batch_size]):
                data = self.process_file(file_path)
                height, width, channels = data.shape
                data_batch[i, : height, :width, :] = data
                label_batch[i, :] = to_categorical([label], num_classes=self.num_classes)  # one-hot encoding

            start += self.batch_size

            # Reset generator
            if start + self.batch_size > self.get_num_files():
                start = 0
                if should_shuffle:
                    np.random.shuffle(self.images_label_pairs)

            # For predictions only return the data
            if is_prediction:
                yield data_batch
            else:
                yield data_batch, label_batch

    def get_input_shape(self):

        return self.input_shape

    def get_num_files(self):
        # Minimum number of data points without overlapping batches
        return (len(self.images_label_pairs) // self.batch_size) * self.batch_size

    def get_labels(self):
        return [label for (file_path, label) in self.images_label_pairs]

    @abc.abstractmethod
    def process_file(self, file_path):
        raise NotImplementedError("Implement in child class.")


class ImageLoader(CSVLoader):

    def process_file(self, file_path):
        image = SpectrogramGenerator.audio_to_spectrogram(file_path, 50, 129)

        # Image shape should be (cols, rows, channels)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

        assert len(image.shape) == 3, "Error: Image dimension mismatch"

        return np.divide(image, 255.0)  # Normalize images


# REFERENCES:
# 1. Bartz, C., Herold, T., Yang, H., and Meinel, C.: ‘Language identification using deep convolutional
#     recurrent neural networks’, in Editor (Ed.)^(Eds.): ‘Book Language identification using deep convolutional
#     recurrent neural networks’ (Springer, 2017, edn.), pp. 880-889
#     https://arxiv.org/pdf/1708.04811v1.pdf
#
# 2. Original code for the paper that can be found at
#     https://github.com/HPI-DeepLearning/crnn-lid
