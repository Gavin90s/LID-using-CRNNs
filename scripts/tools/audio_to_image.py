import fnmatch
import os
import random
from queue import Queue
from subprocess import Popen, PIPE, STDOUT

import numpy as np
from PIL import Image


def recursive_glob(path, pattern):
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                if os.path.isfile(filename):
                    yield filename


class SpectrogramGenerator(object):

    def __init__(self, source, config, shuffle=False, max_size=100, run_only_once=False):

        self.source = source
        self.config = config
        self.queue = Queue(max_size)
        self.shuffle = shuffle
        self.run_only_once = run_only_once

        if os.path.isdir(self.source):
            files = []
            files.extend(recursive_glob(self.source, "*.wav"))
            files.extend(recursive_glob(self.source, "*.mp3"))
            files.extend(recursive_glob(self.source, "*.m4a"))
        else:
            files = [self.source]

        self.files = files

    @staticmethod
    def audio_to_spectrogram(file, pixel_per_sec, height):
        """
        Convert audio sample to spectrogram images
        :param file: audio file
        :param pixel_per_sec: pixels per second
        :param height: height of the spectrogram image

        Arguments for sox command:
        V0 - Verbosity level: ignore everything
        c 1 - channel 1 / mono
        n - apply filter/effect
        rate 10k - limit sampling rate to 10k --> max frequency 5kHz (Shenon Nquist Theorem)
        y - small y: defines height
        X capital X: defines pixels per second
        m - monochrome
        r - no legend
        o - output to stdout (-)
        """

        file_name = "tmp_{}.png".format(random.randint(0, 100000))
        command = "sox -V0 '{}' -n remix 1 rate 10k spectrogram -y {} -X {} -m -r -o {}".format(
            file, height, pixel_per_sec, file_name)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        output, errors = p.communicate()

        if errors:
            print(errors)

        # image = Image.open(StringIO(output))
        if not os.path.exists(file_name):
            print("Audio to Spectrogram Error: File not found (Spectrogram conversion error)")
            exit()

        image = Image.open(file_name)
        os.remove(file_name)

        return np.array(image)

    def get_generator(self):
        start = 0

        while True:
            file = self.files[start]

            try:
                target_height, target_width, target_channels = self.config["input_shape"]

                image = self.audio_to_spectrogram(file, self.config["pixel_per_second"], target_height)
                image = np.expand_dims(image, -1)  # add dimension for mono channel
                height, width, channels = image.shape

                assert target_height == height, "Height mismatch {} vs {}".format(target_height, height)

                num_segments = width // target_width

                for i in range(0, num_segments):
                    slice_start = i * target_width
                    slice_end = slice_start + target_width
                    slice_ = image[:, slice_start:slice_end]

                    # Ignore black images
                    if slice_.max() == 0 and slice_.min() == 0:
                        continue

                    yield slice_

            except Exception as e:
                print("SpectrogramGenerator Exception: ", e, file)
                pass

            finally:
                start += 1
                if start >= len(self.files):
                    if self.run_only_once:
                        break
                    start = 0

                    if self.shuffle:
                        np.random.shuffle(self.files)

    def get_num_files(self):
        return len(self.files)


# REFERENCES:
# Bartz, C., Herold, T., Yang, H., and Meinel, C.: ‘Language identification using deep convolutional recurrent neural
# networks’, in Editor (Ed.)^(Eds.): ‘Book Language identification using deep convolutional recurrent neural networks’
# (Springer, 2017, edn.), pp. 880-889 https://arxiv.org/pdf/1708.04811v1.pdf
