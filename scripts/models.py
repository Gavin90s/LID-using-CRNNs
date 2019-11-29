from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Permute, Reshape, Bidirectional, LSTM
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import plot_model


class CRNN:

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.model = Sequential()

    # TODO: tune model hyper-parameters
    def build_model(self, input_shape, weight_decay=0.001, convolution_activation='relu', padding='same',
                    pool_size=(2, 2), strides=(2, 2), output_layer_activation='softmax'):
        kernel_regularizer = l2(weight_decay)

        # Layer 1
        self.model.add(Conv2D(16, kernel_size=7, padding=padding, activation=convolution_activation,
                              kernel_regularizer=kernel_regularizer, input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=pool_size, strides=strides))

        # Layer 2
        self.model.add(Conv2D(32, kernel_size=5, padding=padding, activation=convolution_activation,
                              kernel_regularizer=kernel_regularizer))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=pool_size, strides=strides))

        # Layer 3
        self.model.add(Conv2D(64, kernel_size=3, padding=padding, activation=convolution_activation,
                              kernel_regularizer=kernel_regularizer))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=pool_size, strides=strides))

        # Layer 4
        self.model.add(Conv2D(128, kernel_size=3, padding=padding, activation=convolution_activation,
                              kernel_regularizer=kernel_regularizer))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=pool_size, strides=strides))

        # Layer 5
        self.model.add(Conv2D(256, kernel_size=3, padding=padding, activation=convolution_activation,
                              kernel_regularizer=kernel_regularizer))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=pool_size, strides=strides))

        # (bs, y, x, c) --> (bs, x, y, c)
        self.model.add(Permute((2, 1, 3)))

        # (bs, x, y, c) --> (bs, x, y * c)
        bs, x, y, c = self.model.layers[-1].output_shape
        self.model.add(Reshape((x, y * c)))

        self.model.add(Bidirectional(LSTM(512, return_sequences=False), merge_mode="concat"))
        self.model.add(Dense(self.num_classes, activation=output_layer_activation))

        print(self.model.summary())
        plot_model(self.model, to_file="model.png")
        return self.model
