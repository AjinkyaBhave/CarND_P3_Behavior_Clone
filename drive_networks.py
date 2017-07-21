from keras.layers import Flatten, Dense, Conv2D, Dropout
from keras.models import Sequential
from keras.utils import plot_model
from keras import backend as K

def alvinn(img_height, img_width, img_channels):
    # Network to steer car based on Deam Pomerleau's PhD thesis.
    model = Sequential()
    model.add(Flatten(input_shape=(img_height, img_width, img_channels)))
    # Hidden layer of 4 units
    model.add(Dense(4,activation='tanh'))
    # Output steering angle
    model.add(Dense(1))
    plot_model(model, to_file='alvinn.png')
    print(model.summary())
    return model

def nvidia(img_height, img_width, img_channels):
    # Network to steer car based on Nvidia's End-to-end learning paper.
    model = Sequential()
    # Set of convolutional layers to process image features
    model.add(Conv2D(24, (5,5), strides=2, padding="valid", activation='relu', input_shape=(img_height, img_width, img_channels), data_format='channels_last'))
    model.add(Conv2D(36, (5,5), strides=2, padding="valid", activation='relu'))
    model.add(Conv2D(48, (5,5), strides=2, padding="valid", activation='relu'))
    model.add(Conv2D(64, (3,3), strides=1, padding="valid", activation='relu'))
    model.add(Conv2D(64, (3,3), strides=1, padding="valid", activation='relu'))
    # Set of fully-connected layers to calculate final steering
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    # Output steering angle
    model.add(Dense(1))
    plot_model(model, to_file='nvidia.png')
    print(model.summary())
    return model
