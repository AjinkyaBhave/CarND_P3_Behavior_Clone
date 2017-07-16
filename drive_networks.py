from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.models import Sequential, load_model

# Network parameters
crop_top = 50     # number of pixel rows to remove from image top
crop_bottom = 20  # number of pixel rows to remove from image bottom
img_height = 160
img_width  = 320
img_channels = 3

def alvinn():
    # Define network to steer car
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(img_height, img_width, img_channels)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Flatten())
    # Hidden layer of 4 units
    model.add(Dense(4))
    # Output steering angle
    model.add(Dense(1))
    print(model.summary())
    return model