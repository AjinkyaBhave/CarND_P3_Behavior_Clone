from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.models import Sequential, load_model

def alvinn(img_height, img_width, img_channels):
    # Network to steer car based on Deam Pomerleau's PhD thesis.
    model = Sequential()
    #model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=(img_height, img_width, img_channels)))
    #model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Flatten(input_shape=(img_height, img_width, img_channels)))
    # Hidden layer of 4 units
    model.add(Dense(4))
    # Output steering angle
    model.add(Dense(1))
    print(model.summary())
    return model