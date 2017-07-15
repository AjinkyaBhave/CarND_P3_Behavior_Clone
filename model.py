import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, Cropping2D
from keras.models import Sequential, load_model

def process_image(X):
    crop_top = 50     # number of pixel rows to remove from image top
    crop_bottom = 20  # number of pixel rows to remove from image bottom
    # Normalise pixels between (-0.5,0.5)
    X_proc = (X/255.0)-0.5
    return X_proc

#def augment_data(X):
    #images.append(np.fliplr(img_centre))
    #controls.append(-steer_centre)

def plot_history(train_history):
    # Print the keys contained in the history object
    print(train_history.history.keys())
    # Plot the training and validation loss for each epoch
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# Data directory containing images and control measurements
data_dir = './dataset/initial/'
# File to save current best network with weights
checkpoint_file = './model_checkpoint/model.h5'

samples=[]
# Read in contents of driving data CSV file
with open(data_dir + 'driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)
# Remove first line containing column headings
del samples[0]
# Correction to steering for left and right images. Tunable.
steer_offset = 0.001

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            controls = []
            for batch_sample in batch_samples:
                img_centre = cv2.imread(data_dir + batch_sample[0])
                img_left   = cv2.imread(data_dir + batch_sample[1])
                img_right  = cv2.imread(data_dir + batch_sample[2])
                # Read steering command
                steer_centre = float(line[3])
                # Create adjusted steering commands for the side camera images
                steer_left = steer_centre + steer_offset
                steer_right = steer_centre - steer_offset
                images.append(img_centre)
                #images.append(img_left)
                #images.append(img_right)
                controls.append(steer_centre)
                #controls.append(steer_left)
                #controls.append(steer_right)

            # trim image to only see section with road
            X = np.array(images)
            y = np.array(controls)
            yield sklearn.utils.shuffle(X, y)

#img_height, img_width, img_channels = X_train[0].shape
#print(X_train.shape, X_train[0].shape)
# Split samples into training and validation data sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Keras callback functions used during training
# Stop training if validation accuracy decreases after 'patience' epochs
early_stopping = EarlyStopping(monitor='val_acc', patience=3)
# Save best model based on validation accuracy
model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='auto')

# Define network to steer car
model = Sequential()
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(img_height, img_width, img_channels)))
#model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(img_height, img_width, img_channels)))
model.add(Flatten(input_shape=(160,320,3)))
# Hidden layer of 4 units
model.add(Dense(4))
# Output steering angle
model.add(Dense(1))

# Train model
#model.load_model(checkpoint_file)
model.compile(loss='mse', optimizer='adam')
#train_history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, callbacks=[early_stopping])
train_history=model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=1)
model.save('model.h5')
plot_history(train_history)

