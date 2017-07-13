import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Cropping2D
from keras.models import Sequential

data_dir = './dataset/initial/'
lines=[]
# Read in contents of driving data CSV file
with open(data_dir + 'driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
# Remove first line containing column headings
del lines[0]

images = []
controls = []
steer_offset = 0.001

# Read images from all cameras and steering angles
for line in lines:
    img_centre = cv2.imread(data_dir + line[0])
    img_left   = cv2.imread(data_dir + line[1])
    img_right  = cv2.imread(data_dir + line[2])
    steer_centre = float(line[3])
    # create adjusted steering measurements for the side camera images
    steer_left  = steer_centre + steer_offset
    steer_right = steer_centre - steer_offset
    # Add images and controls to data set
    #images.extend([img_centre, img_left, img_right])
    #controls.extend([steer_centre, steer_left, steer_right])
    images.append(img_centre)
    images.append(img_left)
    images.append(img_right)
    controls.append(steer_centre)
    controls.append(steer_left)
    controls.append(steer_right)

def process_images(X):
    # Normalise pixels between (-0.5,0.5)
    X_proc = (X/255.0)-0.5
    return X_proc

#def augment_data(X):
    #images.append(np.fliplr(img_centre))
    #controls.append(-steer_centre)

crop_top = 50 # number of pixel rows to remove from image top
crop_bottom  = 20 # number of pixel rows to remove from image bottom

X_train = np.array(images)
y_train = np.array(controls)
img_height, img_width, img_channels = X_train[0].shape
print(X_train.shape, y_train.shape, len(images))

model = Sequential()
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(img_height, img_width, img_channels)))
#model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(img_height, img_width, img_channels)))
model.add(Flatten(input_shape=(img_height, img_width, img_channels)))
model.add(Dense(1))

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.compile(loss='mse', optimizer='adam')
train_history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, callbacks=[early_stopping])
model.save('model.h5')

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