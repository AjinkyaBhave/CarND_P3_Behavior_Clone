import csv
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os.path
import drive_networks

# Training parameters
batch_size = 32

def process_image(X):
    # Normalise pixels between (-0.5,0.5)
    X_proc = (X/255.0)-0.5
    return X_proc

#def augment_data(X):
    #images.append(np.fliplr(img_centre))
    #controls.append(-steer_centre)

def display_images(X, num_images = 10):
    # Select ten random images to display for visual confirmation of processing
    img_idx = np.random.randint(0, len(X), num_images)
    ncols = 10
    nrows = int(num_images/ncols) + 1
    # Display images in subplots
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))
    fig.subplots_adjust(hspace=.1, wspace=.05)
    ax = ax.ravel()
    for i in range(num_images):
        ax[i].imshow(X[img_idx[i]])
        ax[i].axis('off')

def plot_history(train_history):
    # Print the keys contained in the history object
    print(train_history.history.keys())
    # Plot the training and validation loss for each epoch
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Training Performance')
    plt.ylabel('MSE')
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

def generator(samples, batch_size):
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
                images.extend([img_centre,np.fliplr(img_centre)] )
                #images.append()
                #images.append(img_left)
                #images.append(img_right)
                controls.extend([steer_centre,-steer_centre])
                #controls.append(steer_left)
                #controls.append(steer_right)

            X = np.array(images)
            y = np.array(controls)
            #print(X.shape, y.shape)
            yield sklearn.utils.shuffle(X, y)

#img_height, img_width, img_channels = X_train[0].shape

# Split samples into training and validation data sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Keras callback functions used during training
# Stop training if validation accuracy decreases after 'patience' epochs
early_stopping = EarlyStopping(monitor='val_acc', patience=3)
# Save best model based on validation accuracy
model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', save_best_only=True, save_weights_only=True, mode='auto')

# Create driving network model
if os.path.isfile(checkpoint_file):
    # Load existing pre-trained network if it exists
    model = load_model(checkpoint_file)
    print("Using pre-trained network")
else:
    # Load a new instance of the network
    model = drive_networks.alvinn()
    print("Using new network")

# Train model
steps_per_epoch  = np.math.ceil(len(train_samples)/batch_size)
validation_steps = np.math.ceil(len(validation_samples)/batch_size)
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
train_history=model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator,
                    validation_steps=validation_steps, epochs=2, callbacks=[model_checkpoint, early_stopping])
#model.save('model.h5')
print("Model saved.")
plot_history(train_history)

