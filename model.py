import csv
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os.path
from drive_networks import alvinn, nvidia
from visualise import *
import os

# Set path to Graphviz to visualise network
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

### PARAMETERS SECTION.
# Use ALVINN network if true. Else use NVIDIA network.
ALVINN = 0

## Training parameters
# Batch size for generator function in training and validation
batch_size = 32
# Number of epochs
n_epochs = 2
# Learning rate
learn_rate = 0.001
# Maximum steering angle in degrees
steer_max = 25
steer_min = -25
# Correction to steering for left and right images. Tunable.
steer_offset = 0.25
steer_prob = 0.7  # Threshold for keeping the sample with straight steer.
steer_dev = 1.0   # Deviation from zero to be considered straight driving
# Threshold for flipping image horizontally
flip_prob = 0.5
# Data directory containing images and control measurements
data_dir = './dataset/track1/final/'
img_dir = data_dir+'IMG/'
# File to save current best network with weights
checkpoint_file = './model_checkpoint/model.h5'

## Image parameters
img_height    = 160 # Original image rows
img_width     = 320 # Original image columns
if ALVINN:
    img_channels  = 1   # Grayscale channel
    resize_width  = 30  # Resized image rows
    resize_height = 32  # Resized image columns
else:
    img_channels  = 3   # RGB channels
    resize_width  = 64  # Resized image rows
    resize_height = 64  # Resized image columns
crop_top      = 50  # Number of pixel rows to remove from image top
crop_bottom   = 20  # Number of pixel rows to remove from image bottom
crop_height   = img_height-crop_top-crop_bottom # cropped image height

def analyse_data(samples, num_images=10):
    # Select random images to display for visual confirmation of pre-processing
    img_idx = np.random.randint(0, len(samples), num_images)
    steer_data = []
    image_data = []
    steer_plot = []
    cmap = None
    idx = 0
    for sample in samples:
        if idx in img_idx:
            image, steer_ = select_image(sample)
            steer_plot.append(["%.3f" % steer_])
            image = process_image(image)
            image_data.append(image)
        steer_data.append(float(sample[3]))
        idx+=1

    steer_proc = process_steer(steer_data)
    fig, [ax1, ax2] = plt.subplots(2)
    fig.suptitle('Steering Data Analysis', fontsize=20)
    # Display histogram of steering control.
    hist, bins = np.histogram(steer_data, bins='auto')
    ax1.bar(bins[:-1], hist, width=0.1)
    ax1.set_title('Histogram')
    ax2.plot(steer_data)
    ax2.plot(steer_proc)
    ax2.set_title('Time Series')

    ncols = 10
    nrows = np.math.ceil(num_images/ncols)
    print(nrows)
    # Display images in subplots
    fig, ax = plt.subplots(nrows, ncols, figsize=(64, 64))
    fig.suptitle("Processed Images", fontsize=20)
    fig.subplots_adjust(hspace=.1, wspace=.05)
    ax = ax.ravel()
    print(len(ax))
    if ALVINN:
        cmap='gray'
    for i in range(num_images):
        if ALVINN:
            ax[i].imshow(image_data[i][:,:,0], cmap=cmap)
        else:
            ax[i].imshow(image_data[i], cmap=cmap)
        ax[i].set_title(steer_plot[i])
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def select_image(batch_sample):
    # Choose randomly which camera image to use
    img_id = np.random.randint(2)
    # Read steering command
    steer_centre = float(batch_sample[3])
    if img_id == 0:
        image = cv2.imread(img_dir + batch_sample[0].split('/')[-1])
        # Use original steering for the centre camera image
        steer = steer_centre
    elif img_id == 1:
        image = cv2.imread(img_dir + batch_sample[1].split('/')[-1])
        # Create adjusted steering commands for the left camera image
        steer = steer_centre + steer_offset
    else:
        image = cv2.imread(img_dir + batch_sample[2].split('/')[-1])
        # Create adjusted steering commands for the right camera image
        steer = steer_centre - steer_offset
    # Convert image to RGB format since cv2.imread() uses BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Choose randomly whether to flip this image horizontally
    if np.random.uniform() > flip_prob:
        # Flip the image half the time
        image = np.fliplr(image)
        # Correct steering for flipped image
        steer = -steer
    # Brightness augmentation for robustness
    image = brighten_img(image)
    return image, steer

def brighten_img(image):
    # Randomly brighten image by up to 30%
    dbright = 0.3 + np.random.uniform()
    image_out = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_out[:,:,2] = image_out[:,:,2]*dbright
    image_out[image_out[:,:,2]>255] = 255
    image_out = cv2.cvtColor(image_out, cv2.COLOR_HSV2RGB)
    return image_out

def process_image(image):
    # Crop image  rows based on crop_top and crop_bottom parameters
    image = image[crop_top:img_height - crop_bottom, :, :]
    # Resize image to reduce processing
    image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    if ALVINN:
        # Pre-process single channel to emphasise road structure
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2]
        image = 255 - image
        '''image_sum = (image[:, :, 0].astype(np.float32) + image[:, :, 1].astype(np.float32)
                     + image[:, :, 2].astype(np.float32))
        # Prevent division by zero value pixels
        image_sum[image_sum[:, :] < 1.0] = 1.0
        image = (image[:, :, 0] / image_sum + image[:, :, 0] / 255.0).astype(np.float32)
        '''
        image = image.reshape(image.shape + (1,))
    # Normalise pixels between (-0.9,0.9)
    image = (image / 255.0 - 0.5).astype(np.float32)
    return image

def process_steer(samples, n=3):
    # Filter steering commands using a moving average of size 3
    ret = np.cumsum(samples, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret/n

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

def generator(samples, batch_size):
    # Generates batches of image input and steering output for use with fit_generator()
    X = np.zeros((batch_size, resize_height, resize_width, img_channels),dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.float32)
    np.random.shuffle(samples)
    n_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        # Generate batch_size number of examples during each function call
        for idx in range(batch_size):
            keep_straight_steer = 0
            while not keep_straight_steer:
                # Sample random example and read steering angle
                sample_idx = np.random.randint(n_samples)
                batch_sample = samples[sample_idx]
                steer = float(batch_sample[3])
                if abs(steer)<steer_dev:
                    # Select steering angle close to zero with probability steer_prob
                    if abs(np.random.uniform())> steer_prob:
                        keep_straight_steer = 1
            # Choose the image corresponding to the steering angle
            image, y[idx] = select_image(batch_sample)
            # Pre-process selected image and add to image set
            X[idx] = process_image(image)
        # Return batch of images and steering angles
        yield sklearn.utils.shuffle(X, y)

def train_model():
    # Split samples into training and validation data sets
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    # Keras callback functions used during training
    # Stop training if validation accuracy decreases after 'patience' epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # Save best model based on validation accuracy
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', save_best_only=True, save_weights_only=False,
                                       mode='auto')

    # Create driving network model
    if os.path.isfile(checkpoint_file):
        model = load_model(checkpoint_file)
        print("Using existing model")
    else:
        # Load a new instance of the network
        if ALVINN:
            model = alvinn(resize_height, resize_width, img_channels)
            print("Using ALVINN model")
        else:
            model = nvidia(resize_height, resize_width, img_channels)
            print("Using NVIDIA model")

    # Train model
    steps_per_epoch = np.math.ceil(len(train_samples) / batch_size)
    validation_steps = np.math.ceil(len(validation_samples) / batch_size)
    adam = Adam(lr=learn_rate)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    train_history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_generator,
                                        validation_steps=validation_steps, epochs=n_epochs,
                                        callbacks=[early_stopping, model_checkpoint])
    print("Training completed.")
    return train_history

def visualise_network():
    # Use the functions in visualise.py to display each network layer
    if os.path.isfile(checkpoint_file):
        model = load_model(checkpoint_file)
    idx= np.random.randint(len(samples))
    visual_sample = samples[idx]
    image = cv2.imread(img_dir + visual_sample[0].split('/')[-1])
    image = process_image(image)
    plt.imshow(image)
    activation_maps = get_activations(model, [image], print_shape_only=True)
    display_activations(activation_maps)

def read_data():
    # Read in contents of driving data CSV file
    samples = []
    with open(data_dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    # Remove first line containing column headings
    del samples[0]
    return samples

if __name__ == "__main__":
    # Read training data
    samples = read_data()
    # Display data statistics
    analyse_data(samples)
    # Train neural network
    train_history = train_model()
    # Display network activations
    #visualise_network()
    # Display training performance
    plot_history(train_history)