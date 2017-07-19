import csv
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os.path
from drive_networks import alvinn, nvidia
#from process_images import brighten_img

### PARAMETERS SECTION.
# Use ALVINN network if true. Else use Nvidia network.
ALVINN = 0

## Training parameters
# Batch size for generator function in training and validation
batch_size = 32
# Maximum steering angle in degrees
steer_max = 25
steer_min = -25
# Correction to steering for left and right images. Tunable.
steer_offset = 0.25
# Data directory containing images and control measurements
data_dir = './dataset/initial/'
img_dir = data_dir+'IMG/'
# File to save current best network with weights
checkpoint_file = './model_checkpoint/model.h5'

## Image parameters
img_height    = 160 # Original image rows
img_width     = 320 # Original image columns
img_channels  = 3   # RGB channels
resize_width  = 64  # Resized image rows
resize_height = 64  # Resized image columns
crop_top      = 50  # Number of pixel rows to remove from image top
crop_bottom   = 20  # Number of pixel rows to remove from image bottom
crop_height   = img_height-crop_top-crop_bottom # cropped image height

def analyse_data(samples, num_images=20):
    # Select ten random images to display for visual confirmation of processing
    img_idx = np.random.randint(0, len(samples), num_images)
    steer_data = []
    image_data = []
    cmap = None
    idx = 0
    for sample in samples:
        if idx in img_idx:
            image = cv2.imread(data_dir + sample[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[crop_top:img_height-crop_bottom,:,:]
            image = brighten_img(image)
            if ALVINN:
                image_sum = (image[:,:,0].astype(np.float32)+image[:,:,1].astype(np.float32)+image[:,:,2].astype(np.float32))
                image_sum[image_sum[:,:]<1.0]=1.0
                image = (image[:,:,2]/image_sum + image[:,:,2]/255.0).astype(np.float32)
                cmap = 'gray'
            image_data.append(image)
        steer_data.append(float(sample[3]))
        idx+=1

    fig, ax = plt.subplots(1)
    # Display histogram of steering control.
    hist, bins = np.histogram(steer_data, bins=20)
    ax.bar(bins[:-1], hist, width=0.5)

    ncols = 10
    nrows = np.math.ceil(num_images/ncols)
    # Display images in subplots
    fig, ax = plt.subplots(nrows, ncols, figsize=(64, 64))
    fig.subplots_adjust(hspace=.1, wspace=.05)
    ax = ax.ravel()
    for i in range(num_images):
        ax[i].imshow(image_data[i], cmap=cmap)
        ax[i].axis('off')
    plt.show()

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
    return image, steer

def brighten_img(img):
    # Randomly brighten image by up to 30%
    img_out = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    dbright = 0.3+np.random.uniform()
    img_out[:,:,2] = img_out[:,:,2]*dbright
    #img_out[img_out[:,:,2]>255] = 255
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2RGB)
    return img_out

def process_image(image, steer, mode = 'train'):
    # Flip and brighten images only during training phase
    if mode == 'train':
        # Choose randomly whether to flip this image horizontally
        flip_prop = np.random.uniform()
        if flip_prop > 0.5:
            # Flip the image half the time
            image = np.fliplr(image)
            # Correct steering for flipped image
            steer = -steer
        # Brightness augmentation for robustness
        image = brighten_img(image)
    # Crop image  rows based on crop_top and crop_bottom parameters
    image = image[crop_top:img_height - crop_bottom, :, :]
    # Resize image to reduce processing
    image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    # Normalise pixels between (-0.9,0.9)
    image = (image / 255.0 - 0.9).astype(np.float32)
    return image, steer

def generator(samples, batch_size):
    X = np.zeros((batch_size, resize_height, resize_width, img_channels),dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.float32)
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            i=0
            batch_samples = samples[offset:offset+batch_size]
            for batch_sample in batch_samples:
                steer_prob = np.random.uniform()
                image, steer = select_image(batch_sample)
                X[i], y[i] = process_image(image, steer)
                i+=1
            yield sklearn.utils.shuffle(X, y)

def train_model():
    # Split samples into training and validation data sets
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    # Keras callback functions used during training
    # Stop training if validation accuracy decreases after 'patience' epochs
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    # Save best model based on validation accuracy
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', save_best_only=True, save_weights_only=False,
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
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_generator,
                                        validation_steps=validation_steps, epochs=5,
                                        callbacks=[early_stopping, model_checkpoint])
    print("Training completed.")
    return train_history

if __name__ == "__main__":
    # Read in contents of driving data CSV file
    samples = []
    with open(data_dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    # Remove first line containing column headings
    del samples[0]
    # Display data statistics
    analyse_data(samples)
    # Train neural network
    train_history = train_model()
    # Display training performance
    plot_history(train_history)