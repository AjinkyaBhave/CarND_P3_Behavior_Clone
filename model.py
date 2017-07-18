import csv
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os.path
from drive_networks import alvinn
from process_images import brighten_img

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
steer_offset = 0.001
# Data directory containing images and control measurements
data_dir = './dataset/initial/'
img_dir = data_dir+'IMG/'
# File to save current best network with weights
checkpoint_file = './model_checkpoint/model.h5'

## Image parameters
img_height = 160
img_width  = 320
img_channels = 3
crop_top = 50     # number of pixel rows to remove from image top
crop_bottom = 20  # number of pixel rows to remove from image bottom
crop_height = img_height-crop_top-crop_bottom

def process_image(X):
    # Normalise pixels between (-0.5,0.5)
    X_proc = (X/255.0)-0.5
    return X_proc

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

# Read in contents of driving data CSV file
samples=[]
with open(data_dir + 'driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)
# Remove first line containing column headings
del samples[0]

#analyse_data(samples)

def generator(samples, batch_size):
    X = np.zeros((batch_size, crop_height, img_width, img_channels),dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.float32)
    num_samples = len(samples)
    print(samples[0])

    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            i=0
            batch_samples = samples[offset:offset+batch_size]
            for batch_sample in batch_samples:
                # Choose randomly which camera image to use
                img_id = np.random.randint(1)
                # Read steering command
                steer_centre = float(batch_sample[3])
                if img_id == 0:
                    image = cv2.imread(img_dir + batch_sample[0].split('/')[-1])
                    # Use original steering for the centre camera image
                    y[i] = steer_centre
                elif img_id == 1:
                    image = cv2.imread(img_dir + batch_sample[1].split('/')[-1])
                    # Create adjusted steering commands for the left camera image
                    y[i] = steer_centre + steer_offset
                else:
                    image = cv2.imread(img_dir + batch_sample[2].split('/')[-1])
                    # Create adjusted steering commands for the right camera image
                    y[i] = steer_centre - steer_offset

                # Convert image to RGB format since cv2.imread() uses BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Choose randomly whether to flip this image horizontally
                flip_prop = np.random.uniform()
                if flip_prop > 0.5:
                    # Flip the image half the time
                    image = np.fliplr(image)
                    # Correct steering for flipped image
                    y[i] = -y[i]
                # Brightness augmentation for robustness
                image = brighten_img(image)
                # Crop image  rows based on crop_top and crop_bottom parameters
                image = image[crop_top:img_height-crop_bottom, :, :]
                # Normalise pixels between (-0.5,0.5)
                X[i] = (image/255.0-0.5).astype(np.float32)
                i+=1
            yield sklearn.utils.shuffle(X, y)

# Split samples into training and validation data sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Keras callback functions used during training
# Stop training if validation accuracy decreases after 'patience' epochs
early_stopping = EarlyStopping(monitor='val_acc', patience=3)
# Save best model based on validation accuracy
model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='auto')

# Create driving network model
if os.path.isfile(checkpoint_file):
    model = load_model(checkpoint_file)
    #model.load_weights(checkpoint_file)
    print("Using existing network")
else:
    # Load a new instance of the network
    model = alvinn(crop_height, img_width, img_channels)
    print("Using new network")

# Train model
steps_per_epoch  = np.math.ceil(len(train_samples)/batch_size)
validation_steps = np.math.ceil(len(validation_samples)/batch_size)
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
train_history=model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator,
                    validation_steps=validation_steps, epochs=5, callbacks=[early_stopping,model_checkpoint])
print("Training completed.")
plot_history(train_history)