import numpy as np
import cv2

### Augment the training data by translating, scaling, rotating, and brightning
def translate_img(img):
    # Randomly translate image by [-2, 2] pixels in x and y directions
    # Values taken from Sermanet paper
    rows,cols,_ = img.shape
    tx,ty = np.random.randint(-2,2,2)
    M = np.float32([[1,0,tx],[0,1,ty]])
    img_out = cv2.warpAffine(img,M,(cols,rows))
    return img_out

def rotate_img(img):
    # Randomly rotate image by [-15, 15] degrees around the center of the image
    rows,cols,_ = img.shape
    drot = np.random.randint(-15,15,1)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),drot,1)
    img_out = cv2.warpAffine(img,M,(cols,rows))
    return img_out

def scale_img(img):
    # Randomly scale image by a factor of [0.9, 1.1]
    # Values taken from Sermanet paper
    rows,cols,_ = img.shape
    dzoom = (0.2)*np.random.rand() + 0.9
    M = cv2.getRotationMatrix2D((cols/2,rows/2),0,dzoom)
    img_out = cv2.warpAffine(img,M,(cols,rows))
    return img_out

def brighten_img(img):
    # Randomly brighten image by up to 30%
    img_out = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    dbright = 0.3+np.random.uniform()
    img_out[:,:,2] = img_out[:,:,2]*dbright
    #img_out[img_out[:,:,2]>255] = 255
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2RGB)
    return img_out