#**Behavioral Cloning** 

##Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/nvidia_network.png "NVIDIA Network"
[image2]: ./report/train_perf_t1.png  "Training Performance"
[image3]: ./report/steer_analysis.png "Steering Analysis"
[image4]: ./report/track1_images.jpg  "Example Images"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to train and test the model
* drive_networks.py containing the NVIDIA and ALVINN models
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1.mp4 video file showing successful lap around first track

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py ./model.h5
```
The video file is placed in ./video/track1.mp4 and shows the performance of the NVIDIA-based network around one lap of Track 1.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The supporting file drive_networks.py contains implementations for two driving networks, NVIDIA and ALVINN. Both are defined as functions and layers are commented for readability. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of the NVIDIA convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (drive_networks.py). The model includes RELU layers to introduce nonlinearity. I did not use ELU because the performance was good with RELU as is. 

I have also implemented the ALVINN neural network based on the work done by Dean Pomerleau, mainly to compare the performance of the state-of-art in 1995 with the deep networks we have today. However, I was not able to achieve a satisfactory performance with ALVINN, so NVIDIA is the final driving network that I have submitted for performance evaluation.

I preferred to normalise the data outside the model because I like my networks to implement the core learning algorithm and give them pre-processed images directly. This also allows me to modify the pre-processing pipeline externally without modifying the network architecture for future extensibility. I also read that some people had issues with keras lambda layers so I did not want to introduce unneeded bugs. 

####2. Attempts to reduce overfitting in the model

The model uses early stopping by monitoring the validation accuracy with a patience of 3 epochs to reduce over-fitting (model.py line 211). I used validation loss instead of training loss because I believe it is a better indicator of model generalisation and robustness as it is calculated on unseen data.

The model was trained and validated on different data sets to ensure that the model was not over-fitting (model.py lines 204-207). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam optimizer with a default learning rate of 0.001, so the learning rate was not tuned manually (model.py lines 28, 241). I read the Keras documentation and found out that the fully-connected layers were default initialised with the Glorot/Xavier uniform initialiser, which is quite good. So I did not experiment any further with different initial weights or bias settings for the dense layers. I used 10 epochs to start training and used 3-4 epochs each time I fine-tuned the model for certain road sections. I kept the batch size at a constant 32 throughout, since this did not seem to have much effect on the training error or speed.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving along the centre of the road, using the Udacity data as the initial set. I found that this was not enough to complete the track at all points, so I augmented the data with specific road scenarios. I collected more data by slowly drove along the dirt road section and the sharp turns around the striped road boundaries, since the network had difficulties with these parts of the track more often. I also created a small data set using my mouse and controlled centre lane driving to smoothen the network performance. 

For details about how I created the training data, see the next sub-section 3. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to choose a minimal and robust network that can generalise well. Since NVIDIA has a proven model that works well in practice, and it is not very large, I chose this as my starting point. The network has convolutional layers to derive features from the raw colour images and fully connected layers to derive steering commands from these features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used 20% of the data for validation (model.py line 204). Since I am using early stopping, the training will automatically stop once the validation error is not improving after a fixed number of epochs, thus preventing overfitting. After training the model for 10 epochs with the Udacity data, I found it was driving around the track reasonably well. 

The main problem areas were the dirt boundary and the sharp turn towards the end of the track with the striped boundary. To improve the driving behavior in these cases, I augmented the data for those sections of the road and retrained the model iteratively on these data sets. The training performance after fine-tuning is shown below. The very small gap between training and validation error shows that the network has not over-fitted on the training data.

![Training Performance][image2]

At the end of the training process, the vehicle is able to drive autonomously around Track 1 for multiple laps without leaving the road. I tested with speeds of  9 mph and 15 mph by changing the setpoint in drive.py, and the performance is similar at both speeds. The final video is placed at *./video/track1.mp4.* 


####2. Final Model Architecture

The final model architecture (drive_networks.py lines 22-42) consists of the NVIDIA convolution neural network with the following layers and layer sizes:

![NVIDIA Network][image1]


####3. Creation of the Training Set & Training Process

I analysed the Udacity data set by looking at the images and plotting the steering angles. The steering analysis below shows a large percentage of angles around zero, which implies a highly unbalanced data set. Training on this naively would lead to the vehicle learning to drive straight on almost all sections on the road. So I realised I would have to ensure that the training data had more number of turns examples to balance out the effect of the straight driving. The exact approach is described in the paragraphs on the creation of the Keras generator function for fit_generator() (model.py lines 184 to 209). 

![Steering Analysis][image3]

I used the Udacity data to train the vehicle initially for 10 epochs. I tested it in the simulator to see what parts of the track it fails on. Then I created training data specifically for those sections of the track and iteratively trained the network, lowering the learning rate (0.0001) and epochs (3-4). The final training data set contains 9417 examples without augmentation.

I used the Keras generator approach to allow the training process to pick up batches of examples from disk instead of storing everything in memory. Since the data is biased towards straight driving, the generator only selects straight driving examples with a probability of *steer_prob*, set at a default of 30% (model.py lines 34 and 202). This ensures that each batch contains a balanced mix of straight drive and turns. 

Since the car needs to know how to recover when close to the road edge, I used the idea suggested in the lectures and augmented the centre image with both left and right camera images. The generator randomly chooses whether which image to pick, in the select_image() function (model.py lines 110 to 124). I also corrected the steering offset for the non-centre images. I experimented with values between 0.2 to 0.3 and finally settled on 0.25 as a reasonable number for performance (model.py line 33).

Since the NVIDIA network has a large number of parameters, it would require more data than just the initial Udacity set. To augment this data, I chose to flip the image and brighten it randomly, to help the network deal with tree shadows on the track. I did not do further translation since the left/right cameras are already providing translated images. I also did not rotate or zoom the images. This is because rotation caused black borders for the missing pixels, which could make the network see those as features, and zooming would also confuse it since the steering angle would also have to be augmented for nearer images. I wanted to try shadow augmentation but since the network was performing adequately with flipping and brightening only, I did not implement anything further.

The process_image() function crops, resizes, and normalises the image so that the final 64x64 image has only the road ahead, without the trees and the car bonnet. This is then given as a training image to the network. The process_image() function is also called in drive.py (line 64) to ensure the same pre-processed images are given to the network during prediction. Examples of 20 randomly selected training images with augmentation applied are shown below.

![Training Images][image4]
