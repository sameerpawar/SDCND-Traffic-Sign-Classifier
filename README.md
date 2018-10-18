# **German Traffic Sign Classification** 
[//]: # (Image References)
[classhist]: ./class_stats_histogram.jpg "Class histogram"
[ex_class_images]: ./train_images.jpg
[histEqual_images]: ./histogram_equalized_image.jpg
[rotated_images]: ./rotate_image.jpg
[translated_images]: ./translate_image.jpg
[sheared_images]: ./shear_image.jpg
[motion_blurred_images]: ./motion_blur_image.jpg
[brightness_images]: ./change_brightness_image.jpg

---
## Basic summary of the data set
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of test examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### Histogram of distribution of samples per class in a training and validation data sets

![class histogram][classhist]

## Visual exploration
Next, we take look at 10 random example images from 5 random classes.

![class 5][ex_class_images]


## Data Augmentation
Visual observation of the sample images from the training data set for different classes shows that dataset contains images that are 1) dark, 2) blurred 3) tilted 4) sheared 5) off center. In order to provide our model an opportunity to learn and master these traits, we perform image data augmentation. The data augmentation is performed during run time based on random bits, it has following effects:
1. Training data size in theory becomes infinite, allowing model to train without memorizing all the data set.
1. No lengthy time consuming offline data crunching or huge memory requirements to load augmented data.
1. We can control the statistical characteristics of different augmentation in training data fed to model, while providing equal data augementation opportunity to all classes.

Next, we visualize the augmented data
### <u>Histogram equalization and brightness perturbation</u>
Histogram equalization is a method in image processing of contrast adjustment by adjusting the histogram of pixel intensities in an image to follow certain shape. The method is particularly useful in images with backgrounds and foregrounds that are both bright or both dark. This allows for areas of lower local contrast to gain a higher contrast. 

In addition to histogram equalization (which takes care of local changes), we also perturb the overall brightness of an image by a random amount to generalize the model to be less sensitive to intensities while performing classification task.

Here, are some examples of histogram equalized and brightness perturbed images

![hist equalized images][histEqual_images]
![brightness changed images][brightness_images]


### <u>Rotation</u>
We apply random rotatation of angle [-15, +15] degrees around the center of image. Here are some examples of original images and rotation applied images. 

![rotated images][rotated_images]


### <u>Translation</u>
We apply random [-2, +2] pixel translation in both x-y directions.

Here are some examples of original images and translation applied images. 

![translated images][translated_images]

### <u>Shear and Motion blur, </u>
Below are some examples of original images and shear and motion blur applied images. 

![sheared images][sheared_images]
![motion blur images][motion_blurred_images]


## Model Architecture
I tried two models. LeNet and simpler version of VGG. As an interface for both models,I used a 1x1 conv to convert a 3-channel color image to a generalized mono channel image input. The two models then process this single channel 32 x 32 image for classification task. 

### <b> LeNet </b>

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| conv [1x1x3, 1]       | valid padding, output: 32x32x1|
| Linear activation     | output: 32x32x1|
| conv [5x5x1, 6]     	| valid padding, output: 28x28x6            	|
| maxpool [2x2]         | stride = 2, output: 14x14x6            	|
| ReLu activation + batch normalization       | output: 14x14x6|
| conv [5x5x6, 16]     	| valid padding, output: 10x10x16            	|
| maxpool [2x2]         | stride = 2, output: 5x5x16            	|
| ReLu activation + batch normalization       | output: 5x5x16|
| Fully connected [400 x 120]| output: 120 					|
| ReLu activation + batch normalization       | output: 120|
| dropout       | output: 120|
| Fully connected [120 x 84]| output: 84 					|
| ReLu activation + batch normalization       | output: 84|
| dropout       | output: 84|
| Fully connected [84 x 43]| output: 43 					|
| Softmax        | output: 43 class probabilities
 

### <b>VGG </b>
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| conv [1x1x3, 1]       | valid padding, output: 32x32x1|
| Linear activation     | output: 32x32x1|
| conv [3x3x1, 32]     	| valid padding, output: 30x30x32            	|
| ReLu activation + batch normalization       | output: 30x30x32|
| dropout       | output: 30x30x32|
| conv [3x3x32, 32]     	| valid padding, output: 28x28x32            	|
| ReLu activation + batch normalization       | output: 28x28x32|
| dropout       | output: 28x28x32|
| maxpool [2x2]         | stride = 2, output: 14x14x32            	|
| conv [3x3x32, 64]     	| valid padding, output: 12x12x64            	|
| ReLu activation + batch normalization       | output: 12x12x64|
| dropout       | output: 12x12x64|
| conv [3x3x64, 64]     	| valid padding, output: 10x10x64            	|
| ReLu activation + batch normalization       | output: 10x10x64|
| dropout       | output: 10x10x64|
| maxpool [2x2]         | stride = 2, output: 5x5x64            	|
| conv [3x3x64, 128]     	| valid padding, output: 3x3x128            	|
| ReLu activation + batch normalization       | output: 3x3x128|
| dropout       | output: 3x3x128|
| conv [3x3x128, 128]     	| valid padding, output: 1x1x128            	|
| ReLu activation + batch normalization       | output: 1x1x128|
| dropout       | output: 1x1x128|
| Fully connected [128 x 128]| output: 128 					|
| ReLu activation + batch normalization       | output: 128
| dropout       | output: 128|
| Fully connected [128 x 84]| output: 84 					|
| ReLu activation + batch normalization       | output: 84|
| dropout       | output: 84|
| Fully connected [84 x 43]| output: 43 					|
| Softmax        | output: 43 class probabilities


#### Hyper-parameters
1. optimizer: Adam optimizer with learning rate 1e-3
1. batch size = 64
1. number of epochs = 100
1. dropouts: 0.5
    

### <b> Final model (VGG) results </b>
* training set accuracy = 1
* validation set accuracy = 0.99
* test set accuracy = 0.98
 

