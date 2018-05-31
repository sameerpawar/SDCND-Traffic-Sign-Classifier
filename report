# **Traffic Sign Recognition** 

## Writeup

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32 32]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart.

![](/home/sameerp/Documents/Udacity/SDCND/Term-1/CarND-Traffic-Sign-Classifier-Project/histogram_training.png )
**"Hisogram of percentage count of different classes in training set"**

![](/home/sameerp/Documents/Udacity/SDCND/Term-1/CarND-Traffic-Sign-Classifier-Project/histogram_validation.png)  
**"Hisogram of percentage count of different classes in validation set"**

Also, the distirbution of different classes in the training set are as follows

 [ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920  690
  540  360  990 1080  180  300  270  330  450  240 1350  540  210  480  240
  390  690  210  599  360 1080  330  180 1860  270  300  210  210]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried both color and gray scale images and found very little performance difference. So either types of images, i.e., color or grayscale, are okay with the network architecture I tried.

After grayscale conversion, I normalized the data per channel using the following equation 

$$\text{img}[ch] = \frac{\text{img}[ch] - \min(\text{img}[ch])}{\max(\text{img}[ch])-\min(\text{img}[ch])}$$


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   							| 
| Convolution 5x5x6     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400 x 120        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| 120 x 43 (# of classes)        									|
| Softmax				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used
* EPOCHS = 20
* BATCH_SIZE = 128
* Learning rate = 0.001
* AdamOptimizer
* Dropout_prob = 0.65


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of  0.951 
* test set accuracy of  0.938

I tried two network architectures, LeNet and modified LeNet. I used a global flag to choose the model as needed. I also used drop out for regularization. I played with fine tuning drop out prob and found a sweet spot near 0.65.  I did not find a significant difference between the two architectures, but slightly better performance for modified LeNet. The modification has to do with removing one layer of fully connected layer from LeNet. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

I downloaded eight German traffic signs that I found on the web. Please find them in folder ./traffic-signs-data/webimages/

The class labels for these web images are 

[23, 23, 28, 1, 9, 28, 23, 28]


I had to resize the images to fit 32x32x3 size. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 2 of the 8 traffic signs, which gives an accuracy of 25%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

If I consider success if the correct class belongs to top-5 softmax probabilities, then my model provided following outcome for true labels being  [23, 23, 28, 1, 9, 28, 23, 28]. Thus identifying correctly 6 out of 8 images.

[array([[12, 40, 20, 26, 37],
       [20, 23, 10,  9, 17],
       [30, 11, 28, 12, 23],
       [31, 25, 21,  1,  2],
       [26, 40, 12, 18, 24],
       [28,  3, 35, 36, 11],
       [11, 23, 40, 21, 42],
       [21, 28, 31, 23, 11]], dtype=int32)]


