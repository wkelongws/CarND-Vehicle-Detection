##Vehicle Detection
###Detect and locate vehicle in a project video

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_features_0]: ./output_images/car_features_0.png
[car_features_1]: ./output_images/car_features_1.png
[car_features_2]: ./output_images/car_features_2.png
[car_features_3]: ./output_images/car_features_3.png
[car_features_4]: ./output_images/car_features_4.png
[non_car_features_0]: ./output_images/non_car_features_0.png
[non_car_features_1]: ./output_images/non_car_features_1.png
[non_car_features_2]: ./output_images/non_car_features_2.png
[non_car_features_3]: ./output_images/non_car_features_3.png
[non_car_features_4]: ./output_images/non_car_features_4.png
[car_searching_1]: ./output_images/car_searching_1.png
[car_searching_3]: ./output_images/car_searching_3.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###There are three major tasks in my project: 1. building a car classifier; 2. frame-wise  car search; and 3. video implementation

###Car classifier

####Features

As inspired by the lecture videos, I decided to used all three kinds of features in my model: 1. space_binned pixel intensity, 2. histogram of all color space channels and 3. HOG features.

Before extracting all the features, the first step is decide which color space I am gona working on. After reading through people's discussion on slack and experimenting a little a bit, I decided to use LUV as suggested by a lot people. These two figure shows the features I extracted from two sample car images using LUV color space and a certain combination of hyperparameters:
![alt text][car_feature_0]
![alt text][car_feature_1]

Correspondingly, these two figures show the features I extracted from two sample non_car images using LUV color space and the same combination of hyperparameters:
![alt text][non_car_feature_0]
![alt text][non_car_feature_1]

As shown in the above figures, the features finally used in building the classifier are the 2nd, 3rd and 4th columns. All the pixel intensities in LUV channels of the reduced size image, histogram values of all LUV channels of the raw_size image and the HOG features of all LUV channels of the raw_size image are vectorized into a long vetor which is one training sample for the classifier.

To find the corresponding codes for the described tasks above, you can search: 

* "define feature extraction function for experiment"
* "load test images"
* "features visualization"

in "Vehicle_Detection_visualization_module.ipynb".


####Search for the best hyperparameters for feature extraction

After I decided to use all three kinds of features described above in my model, I need to find the best hyperparameters to use for extracting features.
The hyperparameters include: hist_bins, small_size, orientations, pix_per_cell,cell_per_block. The description and the experimented values for these hyperparameters are shown in the table below:

| hyperparameter | description                       | values |     |      |
|----------------|-----------------------------------|--------|-----|------|
| hist_bins      | number of bins for histogram      | "64    | 128 | 256" |
| small_size     | size of reduced size image        | "10    | 20" |      |
| orientations   | number of orientations for HOG    | "6     | 9   | 12"  |
| pix_per_cell   | number of pixels per cell for HOG | "8     | 12" |      |
| cell_per_block | number of cells per block for HOG | "1     | 2"  |      |

I made a grid search using all the combinations of the hyperparameters shown in the above table. Training and test sample are split from the vehicle and non_vehicle images provided by Udacity. Features are extracted from the training samples using each combination of the hyperparameters and are feed to a linear SVM for training. The trained SVM are tested on the test samples and the model performance are shown in this table below:

| rank | hist_bins | small_size | orientations | pix_per_cell | cell_per_block | feature_number | True Positives | True Negatives | False Negatives | False Positives | score  |
|------|-----------|------------|--------------|--------------|----------------|----------------|----------------|----------------|-----------------|-----------------|--------|
| 1    | 128       | 20         | 12           | 8            | 2              | 8640           | 99.88%         | 99.68%         | 0.32%           | 0.12%           | 99.77% |
| 2    | 64        | 20         | 9            | 8            | 2              | 6684           | 99.65%         | 99.78%         | 0.22%           | 0.35%           | 99.72% |
| 3    | 64        | 20         | 12           | 8            | 1              | 3696           | 99.88%         | 99.57%         | 0.43%           | 0.12%           | 99.72% |
| 4    | 64        | 20         | 12           | 8            | 2              | 8448           | 99.88%         | 99.57%         | 0.43%           | 0.12%           | 99.72% |
| 5    | 128       | 20         | 12           | 8            | 1              | 3888           | 99.88%         | 99.57%         | 0.43%           | 0.12%           | 99.72% |
| 6    | 64        | 10         | 12           | 8            | 2              | 7548           | 99.88%         | 99.46%         | 0.54%           | 0.12%           | 99.66% |
| 7    | 64        | 20         | 6            | 8            | 2              | 4920           | 99.65%         | 99.67%         | 0.33%           | 0.35%           | 99.66% |
| 8    | 64        | 20         | 9            | 8            | 1              | 3120           | 99.76%         | 99.57%         | 0.43%           | 0.24%           | 99.66% |
| 9    | 128       | 20         | 6            | 8            | 2              | 5112           | 99.53%         | 99.78%         | 0.22%           | 0.47%           | 99.66% |
| 10   | 128       | 20         | 9            | 8            | 2              | 6876           | 99.53%         | 99.78%         | 0.22%           | 0.47%           | 99.66% |
| 11   | 256       | 20         | 9            | 8            | 2              | 7260           | 99.53%         | 99.78%         | 0.22%           | 0.47%           | 99.66% |
| 12   | 256       | 20         | 12           | 8            | 1              | 4272           | 99.65%         | 99.67%         | 0.33%           | 0.35%           | 99.66% |
| 13   | 256       | 20         | 12           | 8            | 2              | 9024           | 99.65%         | 99.67%         | 0.33%           | 0.35%           | 99.66% |
| 14   | 64        | 10         | 9            | 8            | 2              | 5784           | 99.65%         | 99.57%         | 0.43%           | 0.35%           | 99.61% |
| 15   | 64        | 10         | 12           | 8            | 1              | 2796           | 99.76%         | 99.46%         | 0.54%           | 0.24%           | 99.61% |
| 16   | 64        | 20         | 12           | 12           | 2              | 3696           | 99.76%         | 99.46%         | 0.54%           | 0.24%           | 99.61% |
| 17   | 128       | 10         | 6            | 8            | 2              | 4212           | 99.53%         | 99.67%         | 0.33%           | 0.47%           | 99.61% |
| 18   | 128       | 10         | 12           | 8            | 1              | 2988           | 99.76%         | 99.46%         | 0.54%           | 0.24%           | 99.61% |
| 19   | 128       | 10         | 12           | 8            | 2              | 7740           | 99.76%         | 99.46%         | 0.54%           | 0.24%           | 99.61% |
| 20   | 128       | 20         | 9            | 8            | 1              | 3312           | 99.65%         | 99.57%         | 0.43%           | 0.35%           | 99.61% |
| 21   | 256       | 10         | 12           | 8            | 2              | 8124           | 99.65%         | 99.57%         | 0.43%           | 0.35%           | 99.61% |
| 22   | 256       | 20         | 9            | 8            | 1              | 3696           | 99.65%         | 99.57%         | 0.43%           | 0.35%           | 99.61% |
| 23   | 64        | 10         | 6            | 8            | 2              | 4020           | 99.65%         | 99.46%         | 0.54%           | 0.35%           | 99.55% |
| 24   | 128       | 10         | 9            | 8            | 1              | 2412           | 99.65%         | 99.46%         | 0.54%           | 0.35%           | 99.55% |
| 25   | 128       | 10         | 9            | 8            | 2              | 5976           | 99.53%         | 99.57%         | 0.43%           | 0.47%           | 99.55% |
| 26   | 128       | 20         | 6            | 8            | 1              | 2736           | 99.53%         | 99.57%         | 0.43%           | 0.47%           | 99.55% |
| 27   | 128       | 20         | 12           | 12           | 2              | 3888           | 99.76%         | 99.35%         | 0.65%           | 0.24%           | 99.55% |
| 28   | 256       | 10         | 9            | 8            | 1              | 2796           | 99.53%         | 99.57%         | 0.43%           | 0.47%           | 99.55% |
| 29   | 256       | 20         | 12           | 12           | 2              | 4272           | 99.65%         | 99.46%         | 0.54%           | 0.35%           | 99.55% |
| 30   | 64        | 10         | 9            | 8            | 1              | 2220           | 99.65%         | 99.35%         | 0.65%           | 0.35%           | 99.49% |
| 31   | 64        | 10         | 12           | 12           | 2              | 2796           | 99.65%         | 99.35%         | 0.65%           | 0.35%           | 99.49% |
| 32   | 64        | 20         | 12           | 12           | 1              | 2292           | 99.65%         | 99.35%         | 0.65%           | 0.35%           | 99.49% |
| 33   | 256       | 10         | 9            | 8            | 2              | 6360           | 99.41%         | 99.57%         | 0.43%           | 0.59%           | 99.49% |
| 34   | 256       | 10         | 12           | 8            | 1              | 3372           | 99.65%         | 99.35%         | 0.65%           | 0.35%           | 99.49% |
| 35   | 256       | 10         | 12           | 12           | 2              | 3372           | 99.53%         | 99.46%         | 0.54%           | 0.47%           | 99.49% |
| 36   | 256       | 20         | 9            | 12           | 2              | 3696           | 99.41%         | 99.57%         | 0.43%           | 0.59%           | 99.49% |
| 37   | 64        | 10         | 12           | 12           | 1              | 1392           | 99.53%         | 99.35%         | 0.65%           | 0.47%           | 99.44% |
| 38   | 64        | 20         | 6            | 8            | 1              | 2544           | 99.41%         | 99.46%         | 0.54%           | 0.59%           | 99.44% |
| 39   | 64        | 20         | 9            | 12           | 2              | 3120           | 99.53%         | 99.35%         | 0.65%           | 0.47%           | 99.44% |
| 40   | 128       | 10         | 12           | 12           | 2              | 2988           | 99.53%         | 99.35%         | 0.65%           | 0.47%           | 99.44% |
| 41   | 128       | 20         | 9            | 12           | 1              | 2259           | 99.41%         | 99.46%         | 0.54%           | 0.59%           | 99.44% |
| 42   | 128       | 20         | 9            | 12           | 2              | 3312           | 99.41%         | 99.46%         | 0.54%           | 0.59%           | 99.44% |
| 43   | 128       | 20         | 12           | 12           | 1              | 2484           | 99.41%         | 99.46%         | 0.54%           | 0.59%           | 99.44% |
| 44   | 256       | 10         | 6            | 8            | 2              | 4596           | 99.41%         | 99.46%         | 0.54%           | 0.59%           | 99.44% |
| 45   | 256       | 20         | 6            | 8            | 1              | 3120           | 99.53%         | 99.35%         | 0.65%           | 0.47%           | 99.44% |
| 46   | 256       | 20         | 6            | 8            | 2              | 5496           | 99.30%         | 99.57%         | 0.43%           | 0.70%           | 99.44% |
| 47   | 256       | 20         | 12           | 12           | 1              | 2868           | 99.53%         | 99.35%         | 0.65%           | 0.47%           | 99.44% |
| 48   | 64        | 20         | 9            | 12           | 1              | 2067           | 99.41%         | 99.35%         | 0.65%           | 0.59%           | 99.38% |
| 49   | 128       | 10         | 6            | 8            | 1              | 1836           | 99.41%         | 99.35%         | 0.65%           | 0.59%           | 99.38% |
| 50   | 128       | 10         | 12           | 12           | 1              | 1584           | 99.41%         | 99.35%         | 0.65%           | 0.59%           | 99.38% |
| 51   | 256       | 20         | 9            | 12           | 1              | 2643           | 99.41%         | 99.35%         | 0.65%           | 0.59%           | 99.38% |
| 52   | 64        | 10         | 9            | 12           | 2              | 2220           | 99.53%         | 99.14%         | 0.86%           | 0.47%           | 99.32% |
| 53   | 128       | 10         | 9            | 12           | 1              | 1359           | 99.41%         | 99.24%         | 0.76%           | 0.59%           | 99.32% |
| 54   | 128       | 10         | 9            | 12           | 2              | 2412           | 99.41%         | 99.24%         | 0.76%           | 0.59%           | 99.32% |
| 55   | 128       | 20         | 6            | 12           | 2              | 2736           | 99.41%         | 99.24%         | 0.76%           | 0.59%           | 99.32% |
| 56   | 256       | 10         | 12           | 12           | 1              | 1968           | 99.41%         | 99.24%         | 0.76%           | 0.59%           | 99.32% |
| 57   | 64        | 10         | 6            | 8            | 1              | 1644           | 99.30%         | 99.24%         | 0.76%           | 0.70%           | 99.27% |
| 58   | 64        | 20         | 6            | 12           | 2              | 2544           | 99.30%         | 99.24%         | 0.76%           | 0.70%           | 99.27% |
| 59   | 256       | 10         | 9            | 12           | 2              | 2796           | 99.18%         | 99.35%         | 0.65%           | 0.82%           | 99.27% |
| 60   | 256       | 10         | 6            | 8            | 1              | 2220           | 99.29%         | 99.14%         | 0.86%           | 0.71%           | 99.21% |
| 61   | 256       | 10         | 9            | 12           | 1              | 1743           | 99.18%         | 99.24%         | 0.76%           | 0.82%           | 99.21% |
| 62   | 64        | 10         | 6            | 12           | 2              | 1644           | 99.18%         | 99.13%         | 0.87%           | 0.82%           | 99.16% |
| 63   | 128       | 10         | 6            | 12           | 2              | 1836           | 99.29%         | 99.03%         | 0.97%           | 0.71%           | 99.16% |
| 64   | 256       | 20         | 6            | 12           | 2              | 3120           | 99.29%         | 99.03%         | 0.97%           | 0.71%           | 99.16% |
| 65   | 64        | 10         | 9            | 12           | 1              | 1167           | 99.41%         | 98.82%         | 1.18%           | 0.59%           | 99.10% |
| 66   | 256       | 10         | 6            | 12           | 2              | 2220           | 99.29%         | 98.92%         | 1.08%           | 0.71%           | 99.10% |
| 67   | 64        | 20         | 6            | 12           | 1              | 1842           | 99.18%         | 98.92%         | 1.08%           | 0.82%           | 99.04% |
| 68   | 128       | 20         | 6            | 12           | 1              | 2034           | 99.18%         | 98.92%         | 1.08%           | 0.82%           | 99.04% |
| 69   | 256       | 20         | 6            | 12           | 1              | 2418           | 99.18%         | 98.92%         | 1.08%           | 0.82%           | 99.04% |
| 70   | 128       | 10         | 6            | 12           | 1              | 1134           | 99.41%         | 98.60%         | 1.40%           | 0.59%           | 98.99% |
| 71   | 64        | 10         | 6            | 12           | 1              | 942            | 99.17%         | 98.71%         | 1.29%           | 0.83%           | 98.93% |
| 72   | 256       | 10         | 6            | 12           | 1              | 1518           | 99.17%         | 98.71%         | 1.29%           | 0.83%           | 98.93% |

The top 5 ranked feature combinations show the best model results (Actually all feature combinations performed very well). Amoung the top 5, the third combination yeilds the less total number of features and thus I selected the linear SVM classifier trained on this combination for my project. As shown in the table, the selected combination yeilds 3696 features in total and 99.72% total classification accuracy.

To find the corresponding codes for the described tasks above, you can search: 

* "define feature extraction function"
* "efine classifier training function"
* "define classifier evaluation function"
* "train and evaluate the classifier"
* "save trained classifier"
* "model hyper-parameters grid search experiment"

in "Vehicle_Detection_visualization_module.ipynb".


###frame-wise  car search

The two examples here inlustrates the entire car search pipeline:
![alt text][car_searching_3]
![alt text][car_searching_1]

As shown in the top-center image, I want to minimized the total number of search windows. I used 6 different sizes of searching windows. For each size of searching windows, I limited the searching area to a box on the corresponding part of the road. For example, when I use a larger searching window, the searching area covers the lower part of the image more and when I use a smaller searching window the searching area focus more on near the vanishing point.

Then each searching window are resized and extracted into 3696 features as described above and feed to the trainer linear SVM classifier. The searching windows classified as "car" are shown in the top-right image.

All the detected car searching windows contribute to the heatmap as shown in the bottom-left image. The heatmap is thresholded to remove the false positives from the image.

The bottem-center image shows the result after applying label function on the filtered heatmap.

And finally the labeled boxes are dawn back on the input image.

To find the corresponding codes for the described tasks above, you can search: 

* "define boxes drawing function"
* "initial global vairables"
* "run car finder"
* "visualize outputs from car finder step by step"

in "Vehicle_Detection_visualization_module.ipynb".


---

### Video Implementation

Since the project video is the same one with the last project. I combined the two project and provided an output video with both lane detection and car detection. 

Here's a [link to my video result](https://youtu.be/dozR4N9QvIs)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

