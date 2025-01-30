# Pedestrian_Detection
The purpose of this work is to train a supervised classifier, specifically a Support Vector Machine (SVM), using labeled images of pedestrians and non-pedestrians. 
Pedestrian images are pre-sized and contain only the object of interest, extracted from larger images with multiple people. Non-pedestrian images are also pre-sized and contain various traffic-related objects or areas without pedestrians, such as buildings, sky, trees, vehicles, asphalt, and fences. For each image, the Histogram of Oriented Gradients (HOG) feature is extracted, creating feature sets for both pedestrian and non-pedestrian images. These labeled feature sets are then used to train an SVM classifier. 
Once the trained model is obtained, it is tested on negative images (without pedestrians) using a sliding window approach of the same size as the training images (128x64), scanning at different scales. This process, known as hard negative mining, identifies regions or objects that the model misclassifies. If the model incorrectly detects pedestrians, the corresponding HOG features are collected to form a set of hard negative instances, which are then used to retrain the SVM classifier. This hard negative mining process is performed only once, as repeating it does not significantly improve model performance. 
Finally, the model is evaluated using test images containing pedestrians in different environments and positions, employing the sliding window technique along with pyramid scaling for multi-scale detection.

Files posted:
- INRIA Person Dataset used for training and testing
- the script for SVM training (svm_liniar_final_1206.m)
- the main code which loads the trained SVM and run it on a new/test image (Ped_Det_RUN.m)
- the Non-Maximum Suppression for sliding windows (nms.m)
- personal implementation of computing HOG Features (but not used for SVM training)

The entire code is written in Matlab.
