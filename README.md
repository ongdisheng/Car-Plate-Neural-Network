# Car-Plate-Neural-Network
## Introduction
This repository contains an implementation of a three-layered neural network designed to classify typical Malaysian car number plates. The neural network is trained to recognize 10 alphabets and 10 numerals commonly found on these plates. 

## Project Overview
### 1. Dataset
To form the dataset, ten specific alphabets (B, F, L, M, P, Q, T, U, V, W) and ten numerals (0-9) are chosen to represent the characters commonly found on Malaysian car number plates. Each alphabet and numeral consists of ten images, resulting in a total of 100 alphabet images and 100 numeral images.

### 2. Neural Network Training
The neural network is trained using 80% of the dataset. This involves selecting eight images of each alphabet and numeral as training inputs to train the network's classification capabilities.

### 3. Neural Network Testing
Once the neural network is trained, the neural network is tested using the remaining 20% of the dataset, which consists of two images of each alphabet and numeral. This testing phase evaluates the network's accuracy in classifying unseen images. 

### 4. Car Number Plate Classification
In this step, an automatic segmentation algorithm is employed to separate individual characters within a given Malaysian car number plate. For example, the plate "VBU 3878" is segmented into "V", "B", "U", "3", "8", "7", and "8". Each segmented character is then presented to the trained neural network for classification. The accuracy of the classification is observed and recorded for each character.

## Directory structure
1. checkpoint - This directory holds the .npy files that store the optimum weights and biases of the neural network. 
2. data - This directory contains the training and testing datasets, consisting of 200 images.
3. plate - This directory includes the 10 given images that will be used for segmentation. 
4. segmented - This directory stores the segmented images obtained from the 10 given images.
5. Assignment3.py - The main program file containing the implementation of the neural network.

## Code Execution
To run the code in this repository, follow these steps:

1. Ensure that the required dependencies are installed.
2. Run the `Assignment3.py` file.
3. By default, the code utilizes a pre-trained neural network with saved weights and biases to execute the `test`, `segment`, and `predict` functions. If you want to train the neural network from scratch, kindly uncomment the `train` function call in the `Assignment3.py` file.
