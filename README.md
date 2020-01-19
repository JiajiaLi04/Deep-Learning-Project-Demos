1.A Multi-layer Percetron for Natural Language Processing 
===============
1.1 Overview
-------
This project will train a multi-layered perceptron for natural language processing. The task is to predict the next word in a sentence given a sequence of words. 

1.2 Network Architecture
-------
This project will train a neural language model using a multi-layered perceptron like figure 1. It receives 3 consecutive words as the input and aims to predict a distribution over the next word. The model is trained by using the cross-entropy criterion, which is equivalent to maximizing the probability it assigns to the target words in the training set. 

<p align="center">
     <img src="docs/network architecture.png" alt="model architecture" width="60%" height="60%">
     <br>Fig.1 Model Architecture
</p>

1.3 Result Analysis
------
Creates a 2-D plot of the distributed representation space using an algorithm called t-SNE. Nearby points in the 2-D space are meant to correspond to nearby points in the 16-D word embedding space. From the learned model, we can create pictures of 2D visualization like figure 2. The common of these words in clusters is that they have similar words property and similar usage. For instance, the word cluster of ‘would, should, could, can, will, may, might’ is all modal verbs, and the word cluster of ‘my, your, our, their, his’ is all possessive adjectives.

<p align="center">
     <img src="docs/2D visualization.png" alt="model architecture" width="60%" height="60%">
     <br>Fig.2 2D Visualization
</p>

By using the model to predict the next word, the result is:
- Input: ‘government of united’. Output: ‘state’ has the highest probability(0.47774) to be the next word.
- Input: ‘city of new’. Output: ‘york’ has the highest probability(0.98458) to be the next word.
- Input: ‘life in the’. Output: ‘world’ has the highest probability(0.18037) to be the next word.
- Input: ‘they could make’. Output: ‘it’ has the highest probability(0.59879) to be the next word.
- Input: ‘we do nt ’. Output: ‘know’ has the highest probability(0.24101) to be the next word.


2.Image Coloraztion 
===============
2.1 Overview
-------
The task is to train a convolutional neural network known as image colorization. That is, given a gray scale image, we wish to predict the color at each pixel. This a difficult problem for many reasons, one of which being that it is ill-posed: for a single gray scale image, there can be multiple, equally valid colorings.

2.2 Colorization as Regression
--------
A simple approach is to frame it as a regression problem, where we build a model to predict the RGB intensities at each pixel given the gray scale input. In this case, the outputs are continuous, and so squared error can be used to train the model. The validation output of the color regression is as figure 3.

<p align="center">
     <img src="docs/color regression.png" alt="model architecture" width="60%" height="60%">
     <br>Fig.3 Validation Output of the Color Regression
</p>

2.2 Colorization as Classification
--------
We will select a subset of 24 colors and frame colorization as a pixel-wise classification problem, where we label each pixel with one of 24 colors. The 24 colors are selected using k-means clusteringa over colors, and selecting cluster centers.The validation output of the color classification is as figure 4.

<p align="center">
     <img src="docs/color classification.png" alt="model architecture" width="60%" height="60%">
     <br>Fig.4 Validation Output of the Color Classification
</p>

2.2 Colorization Using Skip Connection
--------
A skip connection in a neural network is a connection which skips one or more layer and connects to a later layer. Add a skip connection from the first layer to the last, second kater to the second last, etc. That is, the final convolution should have both the output of the previous layer and the initial gray scale input as input. A common type of skip-connection is introduced by Unet.  The validation output of the colorization using Unet is as figure 5.

<p align="center">
     <img src="docs/color Unet.png" alt="model architecture" width="60%" height="60%">
     <br>Fig.5 Validation Output of the Colorization Using Unet
</p>



