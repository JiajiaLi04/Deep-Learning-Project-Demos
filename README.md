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
     <br>Fig.1 model architecture
</p>

1.3 Result analysis
------
Creates a 2-D plot of the distributed representation space using an algorithm called t-SNE. Nearby points in the 2-D space are meant to correspond to nearby points in the 16-D word embedding space. From the learned model, we can create pictures of 2D visualization like figure 2.

<p align="center">
     <img src="docs/2D visualization.png" alt="model architecture" width="60%" height="60%">
     <br>Fig.1 2D visualization
</p>
