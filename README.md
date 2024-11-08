# DeepLearning-CIFAR10

## Overview

**DeepLearning-CIFAR10** is a neural network project developed for the Neural Networks and Deep Learning coursework. The objective is to design, train, and optimize a convolutional neural network (CNN) capable of accurately classifying images from the CIFAR-10 dataset. Through iterative enhancements and advanced training techniques, the project achieved an improvement in classification accuracy from an initial 68% to 82.67% on the test set.

## Objectives

- **Image Classification:** Develop a CNN to classify CIFAR-10 images into 10 distinct categories.
- **Model Optimization:** Enhance the model architecture and training process to improve accuracy.
- **Performance Evaluation:** Monitor and visualize training loss and accuracy to assess model performance.

## Key Findings

- **Initial Model (68% Accuracy):**
  - **Architecture:** Basic CNN with two convolutional layers, batch normalization, ReLU activation, max pooling, and fully connected layers with dropout.
  - **Training:** Utilized Adam optimizer with a learning rate of 0.001.
  
- **Enhanced Model (82.67% Accuracy):**
  - **Data Augmentation:** Applied transformations including random cropping, horizontal flipping, rotation, and color jittering to increase data variability.
  - **Advanced Architecture:** Added more convolutional layers, dropout layers, and batch normalization to improve feature extraction and prevent overfitting.
  - **Optimization Techniques:** Switched to AdamW optimizer and implemented OneCycleLR scheduler for dynamic learning rate adjustments.
  - **Training:** Extended training to 50 epochs, resulting in significant accuracy improvement on the test set.
  
- **Visualization:**
  - **Loss Plot:** Showed a consistent decrease, indicating effective learning.
  - **Accuracy Plot:** Demonstrated steady improvement in both training and testing accuracies with minimal overfitting.

## Technologies Used

- **Programming Languages:**
  - Python

- **Libraries and Frameworks:**
  - PyTorch
  - torchvision
  - NumPy
  - Matplotlib

- **Tools:**
  - Jupyter Notebook

## Conclusion

The **DeepLearning-CIFAR10** project successfully demonstrates the development and optimization of a CNN for image classification. Starting with a foundational model, iterative enhancements through advanced data augmentation, architectural modifications, and optimized training strategies culminated in a robust model achieving 82.67% accuracy on the CIFAR-10 test set. This project highlights the critical importance of data preprocessing, model complexity, regularization techniques, and dynamic learning rate adjustments in enhancing neural network performance.

Future work could explore deeper architectures, transfer learning, and hyperparameter tuning to further enhance model accuracy and efficiency.


