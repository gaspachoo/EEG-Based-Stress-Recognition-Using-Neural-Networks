# EEG-Based Stress Level Classification Using CNNs

This project explores the application of Convolutional Neural Networks (CNNs) for classifying stress levels based on EEG data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Project Overview
The objective of this project is to develop a CNN model capable of accurately classifying stress levels from EEG signals. The approach involves data preprocessing, model training, and evaluation.

## Dataset
The dataset comprises EEG recordings labeled with corresponding stress levels. Detailed information about the dataset is available in the `Data/` directory.

## Project Structure
- **Data/**: Contains EEG datasets used for training and evaluation.
- **support_func/**: Includes auxiliary Python scripts for data processing and model support.
- **Assignment n1.pptx**: Presentation detailing the assignment.
- **Assignment_ EEG-Based Emotion Recognition Using Neural Networks.pdf**: Reference paper on EEG-based emotion recognition.
- **date_analysis.ipynb**: Jupyter Notebook analyzing the dataset.
- **main.py**: Main script for executing the CNN model.
- **neural_network.ipynb**: Jupyter Notebook detailing the neural network implementation.
- **subject head with electrodes.png**: Diagram showing electrode placement on the subject's head.

## Usage
1. **Data Preprocessing**:
   - Utilize `date_analysis.ipynb` to preprocess and visualize the EEG data.

2. **Model Training**:
   - Run `neural_network.ipynb` to train the CNN model on the preprocessed data.

3. **Evaluation**:
   - Use `main.py` to evaluate the trained model's performance on test data.

## Results
The CNN model achieved an accuracy of **[insert accuracy]%** on the test dataset, demonstrating its efficacy in classifying stress levels from EEG signals.

## References
- [Assignment: EEG-Based Emotion Recognition Using Neural Networks](Assignment_%20EEG-Based%20Emotion%20Recognition%20Using%20Neural%20Networks.pdf)
- [Presentation: Assignment n1](Assignment%20n1.pptx)
