# Accent Recognition Using Deep Learning

This repository contains the implementation of my master's thesis on accent recognition using deep learning models. The
work compares the performance of two deep learning architectures: Audio Spectrogram Transformer (AST) and Convolutional
Neural Networks (CNN).

## Table of Contents

1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Data Preparation](#Data-Preparation)
    - Filtering and Splitting Data
4. [Models](#Models)
5. [Training](#Training)
    - AST
    - CNN
6. [Evaluation](#Evaluation)
7. [Results](#Results)

## Introduction

The goal of this project is to evaluate the performance of two deep learning models, AST and CNN, in recognizing English
accents. The study aims to determine which model is more effective in accent recognition.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Przemo258/accent-recognition.git
    cd accent-recognition
    ```
2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

Dataset used in this project is the Mozilla Common Voice dataset. You can download it from
here: [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets).

1. Download the dataset and extract it to the `data` directory.

2. Filtering and Splitting Data:
    - Use `data_prep.py` to filter the dataset based on upvotes and to split it into training, validation, and test
      sets.
   ```bash
   python src/DataPrep/data_prep.py
   ```

3. `data_prep.py` script will create the following files:
    - `data/metadata_train.csv`
    - `data/metadata_val.csv`
    - `data/metadata_test.csv`

## Models

### Audio Spectrogram Transformer (AST)

The [AST](https://github.com/YuanGongND/ast) model is implemented in AST.py and MozillaCommonVoiceDatasetAST.py. It is
loaded via transformers library and processes mel-spectrograms of audio signals using transformer layers.

### Convolutional Neural Networks (CNN)

The [CNN](https://music-classification.github.io/tutorial/part3_supervised/tutorial.html) model is implemented in
CNN.py, ClassificationNet.py, and MozillaCommonVoiceDatasetCNN.py. It processes
mel-spectrograms of audio signals using convolutional layers.

## Training

To train the models, use the following commands:

1. AST:
    ```bash
    python src/AST/AST.py
    ```

2. CNN:
    ```bash
    python src/CNN/CNN.py
    ```

## Evaluation

The evaluation of the models includes calculating accuracy, F1 score, precision, recall, and generating confusion
matrices. The evaluation scripts are included in the respective model files. Additionally, inference time is measured.

## Results

The results of the experiments, can be found in the results directory after training of the model finishes.

## License

This project is licensed under the MIT License
