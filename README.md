
---

# Plant Disease Prediction

Deep learning based plant disease classification system built using TensorFlow and CNNs. The model is trained on the PlantVillage dataset to classify multiple crop diseases from leaf images.

---

## Project Overview

This project builds an end-to-end image classification pipeline that:

* Loads plant leaf images from directory
* Applies real-time data augmentation
* Trains a Convolutional Neural Network
* Evaluates performance using classification metrics
* Generates confusion matrix
* Supports single image prediction
* Saves trained model for deployment

The system predicts 15 different plant disease classes.

---

## Dataset

Dataset used: PlantVillage

* 20,000+ labeled leaf images
* Multiple crop types including Tomato, Potato, Pepper
* Healthy and diseased classes

Images are streamed using ImageDataGenerator to prevent RAM overflow in Google Colab.

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

## Model Architecture

* Conv2D layers
* MaxPooling layers
* Dense layers
* Dropout for regularization
* Softmax output layer

Input size: 224 x 224 x 3
Loss: Categorical Crossentropy
Optimizer: Adam

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## How To Run

1. Clone repository
2. Open notebook in Google Colab
3. Download PlantVillage dataset
4. Run all cells sequentially
5. Model will train and save as:

plant_disease_model.h5

---

## Single Image Prediction

Upload any leaf image and run prediction cell to get disease class output.

---

## Results

Model achieves strong validation accuracy using CNN with data augmentation.

Exact accuracy depends on training configuration and hardware.

---

## Future Improvements

* Transfer learning with MobileNetV2
* Model quantization for edge devices
* Web deployment using Flask or FastAPI
* Mobile app integration

---

## Author

Muhammed Sinan
B.Tech CSE (Data Science)

---

Next step.

You already ran `git init`.

Run this:

git add .
git commit -m "Initial commit - Plant Disease Prediction"
git branch -M main
git remote add origin [https://github.com/YOUR_USERNAME/Plant_Disease_Prediction.git](https://github.com/YOUR_USERNAME/Plant_Disease_Prediction.git)
git push -u origin main

