# Face Identification System    

## Project Overview
This project implements a comprehensive face identification system using multiple feature extraction techniques and machine learning classifiers. The system processes facial images, extracts distinctive features, and trains various models to correctly identify individuals.

## Dataset
The system uses the VGG2_Dataset, which contains facial images organized by individual subjects (each person has their own directory containing multiple images). The dataset provides diverse facial appearances for robust model training.

## Pipeline Architecture

### 1. Data Preprocessing
- **Face Detection**: Haar cascade classifiers are used to detect faces in images
- **Face Cropping**: Detected faces are cropped to eliminate irrelevant background information
- **Resizing**: All face images are standardized to 128x128 pixels

### 2. Feature Extraction
The system uses a multi-feature approach combining three complementary feature extraction techniques:

- **HOG (Histogram of Oriented Gradients)**
  - Captures edge orientations and structural shape information
  - Effective for representing facial contours and distinctive edges

- **LBP (Local Binary Patterns)**
  - Captures texture information and micro-patterns on face surfaces
  - Robust to lighting variations

- **CNN Features**
  - Deep features extracted from a pre-trained ResNet-18 model
  - Provides high-level semantic representations learned from large datasets

### 3. Dimensionality Reduction
- **Standard Scaling**: Features are normalized to have zero mean and unit variance
- **PCA**: Principal Component Analysis reduces dimensionality while preserving 95% of variance

### 4. Classification Models
Multiple classifiers are implemented and compared:

- **Random Forest**: Ensemble of decision trees with hyperparameter tuning
- **K-Nearest Neighbors (KNN)**: Classification based on similarity metrics
- **Support Vector Machines (SVM)**: Using RBF kernel for non-linear classification
- **XGBoost**: Gradient boosting framework optimized for performance
- **AdaBoost**: Adaptive boosting ensemble method
- **Logistic Regression**: Probabilistic classification approach
- **Artificial Neural Network (ANN)**: Multi-layer perceptron with ReLU activations

### 5. Model Evaluation
- Train-test split (80% training, 20% testing)
- Performance evaluated using accuracy metrics
- Visualization of model performance comparisons

## Getting Started

### Prerequisites
- Python 3.6+
- PyTorch
- scikit-learn
- OpenCV
- XGBoost
- NumPy, Pandas
- Matplotlib

### Running the System
1. Ensure the VGG2_Dataset is organized with one folder per person
2. Run the notebook cells sequentially to:
   - Load and preprocess the dataset
   - Extract features
   - Train and evaluate models

## Results
The best-performing models for this face identification task were:
1. SVM (75% accuracy)
2. Logistic Regression (65% accuracy)
3. ANN (62% accuracy)

## File Structure
- `prml.ipynb`: Main Jupyter notebook containing the entire pipeline
- `hog_extractor.py`: Module for HOG feature extraction
- `lbp_extractor.py`: Module for LBP feature extraction
- `cnn_extractor.py`: Module for CNN feature extraction

## Further Improvements
- Implement data augmentation to increase training set diversity
- Try more advanced deep learning architectures
- Experiment with feature fusion techniques
- Implement cross-validation for more reliable performance estimation
