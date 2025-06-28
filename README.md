# Handwritten Digits Classification

This project demonstrates the use of scikit-learn for recognizing handwritten digits (0-9) using Support Vector Machine (SVM) classification.

## Project Structure

```
refactor_code/
├── figures/            # Directory containing output visualizations
│   ├── training_samples.png     # Training data visualization
│   ├── prediction_samples.png   # Prediction results visualization
│   └── confusion_matrix.png     # Confusion matrix plot
├── utils.py            # Utility functions for data processing and visualization
├── plot_digits_classification.py  # Main script
└── README.md           # Project documentation
```

## Features

- Loads and processes the digits dataset from scikit-learn
- Visualizes training data and predictions
- Trains an SVM classifier
- Evaluates model performance
- Generates and saves classification reports and visualizations
- Stores all visualizations in a dedicated figures directory

## Functions Overview

### In `utils.py`:

1. `setup_imports()`
   - Purpose: Imports and returns necessary libraries
   - Returns: `plt, datasets, metrics, svm, train_test_split`

2. `load_and_prepare_data(datasets)`
   - Purpose: Loads the digits dataset and prepares it for training
   - Args: `datasets` module from scikit-learn
   - Returns: `(data, digits)` - flattened images and original dataset

3. `visualize_training_data(plt, digits)`
   - Purpose: Visualizes and saves first 4 images from training data
   - Args: 
     - `plt`: matplotlib.pyplot module
     - `digits`: digits dataset
   - Saves: `figures/training_samples.png`
   - Returns: Figure object

4. `train_model(svm, train_test_split, data, digits)`
   - Purpose: Creates and trains the SVM classifier
   - Args:
     - `svm`: sklearn.svm module
     - `train_test_split`: sklearn function
     - `data`: flattened image data
     - `digits`: digits dataset
   - Returns: `(clf, X_train, X_test, y_train, y_test)`

5. `make_predictions(clf, X_test)`
   - Purpose: Makes predictions using trained classifier
   - Args:
     - `clf`: trained classifier
     - `X_test`: test data
   - Returns: predicted values

6. `visualize_predictions(plt, X_test, predicted)`
   - Purpose: Visualizes and saves first 4 test samples and their predictions
   - Args:
     - `plt`: matplotlib.pyplot module
     - `X_test`: test data
     - `predicted`: predicted values
   - Saves: `figures/prediction_samples.png`
   - Returns: Figure object

7. `evaluate_model(plt, metrics, clf, y_test, predicted)`
   - Purpose: Evaluates model, displays and saves metrics
   - Args:
     - `plt`: matplotlib.pyplot module
     - `metrics`: sklearn.metrics module
     - `clf`: trained classifier
     - `y_test`: true labels
     - `predicted`: predicted labels
   - Saves: `figures/confusion_matrix.png`
   - Returns: confusion matrix

8. `rebuild_classification_report(metrics, confusion_matrix)`
   - Purpose: Rebuilds classification report from confusion matrix
   - Args:
     - `metrics`: sklearn.metrics module
     - `confusion_matrix`: model's confusion matrix

### In `plot_digits_classification.py`:

1. `main()`
   - Purpose: Orchestrates the entire classification process
   - Workflow:
     1. Sets up imports
     2. Loads and prepares data
     3. Visualizes and saves training data
     4. Trains model
     5. Makes predictions
     6. Visualizes and saves predictions
     7. Evaluates model and saves results
     8. Shows results

## Dependencies

- scikit-learn
- matplotlib
- numpy (included with scikit-learn)

## How to Run

1. Make sure you have all dependencies installed:
```bash
pip install scikit-learn matplotlib
```

2. Create the figures directory:
```bash
mkdir figures
```

3. Run the main script:
```bash
python plot_digits_classification.py
```

## Output

The script will:
1. Display and save visualizations of training data (`figures/training_samples.png`)
2. Show and save predictions on test data (`figures/prediction_samples.png`)
3. Print classification report
4. Display and save confusion matrix (`figures/confusion_matrix.png`)
5. Show rebuilt classification report

## Results Visualization

### Training Samples
The first four training samples are saved in `figures/training_samples.png`. Each image shows an 8x8 grayscale digit with its true label.

### Prediction Results
Four test samples and their predictions are saved in `figures/prediction_samples.png`. Each image shows the model's prediction for that digit.

### Confusion Matrix
The confusion matrix visualization is saved in `figures/confusion_matrix.png`. This shows the model's prediction accuracy across all digit classes.

## Model Details

- Classifier: Support Vector Machine (SVM)
- Parameters:
  - gamma: 0.001
- Dataset split: 50% training, 50% testing
- Input: 8x8 pixel images (flattened to 64 features)
- Output: Digit classification (0-9)

## Performance Metrics

The model evaluation includes:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization (saved as PNG)
- Rebuilt classification report from confusion matrix

## License

BSD-3-Clause (as per original scikit-learn example)