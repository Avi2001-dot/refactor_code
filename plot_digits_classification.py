"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from utils import (
    setup_imports,
    load_and_prepare_data,
    visualize_training_data,
    train_model,
    make_predictions,
    visualize_predictions,
    evaluate_model,
    rebuild_classification_report
)

def main():
    """
    Main function to orchestrate the digit classification process.
    """
    # Setup imports
    plt, datasets, metrics, svm, train_test_split = setup_imports()
    
    # Load and prepare data
    data, digits = load_and_prepare_data(datasets)
    
    # Visualize training data
    visualize_training_data(plt, digits)
    
    # Train model
    clf, X_train, X_test, y_train, y_test = train_model(svm, train_test_split, data, digits)
    
    # Make predictions
    predicted = make_predictions(clf, X_test)
    
    # Visualize predictions
    visualize_predictions(plt, X_test, predicted)
    
    # Evaluate model
    confusion_matrix = evaluate_model(plt, metrics, clf, y_test, predicted)
    
    # Rebuild classification report from confusion matrix
    rebuild_classification_report(metrics, confusion_matrix)
    
    # Show all plots
    plt.show()

# Only function calls outside functions
main()
