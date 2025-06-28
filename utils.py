def setup_imports():
    """
    Import and return all necessary libraries.
    Returns:
        tuple: Imported modules and functions
    """
    import matplotlib.pyplot as plt
    from sklearn import datasets, metrics, svm
    from sklearn.model_selection import train_test_split
    return plt, datasets, metrics, svm, train_test_split

def load_and_prepare_data(datasets):
    """
    Load the digits dataset and prepare it for training.
    Returns:
        tuple: (data, digits) where data is the flattened images and digits is the original dataset
    """
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits

def visualize_training_data(plt, digits):
    """
    Visualize the first 4 images from the training data.
    Args:
        plt: matplotlib.pyplot module
        digits: The digits dataset containing images and targets
    """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    
    # Save the figure
    plt.savefig('figures/training_samples.png', bbox_inches='tight', dpi=300)
    return fig

def train_model(svm, train_test_split, data, digits):
    """
    Split the data and train the SVM classifier.
    Args:
        svm: sklearn.svm module
        train_test_split: sklearn.model_selection.train_test_split function
        data: Flattened image data
        digits: The digits dataset containing targets
    Returns:
        tuple: (clf, X_train, X_test, y_train, y_test) - trained classifier and data splits
    """
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    return clf, X_train, X_test, y_train, y_test

def make_predictions(clf, X_test):
    """
    Make predictions using the trained classifier.
    Args:
        clf: Trained classifier
        X_test: Test data
    Returns:
        array: Predicted values
    """
    return clf.predict(X_test)

def visualize_predictions(plt, X_test, predicted):
    """
    Visualize the first 4 test samples and their predictions.
    Args:
        plt: matplotlib.pyplot module
        X_test: Test data
        predicted: Predicted values
    """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    
    # Save the figure
    plt.savefig('figures/prediction_samples.png', bbox_inches='tight', dpi=300)
    return fig

def evaluate_model(plt, metrics, clf, y_test, predicted):
    """
    Evaluate the model and display classification report and confusion matrix.
    Args:
        plt: matplotlib.pyplot module
        metrics: sklearn.metrics module
        clf: Trained classifier
        y_test: True labels
        predicted: Predicted labels
    """
    # Print classification report
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    # Display and print confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    
    # Save the confusion matrix figure
    plt.savefig('figures/confusion_matrix.png', bbox_inches='tight', dpi=300)
    
    return disp.confusion_matrix

def rebuild_classification_report(metrics, confusion_matrix):
    """
    Rebuild classification report from confusion matrix.
    Args:
        metrics: sklearn.metrics module
        confusion_matrix: The confusion matrix
    """
    # The ground truth and predicted lists
    y_true = []
    y_pred = []

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(confusion_matrix)):
        for pred in range(len(confusion_matrix)):
            y_true += [gt] * confusion_matrix[gt][pred]
            y_pred += [pred] * confusion_matrix[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    ) 