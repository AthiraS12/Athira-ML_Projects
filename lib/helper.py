import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(conf_matrix):
    """
    Plot a confusion matrix using Seaborn's heatmap.

    Args:
    conf_matrix (numpy.ndarray): Confusion matrix to plot.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
