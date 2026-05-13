"""Sequence-labeling metrics for radar-source recall."""

import numpy as np


def confusion_matrix(
    y_true,
    y_pred,
    num_classes,
):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.reshape(-1), y_pred.reshape(-1)):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def class_counts(labels, num_classes):
    counts = np.bincount(labels.reshape(-1).astype(np.int64), minlength=num_classes)
    return counts.astype(np.int64)


def recall_per_class(matrix):
    true_counts = matrix.sum(axis=1)
    true_positive = np.diag(matrix)
    return np.divide(
        true_positive,
        true_counts,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=true_counts != 0,
    )


def average_recall(recalls):
    return float(np.mean(recalls))
