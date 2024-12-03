# Import necessary libraries
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import logging

# Set up logging
logger = logging.getLogger(__name__)

def calculate_binary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate metrics for binary classification.
    
    Args:
        predictions: Raw prediction probabilities
        targets: Ground truth labels
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Dictionary containing metrics (accuracy, precision, recall, f1, auc_roc)
    """
    # Convert predictions to binary
    binary_preds = (predictions >= threshold).astype(np.int32)
    
    try:
        # Calculate AUC-ROC
        auc_roc = roc_auc_score(targets, predictions)
    except ValueError:
        auc_roc = 0.0
        logger.warning("Could not calculate AUC-ROC - check class distribution")
    
    # Calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        binary_preds,
        average='binary',
        zero_division=0
    )
    
    # Calculate accuracy
    accuracy = np.mean(binary_preds == targets)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc)
    }

def calculate_multilabel_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculate metrics for multi-label classification.
    
    Args:
        predictions: Raw prediction probabilities
        targets: Ground truth labels
        threshold: Classification threshold (default: 0.5)
        average: Averaging method for metrics ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary containing metrics (accuracy, precision, recall, f1)
    """
    # Convert predictions to binary
    binary_preds = (predictions >= threshold).astype(np.int32)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        binary_preds,
        average=average,
        zero_division=0
    )
    
    # Calculate accuracy
    accuracy = np.mean(binary_preds == targets)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 1,
    threshold: float = 0.5,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculate metrics based on number of classes.
    
    Args:
        predictions: Raw prediction probabilities
        targets: Ground truth labels
        num_classes: Number of classes (default: 1 for binary)
        threshold: Classification threshold (default: 0.5)
        average: Averaging method for multi-label ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary containing appropriate metrics
    """
    if num_classes == 1:
        return calculate_binary_metrics(predictions, targets, threshold)
    else:
        return calculate_multilabel_metrics(predictions, targets, threshold, average)

def get_metric_description() -> Dict[str, str]:
    """
    Get descriptions of each metric.
    
    Returns:
        Dictionary containing metric descriptions
    """
    return {
        'accuracy': 'Proportion of correct predictions (both positive and negative)',
        'precision': 'Proportion of positive identifications that were actually correct',
        'recall': 'Proportion of actual positives that were identified correctly',
        'f1_score': 'Harmonic mean of precision and recall',
        'auc_roc': 'Area under the ROC curve, measuring ability to distinguish between classes'
    }