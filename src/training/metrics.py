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
    try:
        # Ensure predictions are in correct range (apply sigmoid if needed)
        if np.any(predictions > 1.0) or np.any(predictions < 0.0):
            predictions = 1 / (1 + np.exp(-predictions))
            
        # Convert predictions to binary
        binary_preds = (predictions >= threshold).astype(np.int32)
        
        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(targets, predictions)
        except ValueError as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            auc_roc = 0.0
        
        # Calculate precision, recall, f1 with explicit handling of zero division
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            binary_preds,
            average='binary',
            zero_division=0,
            labels=[0, 1]  # Explicitly specify labels
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
    except Exception as e:
        logger.error(f"Error in binary metrics calculation: {str(e)}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0
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
        average: Averaging method for metrics ('macro', 'micro', 'weighted', 'samples')
        
    Returns:
        Dictionary containing metrics (accuracy, precision, recall, f1)
    """
    try:
        # Ensure predictions are in correct range (apply sigmoid if needed)
        if np.any(predictions > 1.0) or np.any(predictions < 0.0):
            predictions = 1 / (1 + np.exp(-predictions))
            
        # Convert predictions to binary
        binary_preds = (predictions >= threshold).astype(np.int32)
        
        # Calculate metrics using samples average for multi-label
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            binary_preds,
            average='samples',  # Use samples average for multi-label
            zero_division=0
        )
        
        # Calculate accuracy (exact match)
        accuracy = np.mean(binary_preds == targets)   
             
        # Calculate AUC-ROC for multi-label
        try:
            auc_roc = roc_auc_score(targets, predictions, average='macro', multi_class='ovr')
        except ValueError as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            auc_roc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc)
        }
    except Exception as e:
        logger.error(f"Error in multilabel metrics calculation: {str(e)}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0
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
        average: Averaging method for multi-label ('macro', 'micro', 'weighted', 'samples')
        
    Returns:
        Dictionary containing appropriate metrics
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions).squeeze()
    targets = np.array(targets).squeeze()
    
    # Log shapes for debugging
    logger.debug(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
    
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