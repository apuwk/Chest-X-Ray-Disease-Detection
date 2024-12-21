import numpy as np
from typing import Dict, Optional, Union, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import logging
import torch

# Set up logging
logger = logging.getLogger(__name__)

def _ensure_numpy(
    data: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

def _apply_sigmoid(
    predictions: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply sigmoid if needed and convert to binary predictions.
    
    Args:
        predictions: Raw predictions
        threshold: Classification threshold
    
    Returns:
        Tuple of (probability predictions, binary predictions)
    """
    # Check if sigmoid is needed (if values are not between 0 and 1)
    if np.any(predictions > 1.0) or np.any(predictions < 0.0):
        prob_preds = 1 / (1 + np.exp(-predictions))
    else:
        prob_preds = predictions
        
    # Get binary predictions
    binary_preds = (prob_preds >= threshold).astype(np.int32)
    
    return prob_preds, binary_preds

def _validate_inputs(
    predictions: np.ndarray,
    targets: np.ndarray
) -> None:
    """Validate input shapes and values."""
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
    
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values")
    
    if not np.isfinite(targets).all():
        raise ValueError("Targets contain NaN or infinite values")

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
        Dictionary containing metrics
    """
    try:
        # Input validation
        _validate_inputs(predictions, targets)
        
        # Ensure correct shapes
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        
        # Get probabilities and binary predictions
        prob_preds, binary_preds = _apply_sigmoid(predictions, threshold)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, binary_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            binary_preds,
            average='binary',
            zero_division=0
        )
        
        # Calculate AUC-ROC for binary case only
        try:
            auc_roc = roc_auc_score(targets, prob_preds)
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
        predictions: Raw prediction probabilities (batch_size, num_classes)
        targets: Ground truth labels (batch_size, num_classes)
        threshold: Classification threshold
        average: Averaging method for metrics
        
    Returns:
        Dictionary containing metrics
    """
    try:
        # Input validation
        _validate_inputs(predictions, targets)
        
        # Get probabilities and binary predictions
        prob_preds, binary_preds = _apply_sigmoid(predictions, threshold)
        
        # Calculate sample-wise accuracy
        accuracy = accuracy_score(targets, binary_preds)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            binary_preds,
            average=average,
            zero_division=0
        )
            
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
    except Exception as e:
        logger.error(f"Error in multilabel metrics calculation: {str(e)}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

def calculate_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_classes: int = 1,
    threshold: float = 0.5,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculate metrics based on number of classes.
    
    Args:
        predictions: Raw prediction probabilities
        targets: Ground truth labels
        num_classes: Number of classes
        threshold: Classification threshold
        average: Averaging method for multi-label
        
    Returns:
        Dictionary containing metrics
    """
    try:
        # Convert to numpy if needed
        predictions = _ensure_numpy(predictions)
        targets = _ensure_numpy(targets)
        
        # Log initial shapes
        logger.debug(f"Initial shapes - Predictions: {predictions.shape}, Targets: {targets.shape}")
        
        # Ensure correct shapes
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
            
        # Log reshaped dimensions
        logger.debug(f"Reshaped - Predictions: {predictions.shape}, Targets: {targets.shape}")
        
        if num_classes == 1:
            return calculate_binary_metrics(predictions, targets, threshold)
        else:
            return calculate_multilabel_metrics(predictions, targets, threshold, average)
            
    except Exception as e:
        logger.error(f"Error in metrics calculation: {str(e)}, shapes - predictions: {predictions.shape if predictions is not None else None}, targets: {targets.shape if targets is not None else None}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

def get_per_class_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    class_names: Optional[list] = None,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each class separately.
    
    Args:
        predictions: Raw prediction probabilities (batch_size, num_classes)
        targets: Ground truth labels (batch_size, num_classes)
        class_names: List of class names
        threshold: Classification threshold
        
    Returns:
        Dictionary containing metrics for each class
    """
    try:
        predictions = _ensure_numpy(predictions)
        targets = _ensure_numpy(targets)
        
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
            
        num_classes = predictions.shape[1]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
            
        per_class_metrics = {}
        
        for i in range(num_classes):
            metrics = calculate_binary_metrics(
                predictions[:, i],
                targets[:, i],
                threshold
            )
            # Remove AUC-ROC from per-class metrics for consistency
            metrics.pop('auc_roc', None)
            per_class_metrics[class_names[i]] = metrics
            
        return per_class_metrics
        
    except Exception as e:
        logger.error(f"Error in per-class metrics calculation: {str(e)}")
        return {}

def test_metrics_calculation():
    """Test function to verify metrics calculation"""
    # Create sample data
    predictions = np.array([
        [ 2.1, -1.2,  -3.1,  2.5,  -2.1,  -1.8,  -2.1,  -1.3,  -2.3,  -1.9,  -3.2,  -1.5,  -1.7,  -1.4],
        [-1.8,  2.3,   2.2, -1.5,   2.4,   2.1,   2.3,  -1.6,  -1.8,  -1.7,  -1.9,  -1.4,  -1.6,  -1.5],
        [ 2.3, -1.5,  -1.7,  2.2,  -1.9,  -1.6,   2.4,  -1.4,  -1.6,  -1.5,  -1.8,  -1.3,  -1.5,  -1.4],
        [-1.9,  2.4,   2.1, -1.7,   2.5,   2.2,  -1.8,   2.3,   2.1,  -1.6,  -1.7,  -1.5,  -1.4,  -1.3]
    ])
    
    targets = np.array([
        [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0.]
    ])
    
    print("\nTesting with sample data:")
    print("Predictions before sigmoid (first row):", predictions[0])
    prob_preds, binary_preds = _apply_sigmoid(predictions)
    print("Predictions after sigmoid (first row):", prob_preds[0])
    print("Binary predictions (first row):", binary_preds[0])
    print("Actual targets (first row):", targets[0])
    
    metrics = calculate_metrics(predictions, targets, num_classes=14)
    print("\nTest metrics:", metrics)

if __name__ == "__main__":
    test_metrics_calculation()