import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from typing import Dict, List, Optional
from src.models.cnn import ChestXrayModel
import src.training.metrics as metrics

class XRayTrainer:
    def __init__(
        self,
        model: ChestXrayModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 1,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize the X-Ray model trainer.
        
        Args:
            model: The neural network model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_classes: Number of classes (default=1 for binary classification)
            criterion: Loss function (defaults to BCELoss)
            optimizer: Optimizer (defaults to Adam)
            lr: Learning rate for optimizer if not provided
            device: Device to train on (defaults to GPU if available)
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        # Set up device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up loss function
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        
        # Set up optimizer
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        
        # Set up checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Log trainer setup
        self.logger.info(f"Trainer initialized with {model.__class__.__name__}")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Number of classes: {num_classes}")
        
    def train(
        self, 
        num_epochs: int, 
        early_stopping_patience: int = 5
    ) -> Dict[str, float]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dict containing best metrics
        """
        best_val_loss = float('inf')
        best_metrics = {}
        patience_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val AUC-ROC: {val_metrics.get('auc_roc', 0):.4f}"
            )
            
            # Check if model improved
            if val_metrics['loss'] < best_val_loss:
                self.logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_metrics['loss']:.4f}")
                best_val_loss = val_metrics['loss']
                best_metrics = val_metrics
                self._save_checkpoint(epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(f"No improvement in validation loss for {patience_counter} epochs")
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        self.logger.info("Training completed!")
        return best_metrics
        
        
    def _train_epoch(self) -> Dict[str, float]:
        """
        Trains the model for one epoch.
        
        Returns:
            Dict containing training metrics for this epoch
        """
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (images, targets) in enumerate(pbar):

            images = images.float().to(self.device)
            targets = targets.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            self.optimizer.step()
            
            running_loss += loss.item()
            
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
        
        metrics = self._calculate_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets)
        )
        
        metrics['loss'] = running_loss / len(self.train_loader)
        
        return metrics
            
    
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validates the model for one epoch.
        
        Returns:
            Dict containing validation metrics for this epoch
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, (images, targets) in enumerate(pbar):

                images = images.float().to(self.device)
                targets = targets.float().to(self.device)
                
                outputs = self.model(images)
                
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
                })
        
        metrics = self._calculate_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets)
        )
        
        metrics['loss'] = running_loss / len(self.val_loader)
        
        return metrics
        
        
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
       return metrics.calculate_metrics(predictions, targets, self.num_classes)


    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Save model checkpoint along with training info.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary containing current metrics
        """
        # Create checkpoint filename with metrics
        f1_score = metrics.get('f1_score', 0)
        auc_roc = metrics.get('auc_roc', 0)
        
        checkpoint_name = (
            f"checkpoint_epoch{epoch}_"
            f"f1_{f1_score:.3f}_"
            f"auc_{auc_roc:.3f}.pt"
        )
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'num_classes': self.num_classes,
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_name}")
            
            # Optionally, remove old checkpoints to save space
            self._cleanup_old_checkpoints()
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def _cleanup_old_checkpoints(self, keep_n: int = 3) -> None:
        """
        Remove old checkpoints, keeping only the n most recent ones.
        
        Args:
            keep_n: Number of checkpoints to keep
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > keep_n:
            # Sort by creation time
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_n]:
                checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint.name}")