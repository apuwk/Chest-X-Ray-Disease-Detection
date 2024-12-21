import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
from src.models.cnn import ChestXrayModel
import src.training.metrics as metrics

class XRayTrainer:
    def __init__(
        self,
        model: ChestXrayModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 14,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,  # Reduced base learning rate
        device: Optional[torch.device] = None,
        checkpoint_dir: str = 'checkpoints',
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        # Set up device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up weighted loss function if class weights provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = criterion or nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.criterion = criterion or nn.BCEWithLogitsLoss()
        
        # Set up optimizer with different learning rates for different parts
        if optimizer is None:
            # Group parameters by module type for DenseNet
            densenet_params = []
            densenet_bn_params = []
            other_params = []
            
            # DenseNet parameters (separate BN params)
            for module in list(model.densenet_global.modules()) + list(model.densenet_local.modules()):
                if isinstance(module, nn.BatchNorm2d):
                    densenet_bn_params.extend(list(module.parameters(recurse=False)))
                elif isinstance(module, (nn.Conv2d, nn.Linear)):
                    densenet_params.extend(list(module.parameters(recurse=False)))
            
            # Other model parameters
            other_modules = [
                model.fpn_global, model.fpn_local,
                model.cbam_global, model.cbam_local,
                model.fusion, model.classifier
            ]
            for module in other_modules:
                other_params.extend(list(module.parameters()))
            
            # Create optimizer with parameter groups
            self.optimizer = optim.AdamW([
                {
                    'params': densenet_params,
                    'lr': lr/20,  # Slower learning for DenseNet conv layers
                    'weight_decay': 0.01
                },
                {
                    'params': densenet_bn_params,
                    'lr': lr/20,  # Slower learning for DenseNet BN
                    'weight_decay': 0  # No weight decay for BN
                },
                {
                    'params': other_params,
                    'lr': lr,  # Full learning rate for new layers
                    'weight_decay': 0.01
                }
            ])
        else:
            self.optimizer = optimizer
        
        # Set up learning rate schedulers
        self.warmup_epochs = 5
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        self.main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        self.current_epoch = 0
        
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
        early_stopping_patience: int = 7  # Increased patience for DenseNet
    ) -> Dict[str, float]:
        best_val_loss = float('inf')
        best_metrics = {}
        patience_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update learning rate scheduler
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.main_scheduler.step(val_metrics['loss'])
            
            self.current_epoch = epoch
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val Acc: {val_metrics['accuracy']:.4f} - "
                f"Val F1: {val_metrics['f1_score']:.4f} - "
                f"LR: {current_lr:.6f}"
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
        """Trains the model for one epoch."""
        self.model.train()
        
        # Keep batch norm layers in eval mode for DenseNet
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, ((images_global, images_local), targets) in enumerate(pbar):
            images_global = images_global.float().to(self.device)
            images_local = images_local.float().to(self.device)
            targets = targets.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images_global, images_local)
            
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.cpu())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate metrics
        metrics_dict = metrics.calculate_metrics(
            predictions=all_predictions,
            targets=all_targets,
            num_classes=self.num_classes
        )
        
        # Add loss to metrics
        metrics_dict['loss'] = running_loss / len(self.train_loader)
        
        return metrics_dict
            
    def _validate_epoch(self) -> Dict[str, float]:
        """Validates the model for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, ((images_global, images_local), targets) in enumerate(pbar):
                images_global = images_global.float().to(self.device)
                images_local = images_local.float().to(self.device)
                targets = targets.float().to(self.device)
                
                outputs = self.model(images_global, images_local)
                
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
                })
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate metrics
        metrics_dict = metrics.calculate_metrics(
            predictions=all_predictions,
            targets=all_targets,
            num_classes=self.num_classes
        )
        
        # Add loss to metrics
        metrics_dict['loss'] = running_loss / len(self.val_loader)
        
        return metrics_dict

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint along with training info."""
        f1_score = metrics.get('f1_score', 0)
        accuracy = metrics.get('accuracy', 0)
        
        checkpoint_name = (
            f"densenet_checkpoint_epoch{epoch}_"
            f"f1_{f1_score:.3f}_"
            f"acc_{accuracy:.3f}.pt"
        )
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict(),
            'main_scheduler_state_dict': self.main_scheduler.state_dict(),
            'metrics': metrics,
            'num_classes': self.num_classes,
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_name}")
            
            # Cleanup old checkpoints
            #self._cleanup_old_checkpoints()
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def _cleanup_old_checkpoints(self, keep_n: int = 3) -> None:
        """Remove old checkpoints, keeping only the n most recent ones."""
        checkpoints = list(self.checkpoint_dir.glob("densenet_checkpoint_*.pt"))
        if len(checkpoints) > keep_n:
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for checkpoint in checkpoints[:-keep_n]:
                checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint.name}")
                
    def evaluate_test_set(self) -> Dict[str, float]:
        """
        Comprehensive evaluation on test set with detailed metrics.
        To be used only for final evaluation, not during training.
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Dictionary to store per-class metrics
        class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Test Evaluation')
            
            for batch_idx, ((images_global, images_local), targets) in enumerate(pbar):
                images_global = images_global.float().to(self.device)
                images_local = images_local.float().to(self.device)
                targets = targets.float().to(self.device)
                
                outputs = self.model(images_global, images_local)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Get overall metrics
        metrics_dict = metrics.calculate_metrics(
            predictions=all_predictions,
            targets=all_targets,
            num_classes=self.num_classes
        )
        
        # Get per-class metrics
        per_class_metrics = metrics.get_per_class_metrics(
            predictions=all_predictions,
            targets=all_targets,
            class_names=class_names
        )
        
        # Print detailed evaluation
        print("\n=== Test Set Evaluation ===")
        print(f"Overall Metrics:")
        print(f"Average Loss: {running_loss / len(self.val_loader):.4f}")
        print(f"Overall Accuracy: {metrics_dict['accuracy']:.4f}")
        print(f"Overall F1 Score: {metrics_dict['f1_score']:.4f}")
        print(f"Overall Precision: {metrics_dict['precision']:.4f}")
        print(f"Overall Recall: {metrics_dict['recall']:.4f}")
        
        print("\nPer-Class Metrics:")
        for class_name, class_metrics in per_class_metrics.items():
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1 Score: {class_metrics['f1_score']:.4f}")
            print(f"  Accuracy: {class_metrics['accuracy']:.4f}")
        
        # Calculate confusion matrix statistics per class
        predictions_binary = (all_predictions >= 0.5).astype(int)
        for i, class_name in enumerate(class_names):
            true_pos = np.sum((predictions_binary[:, i] == 1) & (all_targets[:, i] == 1))
            true_neg = np.sum((predictions_binary[:, i] == 0) & (all_targets[:, i] == 0))
            false_pos = np.sum((predictions_binary[:, i] == 1) & (all_targets[:, i] == 0))
            false_neg = np.sum((predictions_binary[:, i] == 0) & (all_targets[:, i] == 1))
            
            print(f"\nDetailed Statistics for {class_name}:")
            print(f"  True Positives: {true_pos}")
            print(f"  True Negatives: {true_neg}")
            print(f"  False Positives: {false_pos}")
            print(f"  False Negatives: {false_neg}")
        
        return metrics_dict, per_class_metrics