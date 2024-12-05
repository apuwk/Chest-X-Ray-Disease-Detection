from pathlib import Path
import numpy as np
from PIL import Image
import logging
from sklearn.model_selection import train_test_split

class XRayPreprocessor:
    def __init__(self, 
                 target_size=(224, 224),
                 normalize_method='standard',
                 train_split=0.7,
                 val_split=0.15,
                 test_split=0.15,
                 random_seed=42):
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_image(self, folder_path, image_path):
        """
        Load and validate image
        """
        try:
            image_path = Path(folder_path) / image_path
            
            if not image_path.exists():
                self.logger.error(f"Image not found: {image_path}")
                return None
            
            image = Image.open(image_path)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def preprocess_image(self, image):
        """
        Preprocess single image
        """
        
        # You might want to add a check if image is None
        if image is None:
            return None
        try:
            
            if image.size != (224, 224):    
                image = image.resize(self.target_size)
            
            img_array = np.array(image)
            
            if self.normalize_method == 'standard':
                img_array = img_array / 255.0
            elif self.normalize_method == 'minmax':
                img_array = (img_array - 127.5) / 127.5
            
            img_array = np.transpose(img_array, (2, 0, 1))

            return img_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def create_splits(self, metadata_df):
        """
        Create train/val/test splits without requiring stratification
        """
        try:
            findings_dict = {
                'Atelectasis': 0,
                'Cardiomegaly': 1,
                'Effusion': 2,
                'Infiltration': 3,
                'Mass': 4,
                'Nodule': 5,
                'Pneumonia': 6,
                'Pneumothorax': 7,
                'Consolidation': 8,
                'Edema': 9,
                'Emphysema': 10,
                'Fibrosis': 11,
                'Pleural_Thickening': 12,
                'Hernia': 13
            }
            
            all_images = metadata_df['Image Index'].values
            all_labels = metadata_df['Finding Labels']
            
            label_matrix = np.zeros((len(all_labels), len(findings_dict)))
            
            for finding, idx in findings_dict.items():
                label_matrix[:, idx] = all_labels.str.contains(f'\\b{finding}\\b').astype(int)
                    
            binary_labels = label_matrix.astype(np.float32)
            
            # First split: train vs temp (validation + test)
            try:
                train_idx, temp_idx = train_test_split(
                    np.arange(len(all_images)),
                    train_size=self.train_split,
                    random_state=self.random_seed,
                    stratify=binary_labels  # Try with stratification first
                )
            except ValueError:
                # If stratification fails, do split without it
                self.logger.warning("Stratification failed, performing split without stratification")
                train_idx, temp_idx = train_test_split(
                    np.arange(len(all_images)),
                    train_size=self.train_split,
                    random_state=self.random_seed,
                    stratify=None  # Remove stratification
                )
            
            # Second split: validation vs test
            val_size = self.val_split / (self.val_split + self.test_split)
            try:
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    train_size=val_size,
                    random_state=self.random_seed,
                    stratify=binary_labels[temp_idx]  # Try with stratification first
                )
            except ValueError:
                # If stratification fails, do split without it
                self.logger.warning("Stratification failed for val/test split, performing split without stratification")
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    train_size=val_size,
                    random_state=self.random_seed,
                    stratify=None  # Remove stratification
                )
            
            # Create split dictionary
            image_splits = {
                'train': all_images[train_idx],
                'val': all_images[val_idx],
                'test': all_images[test_idx]
            }
            
            label_splits = {
                'train': binary_labels[train_idx],
                'val': binary_labels[val_idx],
                'test': binary_labels[test_idx]
            }
            
            # Log split sizes and class distribution
            self.logger.info(f"Train set size: {len(image_splits['train'])}")
            self.logger.info(f"Validation set size: {len(image_splits['val'])}")
            self.logger.info(f"Test set size: {len(image_splits['test'])}")
                        
            return image_splits, label_splits
                
        except Exception as e:
            self.logger.error(f"Error creating splits: {str(e)}")
            print(f"Detailed error: {e}")  # Add this line
            return None
        
   