from pathlib import Path
import numpy as np
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
import cv2

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
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_image(self, folder_path, image_path):
        """Load and validate image"""
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

    def preprocess_image(self, image, is_local_view=False):
        """Enhanced preprocessing pipeline for X-ray images"""
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Apply CLAHE with different parameters for local view
            if is_local_view:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))  # Stronger contrast for local
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_array = clahe.apply(img_array.astype(np.uint8))

            _, thresh = cv2.threshold(img_array, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Different margins for global and local views
                margin = 30 if is_local_view else 50
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img_array.shape[1] - x, w + 2*margin)
                h = min(img_array.shape[0] - y, h + 2*margin)
                
                img_array = img_array[y:y+h, x:x+w]

            target_size = self.target_size[0]
            aspect_ratio = img_array.shape[1] / img_array.shape[0]
            
            if aspect_ratio > 1:
                new_width = target_size
                new_height = int(target_size / aspect_ratio)
            else:
                new_height = target_size
                new_width = int(target_size * aspect_ratio)
                
            img_array = cv2.resize(img_array, (new_width, new_height))

            delta_w = target_size - new_width
            delta_h = target_size - new_height
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            img_array = cv2.copyMakeBorder(img_array, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=0)

            # Different normalization for local view
            if is_local_view:
                p2 = np.percentile(img_array, 2)
                p98 = np.percentile(img_array, 98)
                img_array = (img_array - p2) / (p98 - p2)
            else:
                p5 = np.percentile(img_array, 5)
                p95 = np.percentile(img_array, 95)
                img_array = (img_array - p5) / (p95 - p5)
            
            img_array = np.clip(img_array, 0, 1)
            img_array = np.stack([img_array] * 3, axis=-1)
            img_array = np.transpose(img_array, (2, 0, 1))

            return img_array

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def create_multiscale_input(self, image):
        """Create global and local views of the image"""
        try:
            # Get global view
            global_view = self.preprocess_image(image, is_local_view=False)
            if global_view is None:
                return None, None

            # Create zoomed local view
            local_view = self.preprocess_image(image, is_local_view=True)
            if local_view is None:
                return global_view, global_view  # Fallback to global view

            return global_view, local_view

        except Exception as e:
            self.logger.error(f"Error creating multi-scale input: {str(e)}")
            return None, None

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