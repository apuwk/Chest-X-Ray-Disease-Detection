from torch.utils.data import Dataset
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, folder_path, image_paths, labels, preprocessor, transform=None, cache_size=100):
        """
        Args:
            folder_path: Path to image directory
            image_paths: List of image file names
            labels: Corresponding labels
            preprocessor: XRayPreprocessor instance
            transform: Additional transforms (optional)
            cache_size: Number of images to cache in memory
        """
        self.folder_path = folder_path
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Check cache first
        if image_path in self.cache:
            global_view, local_view = self.cache[image_path]
        else:
            # Load and process image
            image = self.preprocessor.load_image(self.folder_path, image_path)
            if image is None:
                raise Exception(f"Failed to load image: {image_path}")
            
            # Get both views
            global_view, local_view = self.preprocessor.create_multiscale_input(image)
            if global_view is None or local_view is None:
                raise Exception(f"Failed to create views for: {image_path}")
            
            # Update cache
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[image_path] = (global_view, local_view)
        
        # Convert to tensors
        global_view = torch.tensor(global_view, dtype=torch.float32)
        local_view = torch.tensor(local_view, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            global_view = self.transform(global_view)
            local_view = self.transform(local_view)
            
        return (global_view, local_view), label