from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, folder_path, image_paths, labels, preprocessor, transform=None, cache_size = 100):
        """
        Args:
            image_paths: List of paths to images
            labels: Corresponding labels
            preprocessor: XRayPreprocessor instance
            transform: Additional PyTorch transforms (optional)
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
        
        image = self.preprocessor.load_image(self.folder_path, image_path)
        
        if image is None:
            raise Exception(f"Failed to load image: {image_path}")
        
        image = self.preprocessor.preprocess_image(image)
        
        if image is None:
            raise Exception(f"Failed to preprocess image: {image_path}")

        if self.transform:
            image = self.transform(image)

        return image, label