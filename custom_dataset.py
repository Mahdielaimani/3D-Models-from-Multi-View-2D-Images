import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import logging
import re

class MultiViewAirplaneDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")
        
        logging.info(f"Scanning directory: {root_dir}")
        
        # Look for airplane model directories
        for model_dir in os.listdir(root_dir):
            model_path = os.path.join(root_dir, model_dir)
            if os.path.isdir(model_path):
                screenshots_dir = os.path.join(model_path, 'screenshots')
                if os.path.isdir(screenshots_dir):
                    # Get all PNG files
                    image_files = [f for f in os.listdir(screenshots_dir) 
                                 if f.lower().endswith('.png')]
                    
                    # Sort files by the numeric suffix
                    try:
                        image_files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
                        
                        if len(image_files) == 14:  # We expect 14 views per airplane
                            image_paths = [os.path.join(screenshots_dir, f) for f in image_files]
                            self.samples.append(image_paths)
                            logging.info(f"Found airplane model with 14 views: {model_dir}")
                    except Exception as e:
                        logging.warning(f"Skipping directory due to invalid file naming: {model_dir}")
                        continue
        
        logging.info(f"Found {len(self.samples)} valid airplane models")
        
        if len(self.samples) == 0:
            raise ValueError("No valid airplane models found (each should have 14 views)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_paths = self.samples[idx]
        images = []
        
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                logging.error(f"Error loading image {img_path}: {str(e)}")
                raise
        
        return torch.stack(images)  # Shape: (14, 3, 224, 224)

