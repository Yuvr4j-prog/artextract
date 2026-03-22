import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd

class WikiArtMultiLabelDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mock_images=False):
        """
        Args:
            root_dir (str): Path to the ArtGAN root folder containing the dataset and csvs.
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample.
            mock_images (bool): If True, yields random noise instead of loading images. 
                                Useful for testing the pipeline without downloading the 25GB dataset.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mock_images = mock_images
        
        # Paths to the metadata folders
        self.metadata_dir = os.path.join(root_dir, 'WikiArt Dataset')
        
        # We will load style, artist, and genre
        self.data = self._build_unified_dataset()
        
    def _load_csv(self, category):
        # Because some files (like Artist/artist_train) don't have a .csv extension
        category_dir = os.path.join(self.metadata_dir, category)
        prefix = f"{category.lower()}_{self.split}"
        
        csv_path = None
        for f in os.listdir(category_dir):
            if f.startswith(prefix):
                csv_path = os.path.join(category_dir, f)
                break
                
        if not csv_path:
            raise FileNotFoundError(f"Could not find label file starting with {prefix} in {category_dir}")

        # Format varies. Style/Genre are typically path,class_index. Artist is path,,class_index
        # We can use regex separator to handle multiple commas
        df = pd.read_csv(csv_path, header=None, sep=',+', engine='python', usecols=[0, 1], names=['image_path', 'label'])
        
        # Clean up image paths if needed
        df['image_path'] = df['image_path'].str.strip()
        
        # Ensure labels are integers (some CSVs may parse them as strings)
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        return dict(zip(df['image_path'], df['label']))

    def _build_unified_dataset(self):
        style_dict = self._load_csv('Style')
        artist_dict = self._load_csv('Artist')
        genre_dict = self._load_csv('Genre')
        
        # Find intersection of all three to get images that have all labels
        common_images = set(style_dict.keys()) & set(artist_dict.keys()) & set(genre_dict.keys())
        
        dataset = []
        for img_path in sorted(common_images):
            dataset.append({
                'image_path': img_path,
                'style': style_dict[img_path],
                'artist': artist_dict[img_path],
                'genre': genre_dict[img_path]
            })
            
        print(f"Loaded {len(dataset)} {self.split} images with all 3 labels.")
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']
        full_path = os.path.join(self.root_dir, 'wikiart', img_path) # Assuming unzipped images are in root/wikiart
        
        if self.mock_images:
            # Return random noise image for testing the pipeline
            image = torch.randn(3, 224, 224)
        else:
            try:
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                # Fallback to noise if an image is corrupted or missing
                image = torch.randn(3, 224, 224)

        labels = {
            'style': torch.tensor(item['style'], dtype=torch.long),
            'artist': torch.tensor(item['artist'], dtype=torch.long),
            'genre': torch.tensor(item['genre'], dtype=torch.long)
        }
        
        return image, labels

def get_dataloaders(root_dir, batch_size=32, num_workers=4, mock_images=False):
    """
    Creates and returns the train and validation DataLoaders.
    """
    # Enhanced ResNet-style transforms with stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = WikiArtMultiLabelDataset(root_dir, split='train', transform=train_transform, mock_images=mock_images)
    val_dataset = WikiArtMultiLabelDataset(root_dir, split='val', transform=val_transform, mock_images=mock_images)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Test the dataloader (in mock mode)
    root = '.'
    print("Testing DataLoader with mock images...")
    train_loader, val_loader = get_dataloaders(root, batch_size=4, num_workers=0, mock_images=True)
    
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Style labels: {labels['style']}")
        print(f"Artist labels: {labels['artist']}")
        print(f"Genre labels: {labels['genre']}")
        break
