"""
Feature Extractor for Painting Similarity
==========================================
Uses a pretrained ResNet-50 backbone (same architecture as Task 1) to 
extract dense embeddings from the National Gallery of Art dataset.
Embeddings are saved to disk for fast similarity search.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


class PaintingDataset(Dataset):
    """Dataset for loading NGA paintings with metadata."""
    
    def __init__(self, image_dir, metadata_path=None, transform=None):
        """
        Args:
            image_dir: Directory containing painting images
            metadata_path: Optional path to objects.csv or metadata JSON
            transform: Image transforms
        """
        self.image_dir = image_dir
        self.transform = transform or self.default_transform()
        
        # Collect all image files
        self.image_paths = []
        self.metadata = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        if os.path.isdir(image_dir):
            for root, dirs, files in os.walk(image_dir):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in valid_extensions:
                        full_path = os.path.join(root, f)
                        self.image_paths.append(full_path)
                        self.metadata.append({
                            'filename': f,
                            'path': full_path,
                        })
        
        # If metadata CSV/JSON is provided, enrich the metadata
        if metadata_path and os.path.exists(metadata_path):
            self._load_metadata(metadata_path)
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def _load_metadata(self, metadata_path):
        """Load metadata from NGA's objects.csv or a JSON file."""
        ext = os.path.splitext(metadata_path)[1].lower()
        
        if ext == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(metadata_path, low_memory=False)
                # Build a lookup from filename to metadata
                # NGA's objects.csv has columns like: objectid, title, attribution, etc.
                filename_to_meta = {}
                if 'iiifthumburl' in df.columns or 'imageid' in df.columns:
                    for _, row in df.iterrows():
                        obj = row.to_dict()
                        # Try to match by image ID or filename
                        key = str(obj.get('objectid', ''))
                        filename_to_meta[key] = obj
                
                # Enrich existing metadata
                for item in self.metadata:
                    fname = os.path.splitext(item['filename'])[0]
                    if fname in filename_to_meta:
                        item.update(filename_to_meta[fname])
                        
            except Exception as e:
                print(f"Warning: Could not load CSV metadata: {e}")
                
        elif ext == '.json':
            try:
                with open(metadata_path, 'r') as f:
                    meta_list = json.load(f)
                if isinstance(meta_list, dict):
                    meta_list = list(meta_list.values())
                # Try to match by index or filename
                for i, item in enumerate(self.metadata):
                    if i < len(meta_list):
                        item.update(meta_list[i])
            except Exception as e:
                print(f"Warning: Could not load JSON metadata: {e}")
    
    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        return image, idx  # Return index for metadata lookup


class PaintingFeatureExtractor(nn.Module):
    """
    Feature extractor using ResNet-50 backbone.
    Outputs a 2048-dim embedding for each painting.
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        # Remove the final classification layer — use the 2048-dim feature vector
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Up to avg pool
        self.embedding_dim = 2048
    
    def forward(self, x):
        features = self.backbone(x)         # (B, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
        # L2 normalize for cosine similarity
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features


def extract_embeddings(image_dir, output_path, metadata_path=None,
                       batch_size=32, num_workers=2, device=None):
    """
    Extract embeddings for all images in a directory and save to disk.
    
    Args:
        image_dir: Path to directory of painting images
        output_path: Path to save the output .npz file
        metadata_path: Optional path to metadata CSV/JSON
        batch_size: Batch size for inference
        num_workers: DataLoader workers
        device: torch device
    
    Returns:
        dict with 'embeddings', 'paths', 'metadata'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset & DataLoader
    dataset = PaintingDataset(image_dir, metadata_path=metadata_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # Model
    model = PaintingFeatureExtractor(pretrained=True).to(device)
    model.eval()
    
    all_embeddings = []
    all_indices = []
    
    print(f"\nExtracting embeddings from {len(dataset)} images...")
    with torch.no_grad():
        for images, indices in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_indices.extend(indices.numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    save_data = {
        'embeddings': all_embeddings,
        'image_paths': np.array([dataset.metadata[i]['path'] for i in all_indices]),
        'filenames': np.array([dataset.metadata[i]['filename'] for i in all_indices]),
    }
    np.savez_compressed(output_path, **save_data)
    
    print(f"\nSaved {len(all_embeddings)} embeddings of dim {all_embeddings.shape[1]} to {output_path}")
    print(f"File size: {os.path.getsize(output_path + '.npz' if not output_path.endswith('.npz') else output_path) / 1024 / 1024:.1f} MB")
    
    return save_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract painting embeddings')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing painting images')
    parser.add_argument('--output', type=str, default='embeddings.npz',
                        help='Output path for embeddings')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Optional metadata CSV/JSON path')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    extract_embeddings(
        image_dir=args.image_dir,
        output_path=args.output,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
