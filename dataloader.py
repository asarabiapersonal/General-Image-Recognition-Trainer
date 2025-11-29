# data_loader.py

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
import config
import os

# --- COCO Dataset Class ---
class CocoClassificationDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
            
        self.categories = {cat['id']: cat['name'] for cat in self.coco['categories']}
        sorted_cat_names = sorted(self.categories.values())
        self.class_to_idx = {name: i for i, name in enumerate(sorted_cat_names)}
        self.classes = sorted_cat_names
        
        self.img_id_to_cat = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if img_id not in self.img_id_to_cat:
                self.img_id_to_cat[img_id] = cat_id
                
        self.samples = []
        for img in self.coco['images']:
            img_id = img['id']
            file_name = img['file_name']
            if img_id in self.img_id_to_cat:
                cat_id = self.img_id_to_cat[img_id]
                cat_name = self.categories[cat_id]
                label_idx = self.class_to_idx[cat_name]
                self.samples.append((file_name, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, file_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {file_name}: {e}")
            # Return a dummy tensor or handle skip (simple error handling)
            return torch.zeros((3, 128, 128)), label

# --- Transforms ---
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(140), 
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Custom Dataset Wrapper (optional, but good practice) ---
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        # Forward classes if available in subset
        if hasattr(subset, 'classes'):
            self.classes = subset.classes
            
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

# --- Default Dataloader ---
def get_dataloaders(train_dir, test_dir, batch_size):
    if config.USE_COCO_FORMAT:
        print(f"Loading COCO Data from: {config.COCO_TRAIN_ANN}")
        trainset = CocoClassificationDataset(
            img_dir=config.COCO_TRAIN_IMG_DIR,
            ann_file=config.COCO_TRAIN_ANN,
            transform=transform_train
        )
        testset = CocoClassificationDataset(
            img_dir=config.COCO_TEST_IMG_DIR,
            ann_file=config.COCO_TEST_ANN,
            transform=transform_test
        )
        class_names = trainset.classes
    else:
        print(f"Loading Folder Data from: {train_dir}")
        trainsetOG = ImageFolder(root=train_dir)
        testsetOG = ImageFolder(root=test_dir)
        trainset = TransformedDataset(trainsetOG, transform=transform_train)
        testset = TransformedDataset(testsetOG, transform=transform_test)
        class_names = trainsetOG.classes

    workers = 0 
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    
    return trainloader, testloader, class_names

# --- NEW: Dynamic Dataloader for Custom Path ---
def get_custom_dataloader_from_path(folder_path, batch_size=32, is_train=False):
    """
    Analyzes a folder to determine if it is COCO or ImageFolder structure.
    is_train: If True, uses training transforms and shuffle=True.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Path does not exist: {folder_path}")

    # Determine Transform and Shuffle based on usage
    active_transform = transform_train if is_train else transform_test
    do_shuffle = True if is_train else False

    # 1. Check for COCO Annotation file
    coco_files = [f for f in os.listdir(folder_path) if f.endswith('.json') and 'annotation' in f]
    
    dataset = None
    
    if coco_files:
        # Assume the first json found is the annotation
        ann_file = os.path.join(folder_path, coco_files[0])
        print(f"Detected COCO format. Annotation: {ann_file}")
        try:
            dataset = CocoClassificationDataset(
                img_dir=folder_path,
                ann_file=ann_file,
                transform=active_transform
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load COCO dataset: {e}")
            
    else:
        # 2. Check for Subdirectories (ImageFolder structure)
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        if len(subdirs) > 0:
            print(f"Detected ImageFolder format. Classes: {subdirs}")
            try:
                # Use standard ImageFolder
                raw_dataset = ImageFolder(root=folder_path)
                # Wrap it to apply transforms
                dataset = TransformedDataset(raw_dataset, transform=active_transform)
            except Exception as e:
                raise RuntimeError(f"Failed to load ImageFolder dataset: {e}")
        else:
            raise RuntimeError(f"Invalid Folder Structure. The selected folder must contain either an '_annotations.coco.json' file OR subfolders representing class names.")

    # 3. Check if empty
    if len(dataset) == 0:
        raise RuntimeError("The dataset was created but contains 0 valid images.")

    # 4. Create Loader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=do_shuffle, 
        num_workers=0
    )
    
    # Extract class names to return
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    elif hasattr(dataset, 'coco'):
        class_names = dataset.classes
    else:
        # Fallback for wrapped ImageFolder
        class_names = dataset.subset.classes

    return loader, class_names