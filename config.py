# config.py
import os
import torch

# -- Training Hyperparameters --
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10

# -- Data and Model Configuration --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MANUAL_SEED = 42

# -- File Paths --
BASE_DIR = "C:/Users/adams/OneDrive/Documents/AI/"
TRAIN_DIR = os.path.join(BASE_DIR, "PetImages")
TEST_DIR = os.path.join(BASE_DIR, "PetImages2")

MODEL_SAVE_DIR = "saved_models"
MODEL_NAME_BASE = "img_classifier"

# -- Dataset Format Selector --
# Set to True to use COCO JSON, False to use Folder structure (PetImages)
USE_COCO_FORMAT = False 

# -- COCO Configuration (Only used if USE_COCO_FORMAT is True) --
# Path to the folder containing the images
COCO_TRAIN_IMG_DIR = "C:/Users/adams/OneDrive/Documents/AI/TPV/train/" 
COCO_TEST_IMG_DIR = "C:/Users/adams/OneDrive/Documents/AI/TPV/valid/"

# Path to the .json annotation files
COCO_TRAIN_ANN = "C:/Users/adams/OneDrive/Documents/AI/TPV/train/_annotations.coco.json"
COCO_TEST_ANN = "C:/Users/adams/OneDrive/Documents/AI/TPV/valid/_annotations.coco.json"
