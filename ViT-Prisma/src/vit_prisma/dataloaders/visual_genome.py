import os
import csv
from PIL import Image
import torch as t
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dotenv import load_dotenv
import cv2
from os import listdir
from PIL import Image as PImage
import random
from tqdm import tqdm
# Define image transformations

# Create a custom Dataset class for Visual Genome


        
# Load the Visual Genome dataset
def transform():
    '''
    Necessary transform to preprocess data as it is loaded.
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


# Instantiate the DataLoader
def instantiate_dataloader(data: t.Tensor,
                           labels: list):
    dataset = VisualGenomeDataset(data, labels, transform=transform)
    visual_genome_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset,visual_genome_loader

# Iterate through the DataLoader
def iterate_dataloader(visual_genome_loader):

    for batch_indx, (images, labels) in enumerate(visual_genome_loader):
        print(f"Batch {batch_indx}:")
        print("Batch of images:", images)
        print("Batch of labels:", labels)
        break  # Just to demonstrate, we break after the first batch

def set_up_vg_paths(verbose=True):
    load_dotenv()
    base_dir = os.getenv("visual_genome_directory")
    train_path = os.path.join(base_dir,"images/VG_100K")

    if verbose:
        print(f"Working directory: {os.getcwd()}")
        print(f"Base directory: {base_dir}")
        print(f"Training path exists! {train_path}")
    
    return train_path

def load_images(train_path, batch_size=None,verbose=True):
    '''
    Load images from the training path of Visual Genome. 
    '''
    valid_extensions = {'.jpg', '.jpeg','.png','.bmp','.tiff'}
    train_path = set_up_vg_paths()
    images_list = listdir(train_path)
    loaded_images = []
    valid_count = 0
    pbar = tqdm(total=batch_size,desc="Loading Visual Genome images...")
    for file in images_list:
        # Loop through files in the image list
        # Display Progress Bar 
        ext = os.path.splitext(file)[1].lower() # Obtain the file extention
        if verbose:
            print(f"File: {file}, Extension: {ext}")
        if ext not in valid_extensions:
            if verbose:
                print(f"Skipping file with invalid extension: {file}")
            continue
        full_paths = os.path.join(train_path,file)
        img = cv2.imread(full_paths,batch_size) # Read an image from its full path
        if img is not None:
            # Check if the file is not empty
            
    
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Converting the color scale of an image from BGR to RGB because transforms expects RGB
            loaded_images.append(img)
            pbar.update(1)
        valid_count += 1
        if batch_size is not None and len(loaded_images) >= batch_size:
            # Stop loading at batch size
            break
    print(f"Valid images loaded: {valid_count}")
    pbar.close()
    return loaded_images

class VisualGenomeDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
