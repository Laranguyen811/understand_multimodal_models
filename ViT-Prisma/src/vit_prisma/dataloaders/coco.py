import os
import stat
import torch as t
from dotenv import load_dotenv
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions


class ConvertTo3Channels:
    def __call__(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img


transform = transforms.Compose([
    ConvertTo3Channels(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def coco_collate_fn(batch):
    images, captions = zip(*batch)
    images = t.stack(images, dim=0)
    return images, list(captions)


def setup_coco_paths(verbose=False):
    load_dotenv(dotenv_path='.env')
    base_dir = os.getenv("directory")
    train_path = os.path.join(base_dir, "train2017", "train2017")
    ann_path = os.path.join(base_dir, "annotations_trainval2017", "annotations", "captions_train2017.json")
    ann_path = os.path.normpath(ann_path)
    
    if verbose:
        print(f"Working Directory: {os.getcwd()}")
        print(f"Base directory: {base_dir}")
        print(f"Train path exists: {os.path.exists(train_path)}")
        print(f"Annotation file exists: {os.path.exists(ann_path)}")
    
    return train_path, ann_path


class CocoLoader:
    def __init__(self, batch_size=8, shuffle=True):
        train_path, ann_path = setup_coco_paths()
        
        self.coco_dataset = CocoCaptions(
            root=train_path,
            annFile=ann_path,
            transform=transform,
        )
        
        self.coco_loader = DataLoader(
            self.coco_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=coco_collate_fn,
        )