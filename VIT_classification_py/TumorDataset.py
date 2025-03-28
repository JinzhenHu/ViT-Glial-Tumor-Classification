import os
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset,WeightedRandomSampler,DataLoader,Subset

class TumorDataset(Dataset):
    def __init__(self, image_root,transform = None,samples_per_class =6000):
        self.images = []
        self.labels = []
        self.transform = transform
        self.label_mapping = {"GBM":0, "Astros":1, "Oligos":2}
        self.samples_per_class = samples_per_class


        #GBM
        tumor_type = "GBM"
        root_path_gbm = os.path.join(image_root,tumor_type)
        for root,_,file in os.walk(root_path_gbm):
            for name in file:
                if name.lower().endswith(".jpg"):
                    gbm_path = os.path.join(root, name)
                    self.images.append(gbm_path)
                    self.labels.append(self.label_mapping[tumor_type])

        #Astros
        tumor_type = "Astros"
        root_path_gbm = os.path.join(image_root,tumor_type)
        for root,_,file in os.walk(root_path_gbm):
            for name in file:
                if name.lower().endswith(".jpg"):
                    gbm_path = os.path.join(root, name)
                    self.images.append(gbm_path)
                    self.labels.append(self.label_mapping[tumor_type])              

        #Oligos
        tumor_type = "Oligos"
        root_path_gbm = os.path.join(image_root,tumor_type)
        for root,_,file in os.walk(root_path_gbm):
            for name in file:
                if name.lower().endswith(".jpg"):
                    gbm_path = os.path.join(root, name)
                    self.images.append(gbm_path)
                    self.labels.append(self.label_mapping[tumor_type])  
                    
        self._sample_5000_per_class()

    def _sample_5000_per_class(self):
        combined = list(zip(self.images, self.labels))
        class_samples = {0: [], 1: [], 2: []}
        for img, lbl in combined:
            class_samples[lbl].append((img, lbl))

        sampled_images = []
        sampled_labels = []
        for lbl, samples in class_samples.items():
            sampled = random.sample(samples, min(self.samples_per_class, len(samples)))
            for img, label in sampled:
                sampled_images.append(img)
                sampled_labels.append(label)

        self.images = sampled_images
        self.labels = sampled_labels

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image_T = self.transform(image)
        label = self.labels[idx]
        return image_T, label
    
    def __len__(self):
        return len(self.images)

preprocessing = v2.Compose(
  [
    v2.ToImage(),
   # v2.Resize(size=224),
    v2.CenterCrop(size=224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
      mean=(0.5, 0.5, 0.5),
      std=(0.5, 0.5, 0.5),
    ),
  ]
)

# Run the file
if __name__ == "__main__":
    root_path = r"G:\.shortcut-targets-by-id\1fHTTWYsPLP4S3NscLgtDHETiYzEZR6rL\Eric Files"
    Tumordata = TumorDataset(root_path,transform=preprocessing)



