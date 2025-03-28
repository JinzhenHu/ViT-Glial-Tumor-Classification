import os
import timm
import wandb
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from TumorDataset import TumorDataset,preprocessing
from torch.utils.data import Dataset,WeightedRandomSampler,DataLoader,Subset

#############################################################################################
#Preprocess Dataset
#############################################################################################
root_path = r"G:\.shortcut-targets-by-id\1fHTTWYsPLP4S3NscLgtDHETiYzEZR6rL\Eric Files"
Tumordata = TumorDataset(root_path,transform=preprocessing)

#Check the label distribution
print(f"Label Distribution: {Counter(Tumordata.labels)}")

#Split the data
train_idx, temp_idx = train_test_split(
     range(len(Tumordata)),
      test_size= 0.4,
        random_state=42,
        stratify =Tumordata.labels
        )

val_idx, test_idx = train_test_split(
     temp_idx,
      test_size= 0.5,
        random_state=42,
        stratify =[Tumordata.labels[i] for i in temp_idx]
        )

#Split into subset
train_data = Subset(Tumordata,train_idx)
valid_data = Subset(Tumordata,val_idx)
test_data = Subset(Tumordata,test_idx)

#Put it inside dataloader
Trainloader = DataLoader(train_data, batch_size =48, shuffle = True, pin_memory= True)
Valloader = DataLoader(valid_data, batch_size=48, shuffle=False, pin_memory=True)
Testloader = DataLoader(test_data, batch_size =48, shuffle = False, pin_memory= True)

#############################################################################################
#Load and fintune the model
#############################################################################################
model = timm.create_model(
  model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
  dynamic_img_size=True,
  pretrained=True,
)

# Add a classification head
num_features = model.num_features
num_classes = 3 ### Three classes
model.head = nn.Sequential(
    nn.LayerNorm(num_features),  
    nn.Linear(num_features, 256),  
    nn.GELU(),                  
    nn.Dropout(0.5),              
    nn.Linear(256, num_classes)    
)

# Freeze all the layers
for param in model.parameters():
    param.requires_grad = False
# Finetune the classifcation layer
for param in model.head.parameters():
    param.requires_grad = True
# Finetune the last five transformer block
for param in model.blocks[-5:].parameters():
    param.requires_grad = True

#AdamW Optimizer
optimizer = torch.optim.AdamW([
    {"params": model.head.parameters(), "lr":1e-5},
    {"params": model.blocks[-5:].parameters(), "lr":1e-4},
])
        
#############################################################################################
#Training and validation Phase
#############################################################################################
device = torch.device("cuda")
criteria = nn.CrossEntropyLoss()
epochs = 9  
model = model.to(device)

def train():
    wandb.init(
        project="VIT for classification",
        name="VIT_classification",
    )
    wandb.watch(model)
    steps = 0
    best_acc = 0.0  # track best validation accuracy for early stop
    patience = 3   # number of epochs to wait for improvement

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in Trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # Log training metrics step-by-step
            metrics = {
                "Training Loss inside epoch": loss.item(),
                "Step": steps,
                "Training Accuracy inside epoch": train_correct / train_total,
            }
            wandb.log(metrics)

        avg_train_loss = train_loss / len(Trainloader)
        train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images_val, labels_val in Valloader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                outputs_val = model(images_val)
                loss_val = criteria(outputs_val, labels_val)
                val_loss += loss_val.item()

                _, predicted_val = outputs_val.max(1)
                val_correct += (predicted_val == labels_val).sum().item()
                val_total += labels_val.size(0)

        avg_val_loss = val_loss / len(Valloader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
    # Log aggregated metrics for the epoch
        metrics = {
            "Training Loss": avg_train_loss,
            "Training Accuracy": train_acc,
            "Validation Loss": avg_val_loss,
            "Validation Accuracy": val_acc,
        }
        wandb.log(metrics)

        # Save the model if it achieves a new best validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_since_improvement =0
            torch.save(model.state_dict(), r"D:\JHU\Phedias\VIT\VIT_classification\weights\best.pth")
            print(f"New best model saved with accuracy: {best_acc:.4f}")
        else: 
            epochs_since_improvement +=1
            print(f"No improvement in validation accuracy for {epochs_since_improvement} epoch(s).")

        # Early stopping condition: stop if no improvement for `patience` epochs
        if epochs_since_improvement >= patience:
            print(f"Early stopping: Validation accuracy did not improve for {patience} consecutive epochs.")
            break

# Run the file
if __name__ == "__main__":
    train()
