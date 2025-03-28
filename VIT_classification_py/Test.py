import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Train import Tumordata,model,Testloader,device
from torch.utils.data import Dataset,WeightedRandomSampler,DataLoader,Subset

#############################################################################################
#Test Phase
#############################################################################################
def test():
    state_dict = torch.load(r"D:\JHU\Phedias\VIT\VIT_classification\weights\best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    correct = 0
    total = 0 
    label_true = []
    label_predict = []
    model.eval()
    with torch.no_grad():
        for images_test,label_test in Testloader:
            images_test,label_test = images_test.to(device),label_test.to(device)
            outputs = model(images_test)
            _,predicted = outputs.max(1)
            correct += (predicted == label_test).sum().item()
            total += label_test.size(0) 
            label_true.extend(label_test.cpu().numpy())
            label_predict.extend(predicted.cpu().numpy())
            
        train_acc = correct/total 
    print(f"Test Accuracy: {train_acc*100:2f}") 

    return label_true, label_predict

#############################################################################################
#Confusion Matrix
#############################################################################################
def compute_confusion_matrix(label_true, label_predict):
    cm = confusion_matrix(label_true, label_predict)

    # Define the class names corresponding to the magnifications
    class_names = ['GBM', 'Astros', 'Oligos']

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



# Run the file
if __name__ == "__main__":
    label_true, label_predict = test()
    compute_confusion_matrix(label_true, label_predict)


