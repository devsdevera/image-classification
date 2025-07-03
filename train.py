# COMP309 Capstone 
# Deveremma 300602434

import time
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 300 * 300, 512)   # Flatten 3x300x300 input
        self.fc2 = nn.Linear(512, 128)             # 512 Features from First to 128
        self.fc3 = nn.Linear(128, 3)               # 128 to 3 class Output layer 

    def forward(self, x):
        x = x.view(-1, 3 * 300 * 300) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, 5, stride=1)                         
        self.conv2 = nn.Conv2d(15, 60, 5, stride=1)  
        self.conv3 = nn.Conv2d(60, 180, 5, stride=1) 
        self.conv4 = nn.Conv2d(180, 360, 5, stride=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        
        self.fc1 = nn.Linear(81000, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 3)    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.pool(F.relu(self.conv4(x))) 
        
        x = x.view(-1, 81000)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define pre-processing for images
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0)),               # Resize all images to 300x300
        transforms.RandomHorizontalFlip(),                                             # Randomly Flip Horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2),                          # Randomly alter image values
        transforms.ToTensor(),                                                         # Convert images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])             # This will standardize
    ])

    validation = False
    synthetic = True

    generated = datasets.ImageFolder(root='./generated', transform=transform)
    cleandata = datasets.ImageFolder(root='./cleandata', transform=transform)
    dataset = datasets.ImageFolder(root='./cleandata', transform=transform)

    trainSize = int(len(cleandata) - (len(cleandata) + len(generated)) * 0.2) if synthetic else int(len(dataset) * 0.8)
    testSize = len(cleandata) - trainSize if synthetic else int(len(dataset) - trainSize)

    # Split the dataset to automate selecting random images as the test set
    trainData, testData = random_split(cleandata, [trainSize, testSize]) if synthetic else random_split(dataset, [trainSize, testSize])
    trainData = ConcatDataset([generated, cleandata]) if not validation else ConcatDataset([trainData, generated]) if synthetic else trainData
    if not (validation or synthetic): trainData = dataset 

    if validation: print(f'full dataset size: {len(trainData) + len(testData)}')
    if validation: print(f'test dataset size: {len(testData)}')
    print(f'train dataset size {len(trainData)}')

    trainLoader = DataLoader(trainData, batch_size=24, shuffle=True, num_workers=12)
    testLoader = DataLoader(testData, batch_size=24, shuffle=False, num_workers=12)

    classnames = cleandata.classes
    print("Classes:", classnames)

    def evaluate(model, testloader, name):
        model.to(device)
        model.eval()

        classnames = cleandata.classes
        class_correct = [0] * len(classnames)
        class_total = [0] * len(classnames)
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                # Track correct predictions
                for i in range(len(labels)):
                    label = labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        overall_accuracy = 100 * sum(class_correct) / sum(class_total)
        print(f'{name} Unseen Accuracy: {overall_accuracy:.2f}%\n')

        for i, classname in enumerate(classnames):
            if class_total[i] > 0:
                class_accuracy = 100 * class_correct[i] / class_total[i]
                print(f'Accuracy of {classname}: {class_accuracy:.2f}%')
    
    mlpModel = MLP().to(device)
    mlpCriterion = nn.CrossEntropyLoss()                                      # Cross-entropy loss for classification
    mlpOptimizer = optim.SGD(mlpModel.parameters(), lr=0.001, momentum=0.9)   # Optimizer: Stochastic Gradient Descent

    def train(model, trainLoader, testLoader, criterion, optimizer, num_epochs=5):
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in trainLoader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()               # Clear the gradients from the previous iteration.
                outputs = model(inputs)             # Forward pass: compute model predictions.
                loss = criterion(outputs, labels)
                loss.backward()
                #nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                running_loss += loss.item()
                total += labels.size(0)
                
                # Calculate accuracy for this batch
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

            sched.step()
            # Calculate & print average loss and accuracy
            epoch_loss = running_loss / len(trainLoader)
            epoch_accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    #train(mlpModel, trainLoader, testLoader, mlpCriterion, mlpOptimizer, num_epochs=5)
    #torch.save(mlpModel.state_dict(), 'mlp.pth')
    #print("MLP model saved successfully")
    #if evaluate: evaluate(mlpModel, testLoader, 'MLP')

    import torch.nn.functional as F
    batchNormalize = False
    dropout = True

    for i in range(1):
        startTime = time.time()
        cnnModel = CNN().to(device)
        cnnCriterion = nn.CrossEntropyLoss()    
        cnnOptimizer = torch.optim.Adam(cnnModel.parameters(), lr=0.001, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(cnnOptimizer, max_lr=0.01, epochs=10, steps_per_epoch=len(trainLoader))
        
        train(cnnModel, trainLoader, testLoader, cnnCriterion, cnnOptimizer, num_epochs=10)
        torch.save(cnnModel.state_dict(), f'model.pth')
        print("CNN model saved successfully")
        print(f'CNN total train time {(time.time() - startTime):.2f}s')
        if validation: evaluate(cnnModel, testLoader, 'CNN')




