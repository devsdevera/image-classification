# COMP309 Capstone 
# Deveremma 300602434

import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import CNN


def evaluate(model, testloader, name):
    startTime = time.time()
    model.to(device)
    model.eval()

    classnames = testLoader.dataset.classes
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
    
    print(f'\nElapsed time {(time.time() - startTime):.2f}s')
            

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0)),               # Resize all images to 300x300
        transforms.RandomHorizontalFlip(),                                             # Randomly Flip Horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2),                          # Randomly alter image values
        transforms.ToTensor(),                                                         # Convert images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])             # This will standardize
    ])
    
    # Load test data from the /testdata directory
    testData = datasets.ImageFolder(root='./testdata', transform=transform)
    testLoader = DataLoader(testData, batch_size=32, shuffle=False)

    cnnModel = CNN().to(device)
    cnnModel.load_state_dict(torch.load('model.pth', weights_only=True, map_location=torch.device(device)))
    cnnModel.eval()

    evaluate(cnnModel, testLoader, 'CNN')


