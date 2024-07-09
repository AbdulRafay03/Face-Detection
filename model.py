import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from logger import setup_logging
import torch.optim as optim     
import time
import os
import itertools
import numpy as np

logger = setup_logging('application.log')

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def forward(self, x):
        return self.model(x)

    def training(self , train_loader , val_loader , epochs , output_path):
        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Model training on {device}')
        self.to(device)

        # Loop over the dataset for multiple epochs
        for epoch in range(epochs):
            epoch_start_time = time.time()
            # Set the model to training mode
            self.train()
            # Loop over the training dataset
            for inputs, labels in train_loader:
                # Move inputs and labels to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self(inputs)
                # Calculate the loss
                loss = self.criterion(outputs, labels)
                # Backward pass
                loss.backward()
                # Update the weights
                self.optimizer.step()

            # Set the model to evaluation mode
            self.eval()

            # Track the total loss and number of correct predictions for validation
            val_loss = 0.0
            correct = 0
            total = 0

            # Disable gradient calculation during validation
            with torch.no_grad():
                # Loop over the validation dataset
                for inputs, labels in val_loader:
                    # Move inputs and labels to GPU
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Forward pass
                    outputs = self(inputs)
                    # Calculate the loss
                    val_loss += self.criterion(outputs, labels).item()
                    # Calculate the number of correct predictions
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Print the loss and accuracy for this epoch
            logger.info(f'Epoch [{epoch + 1}/{10}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {(100 * correct / total):.num_classesf}%')

            # Save the model weights
            out_path = os.path.join(output_path , 'face.pth')
        torch.save(self.state_dict(), out_path)
        return out_path


    def evalutaion(self, num_classes , model_path , test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = model_path
        self.load_state_dict(torch.load(model_path))
        self.to(device)

        self.eval()

        # Initialize metrics
        TP = [0] * num_classes
        TN = [0] * num_classes 
        FP = [0] * num_classes 
        FN = [0] * num_classes
        correct = 0   
        total = 0     

        all_labels = []
        all_preds = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
        
            outputs = self(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            for i in range(num_classes):  # Only for num_classes classes
                TP[i] += ((predicted == i) & (labels == i)).sum().item()
                TN[i] += ((predicted != i) & (labels != i)).sum().item()
                FP[i] += ((predicted == i) & (labels != i)).sum().item()
                FN[i] += ((predicted != i) & (labels == i)).sum().item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate metrics for each class
        sensitivity = [TP[i] / (TP[i] + FN[i]) for i in range(num_classes)]
        specificity = [TN[i] / (TN[i] + FP[i]) for i in range(num_classes)]
        weighted_error = [(FP[i] + FN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]) for i in range(num_classes)]
        accuracy = correct / total

        # Calculate overall metrics (average of class-wise metrics)
        overall_sensitivity = sum(sensitivity) / num_classes
        overall_specificity = sum(specificity) / num_classes
        overall_weighted_error = sum(weighted_error) / num_classes

        logger.info(f'Class-wise Sensitivity: {sensitivity}')
        logger.info(f'Class-wise Specificity: {specificity}')
        logger.info(f'Class-wise Weighted Error: {weighted_error}')
        logger.info(f'Class-wise Accuracy: {accuracy:.4f}')
        logger.info(f'Overall Sensitivity: {overall_sensitivity:.4f}')
        logger.info(f'Overall Specificity: {overall_specificity:.4f}')
        logger.info(f'Overall Weighted Error: {overall_weighted_error:.4f}')
        logger.info(f'Overall Accuracy: {accuracy:.4f}')


