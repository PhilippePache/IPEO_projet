import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import os 
import matplotlib.pyplot as plt

## MS2

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
        self.input_size = input_size
        self.n_classes = n_classes
        self.fc1 = nn.Linear(input_size, 512)       # First fully connected layer
        self.dropout1 = nn.Dropout(p=0.4)           # Dropout to prevent overfitting
        self.fc2 = nn.Linear(512, 256)              # Second fully connected layer
        self.dropout2 = nn.Dropout(p=0.4)           
        self.fc3 = nn.Linear(256, 128)              # Third fully connected layer
        self.dropout3 = nn.Dropout(p=0.4)
        self.fc4 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.fc1(x))     # Apply ReLU activation function
        x = self.dropout1(x)        # Apply dropout 
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        preds = self.fc4(x)         # Final layer to get logits for each class
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
        # Feature extraction part - CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.3)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.4)
        
        # Calculate the output size after convolutional layers
        # Original image size: 28x28
        # After 3 pooling layers with stride 2: 28/2/2/2 = 3.5 -> 3
        # So the feature map size will be 3x3 with 128 channels
        self.feature_size = 128 * 3 * 3
        
        # Classification part - Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 256)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Flatten the features for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        preds = self.fc2(x)

        return preds
    

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, device, results_file=None, npy_file=None):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training 
            batch_size (int): number of data points in each batch
        """
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        # Loss function for classification
        self.criterion = nn.CrossEntropyLoss()
        # Adaptive Moment Estimation (Adam) optimizer with weight decay to prevent overfitting
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # Learning rate scheduler to reduce the learning rate by a factor of 0.5 every 10 epochs
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # Save results and numpy file paths
        self.results_file = results_file
        self.npy_file = npy_file
    
    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        epoch_accuracies = []  # List to store average accuracy for each epoch
        for ep in range(self.epochs):
            # Train the model for one epoch and get the accuracies
            accuracies = self.train_one_epoch(dataloader, ep)
            avg_acc = np.mean(accuracies)
            epoch_accuracies.append(avg_acc)    # Store epoch accuracy
            self.scheduler.step()               # Schedule the learning rate
            tqdm.write(f"Epoch {ep+1}/{self.epochs}, Accuracy: {avg_acc:.2f}%")
            
        # Plot and save the accuracy per epoch
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(epoch_accuracies)+1), epoch_accuracies, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Average Accuracy (%)")
        plt.title("Training Accuracy per Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block = False)
        plt.savefig("accuracy_plot.png")
        plt.close()
            
    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        
        # Set the model to training mode and initialize the total loss
        self.model.train()
        total_loss = 0              # Sum of losses for the epoch
        correct = 0                 # Number of correct predictions
        total = 0                   # Number of samples processed
        accuracies = []             # List to store accuracies for each batch
        batch_iterator = tqdm(dataloader, desc=f"Epoch {ep+1}/{self.epochs}", leave=False, ncols=100)

        for it, batch in enumerate(dataloader):

            #Load a batch, break it down in images and targets
            x, y = batch

            # Move data and labels to the device
            x, y = x.to(self.device), y.to(self.device).long()
            
            # Run Forward pass
            logits = self.model(x)      

            #Compute loss 
            loss = self.criterion(logits, y)    
            
            # Run Backward pass
            loss.backward()    
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
            
            # Update weights avec les gradients calcules 
            self.optimizer.step()
            
            # Zero the gradients
            self.optimizer.zero_grad()

            # Update the total loss
            total_loss += loss.item()

            # Update progress bar with current loss
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

            #Compute accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            accuracy = correct / total * 100
            accuracies.append(accuracy)
        return accuracies 
    
        # Average the loss over the number of batches
        #return total_loss / len(dataloader)
    
    
    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        
        # Set the model to evaluation mode and initialize an empty list for predictions
        self.model.eval()
        all_preds = []
        # Loop over the batches in the dataloader
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                x = batch[0]
                x = x.to(self.device)
                logits = self.model(x)
                # Get the predicted labels by taking the argmax of the logits
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
        # Concatenate all predictions into a single tensor
        return torch.cat(all_preds, dim=0)


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
