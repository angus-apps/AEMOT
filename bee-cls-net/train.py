import torch.optim as optim
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

from model import BeeClassifier
from dataloader import BeeDataset

import matplotlib.pyplot as plt

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Initialize the model
model = BeeClassifier()

# Initialize the dataset and dataloader
batch_size = 1024
num_epochs = 100


train_dir = "data/dataset_1/"
train_dataset = BeeDataset(root_dir=train_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dir = "data/dataset_2/"
val_dataset = BeeDataset(root_dir=val_dir, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Define the loss function and optimizer
criterion_BCE = nn.BCELoss() 
criterion_MSE = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to save a checkpoint
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'bee_classifier_epoch_{epoch+1}_bs_{batch_size}_delta.ckpt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')


def compute_loss(outputs, labels, criterion_BCE,criterion_MSE):
    bee_loss = criterion_BCE(outputs[:,0], labels[:,0].float())
    vel_loss = criterion_MSE(outputs[:,1:3], labels[:,1:3])
    loss = 0.5*bee_loss + 0.5*vel_loss
    return loss

# Initialize the plot
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12,6))
train_errors = []
train_accuracy = []
val_errors = []
val_accuracy = []

val_accuracy_max = 0
val_accuracy_max_epoch = 0


for epoch in range(num_epochs):
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    #----- Training -----#
    model.train()  # Set model to training mode
    for images, labels in train_dataloader:
        outputs = model(images)

        loss = compute_loss(outputs, labels, criterion_BCE,criterion_MSE)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store some diagnostics
        running_loss += loss.item()

        predicted = (outputs[:,0] > 0.5).float()
        correct_preds += (predicted == labels[:, 0].float()).sum().item()
        total_preds += labels.size(0)


    epoch_train_loss = running_loss / len(train_dataloader)
    train_errors.append(epoch_train_loss)
    train_accuracy.append(correct_preds / total_preds)


    #----- Validation -----#

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            # labels = labels.float().unsqueeze(1)  # Match output shape
            outputs = model(images)
                      
            loss = compute_loss(outputs, labels, criterion_BCE,criterion_MSE)

            # Diagnostics
            val_loss += loss.item()
            predicted = (outputs[:,0] > 0.5).float()
            correct_preds += (predicted == labels[:,0].float()).sum().item()
            total_preds += labels.size(0)


    epoch_val_loss = val_loss / len(val_dataloader)
    val_errors.append(epoch_val_loss)
    val_accuracy.append(correct_preds / total_preds)

    print(f'Epoch [{epoch+1}/{num_epochs}] \t Training Loss: {epoch_train_loss:.4f} \t Validation Loss: {epoch_val_loss:.4f}')

    if val_accuracy[-1] > val_accuracy_max:
        val_accuracy_max = val_accuracy[-1]
        val_accuracy_max_epoch = epoch

    # Update the plot
    ax[0].clear()
    ax[0].plot(train_errors, label='Training Loss')
    ax[0].plot(val_errors, label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlim([0,num_epochs])
    ax[0].set_ylim([0,7])
    ax[0].grid(which="both")
    ax[0].legend()

    ax[1].clear()
    ax[1].plot(train_accuracy, label='Training Accuracy')
    ax[1].plot(val_accuracy, label='Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_xlim([0,num_epochs])
    ax[1].set_ylim([0,1])
    ax[1].grid(which="both")
    ax[1].legend()
    ax[1].axhline(y=val_accuracy_max, color="r", linestyle="--", alpha=0.5)
    ax[1].axvline(x=val_accuracy_max_epoch, color="r", linestyle="--", alpha=0.5)
    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()

print(f"Maximum Validation Accuracy: {val_accuracy_max:.3f}  ({val_accuracy_max_epoch})")

# Save checkpoint at the end
save_checkpoint(epoch, model, optimizer, loss)
