from PIL import Image
from torchvision import transforms
import torch
from model import BeeClassifier
import os

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Initialize the model
model = BeeClassifier()

# Function to load a pre-trained checkpoint
def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from '{checkpoint_path}'")

# Function to predict and return the label and confidence
def predict(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(image)
        confidence = output.item()
        label = 1 if confidence > 0.5 else 0  # 1 for 'bee', 0 for 'not bee'
        return label, confidence

# Function to evaluate the model on the entire dataset
def evaluate_model(dataset_path):
    correct_predictions = 0
    total_images = 0

    for label_dir in ['false', 'true']:
        full_path = os.path.join(dataset_path, label_dir)
        true_label = 0 if label_dir == 'false' else 1

        for img_name in os.listdir(full_path):
            img_path = os.path.join(full_path, img_name)
            predicted_label, _ = predict(img_path)
            if predicted_label == true_label:
                correct_predictions += 1
            total_images += 1

    average_accuracy = correct_predictions / total_images
    return average_accuracy

# Load the pre-trained model checkpoint
# checkpoint_path = './checkpoints/bee_classifier_epoch_50_bs_512.ckpt'
checkpoint_path = "./checkpoints/bee_classifier_epoch_50_bs_2048_gray_unscaled.ckpt"
load_checkpoint(checkpoint_path, model)

# Evaluate the model
# dataset_path = './dataset/28x28_scaled'
# dataset_path = "/home/angus/Desktop/training_images/"
dataset_path = "/home/angus/Desktop/gray_images/dataset_2/"


average_accuracy = evaluate_model(dataset_path)
print(f'Average Accuracy: {average_accuracy:.4f}')
