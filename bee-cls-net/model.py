import torch
import torch.nn as nn
import torch.nn.functional as F




class BeeClassifier(nn.Module):
    def __init__(self):
        super(BeeClassifier, self).__init__()
        self.fc11 = nn.Linear(28 * 28, 28 * 28)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 1)  # Single output for binary classification
        self.fc4 = nn.Linear(64, 3)  # Three output for binary classification

    def forward(self, x):
        # x = x.view(-1, 28 * 28)  # Flatten the image
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary output
        # output = x

        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        x0 = torch.sigmoid(x[:, 0])
        x12 = torch.tanh(x[:, 1:3])      
        output = torch.cat((x0.unsqueeze(1), x12), dim=1)

        return output


