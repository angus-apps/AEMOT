import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BeeClassifier

model = BeeClassifier()

checkpoint = torch.load("./checkpoints/bee_classifier_epoch_100_bs_1024_delta.ckpt")


# import pdb; pdb.set_trace()
model.load_state_dict(checkpoint['model_state_dict']) 

model.eval()

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Save the TorchScript model
scripted_model.save("bee_classifier_scripted_epoch_100_bs_1024_delta.pt")



print("Model has been successfully saved as TorchScript!")
