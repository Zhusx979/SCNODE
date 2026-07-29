import torch
import torchvision.models as models
import torch.nn as nn


def load_efficientnet_b0(num_classes=1000):
    # Load the pretrained EfficientNetB0 model
    model = models.efficientnet_b0()

    # Replace the classifier layer with a new one for the specific number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

def Get_efficientnet(num_classes):
    return load_efficientnet_b0(num_classes=num_classes)
