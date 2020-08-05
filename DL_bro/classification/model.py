import torch
import torch.nn as nn
import torchvision

def model_selection(model_name, num_classes):

    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'custom':
        print('put your model class')
        # import dl_model
        # model = dl_model.DLbro() <-- put your model class

    else:
        print('Invalid model name.')

    return model