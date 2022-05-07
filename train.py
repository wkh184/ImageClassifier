# Imports python modules
from time import time, sleep

import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
#import helper
#import fc_model
import numpy as np
from torch import nn

from PIL import Image
import glob, os
from torch.autograd import Variable

import matplotlib.pyplot as plt
import os, random
import json
from collections import OrderedDict
#import time
# Imports functions created for this program
from get_input_args import get_input_args


#Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
# validation function
def validate(model, test_dataloader, device, criterion):
    """
    Validate
    Parameters:
     model - model to be used
     test_dataloader - test set dataloader
     device - CPU or GPU
     criterion - criterion to be used 
    Returns:
     None  
    """
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(test_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        loss = criterion(output, target)
        
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    
    return val_loss, val_accuracy

#Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
# training function
def fit(model, train_dataloader, device, optimizer, criterion):
    """
    Fit
    Parameters:
     model - model to be used
     train_dataloader - training set dataloader
     device - CPU or GPU
     optimizer -  optimizer to be used
     criterion - criterion to be used 
    Returns:
     None  
    """
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy

def load_data(data_dir):
    """
    Load data from data directory
    Parameters:
     data_dir - data directory
    Returns:
     Dataloaders for training, testing and validation  
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
         'training' : transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]),
                                                            
        'validation' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

        'testing' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    }
    print("Loading from {}".format(train_dir))
    print("Loading from {}".format(test_dir))
    print("Loading from {}".format(valid_dir))

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=32, shuffle=True),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=32, shuffle=False),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle=True)
    }
    print("Loaded from {}".format(train_dir))
    print("Loaded from {}".format(test_dir))
    print("Loaded from {}".format(valid_dir))
    return image_datasets, dataloaders

def load_model(device, learning_rate):
    """
    Load model
    Parameters:
     device - cpu or gpu
     learning_rate - learning rate to be set
    Returns:
     Model, optimizer and criterion  
    """    
    #Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
    model = models.vgg19(pretrained=True)

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)), # First layer
                          ('relu', nn.ReLU()), # Apply activation function
                          ('fc2', nn.Linear(4096, 102)), # Output layer
                          ('output', nn.LogSoftmax(dim=1)) # Apply loss function
                          ]))
    model.classifier = classifier    

    # freeze convolution weights
    for param in model.features.parameters():
        param.requires_grad = False

    model.to(device)
    
    # optimizer
    print("learning_rate = {}".format(learning_rate))

    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)
    # loss function
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion 

def train(model, dataloaders, epoch_to_train, device, optimizer, criterion):
    """
    Train the model
    Parameters:
     model - model loaded to be trained
     dataloaders - dataloaders to get datasets
     epoch -  number of epoch to run training
     device - cpu or gpu
     optimizer - optimizer
     criterion - criterion
    Returns:
     Model
    """    
    #Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []
    start = time()
    
    print("epoch_to_train = {}".format(epoch_to_train))
    for epoch in range(epoch_to_train):
        print("Starting epoch = {}".format(epoch))
        print("Fitting")
        train_epoch_loss, train_epoch_accuracy = fit(model, dataloaders['training'], device, optimizer, criterion)
        print("Validating")
        val_epoch_loss, val_epoch_accuracy = validate(model, dataloaders['testing'], device, criterion)
        print("val_epoch_accuracy = {}".format(val_epoch_accuracy))
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        #stop training when accuracy exceeds 85
        if val_epoch_accuracy > 85:
            print("Exiting at epoch = {}".format(epoch))
            break
    end = time()
    print('Training time: ', (end-start)/60, ' minutes')
    return model

def validate_model(model, validation_dataloader):
    #run against validation dataset
    model.to('cpu')
    images, labels = next(iter(validation_dataloader))
    # Get the class probabilities
    ps = torch.exp(model(images))
    print(ps.shape)
    top_p, top_class = ps.topk(1, dim=1)
    # Look at the most likely classes for the first 10 examples
    print(top_class[:10,:])
    matches = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(matches.type(torch.FloatTensor))
    return accuracy
        
def save_checkpoint(model, image_datasets):
    model.class_to_idx = image_datasets['training'].class_to_idx
    torch.save({'arch': 'vgg19',
            'input_size': 25088,
            'output_size': 102,
            'state_dict': model.state_dict(), # Holds all the weights and biases
            'class_to_idx': model.class_to_idx},
            'model.pth')
    print("Model saved as 'model.pth'")


# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    # check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    in_arg = get_input_args()
    print(in_arg)
#    check_command_line_arguments(in_arg)
    image_datasets, dataloaders = load_data(in_arg.data_directory)
    #model, optimizer, criterion = load_model(device, 0.001)
    model, optimizer, criterion = load_model(device, in_arg.learning_rate)
    print("Model loaded")
    print(model)
    #model = train(model, dataloaders, 9, device, optimizer, criterion)    
    model = train(model, dataloaders, in_arg.epochs, device, optimizer, criterion)    
    print("Model trained")
    print(model)
    print("Validating model")
    accuracy = validate_model(model, dataloaders['validation'])
    print(f'Accuracy: {accuracy.item()*100}%')
    print("Saving checkpoint")
    save_checkpoint(model, image_datasets)
    
    end_time = time()
    
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
