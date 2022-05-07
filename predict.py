# Imports python modules
from time import time, sleep
from get_input_args_predict import get_input_args
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import json
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms

def get_cat_names(cat_filename):
    print("cat_filename = {}".format(cat_filename))
    with open(cat_filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    print("image_path = {}".format(image_path))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image_path)
    return image

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_path):
    print("checkpoint_path = {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)), # First layer
                          ('relu', nn.ReLU()), # Apply activation function
                          ('fc2', nn.Linear(4096, 102)), # Output layer
                          ('output', nn.LogSoftmax(dim=1)) # Apply loss function
                          ]))
    
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print("topk = {}".format(topk))
    img = Image.open(image_path)
    processed_image = process_image(img)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_labs = probs.topk(topk)

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    # check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    in_arg = get_input_args()
    print(in_arg)
    cat_to_name = get_cat_names(in_arg.category_names)
    print("JSON:\n")    
    print(cat_to_name)
    print("\nJSON Length:", len(cat_to_name))
    print("\nCat 1 name:", cat_to_name['1'])
    print("Load model from {}".format(in_arg.checkpoint))
    model = load_model(in_arg.checkpoint)
    print(model)
    probs, classes, flowers = predict(in_arg.image_full_path, model, cat_to_name, in_arg.top_k)
    print(probs)
    print(classes)
    #print([cat_to_name[x] for x in classes])
    image_path = in_arg.image_full_path
    flower_num = image_path.split('/')[2]
    print("flower_num = {}".format(flower_num))
    flower_image = cat_to_name[flower_num] # Calls dictionary for name

    print("Results for {}:\n".format(flower_image))
    for x in classes:
        #print(x)
        print(cat_to_name[str(x)])    
    end_time = time()
    
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
