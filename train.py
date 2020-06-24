import json
import torch
from get_input_args import get_input_args
from validation_function_check import *
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

#Train the network
def main():
    in_arg = get_input_args()
    
     # Function that checks command line arguments using in_arg  
    check_command_line_arguments(in_arg)
    
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    #Transforms for training
    training_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25), 
                                          transforms.RandomResizedCrop(224), 
                                          transforms.RandomGrayscale(p=0.02),
                                          transforms.RandomRotation(25),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, std)])
    training_datasets = datasets.ImageFolder(in_arg.data_directory+'/train', transform=training_transforms)
    training_loader = torch.utils.data.DataLoader(training_datasets, batch_size=32, shuffle=True)
    
    #Transforms for validation
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, std)])
    validation_datasets = datasets.ImageFolder(in_arg.data_directory+'/valid',                                                                                      transform=validation_transforms)
    validation_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
    
    model = models.__dict__[in_arg.arch](pretrained=True)
    
    densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }

    if in_arg.arch.startswith('vgg'):
        in_features = model.classifier[0].in_features
    if in_arg.arch.startswith('densenet'):
        in_features = densenet_input[in_arg.arch]
        
    
    for param in model.parameters():
        param.requires_grad = False
    
    if not os.path.exists(in_arg.category_names):
        print(f"Category name json file - {in_arg.category_names} does not exist")
        exit(1)
        
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    hidden_layer = in_arg.hidden_units
    hidden_layer.insert(0, in_features)
    print(hidden_layer)
    orderedDict = OrderedDict()
    
    for i in range(len(hidden_layer)-1):
        orderedDict['fc'+str(i+1)] = nn.Linear(hidden_layer[i], hidden_layer[i+1])
        orderedDict['relu'+str(i+1)] = nn.ReLU()
        orderedDict['dropout'+str(i+1)] = nn.Dropout(p=0.15)
    
    orderedDict['output'] = nn.Linear(hidden_layer[i+1], len(cat_to_name))
    orderedDict['softmax'] = nn.LogSoftmax(dim=1)
    
    classifier = nn.Sequential(orderedDict)
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    
    #Selct GPU or CPU device
    if in_arg.choose_gpu and not torch.cuda.is_available(): 
        print(f"GPU is not enabled in your platform. Enable GPU")
        exit(1)
    
    device = torch.device("cuda:0" if in_arg.choose_gpu and torch.cuda.is_available() else "cpu")
    print(f"Device used: {device} ")
    model.to(device)
    
    #Training the network
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 50
    
    for epoch in range(epochs):
        total = 0
        accuracy = 0
        for inputs, labels in training_loader:
            steps +=1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                accuracy = 0
                # Calculate accuracy
                #_, p = torch.max(logps.data, 1)
                total += labels.size(0)
                #accuracy += (p == labels).sum().item()
                
                ps = torch.exp(logps)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Training loss: {running_loss/steps:.3f}.. ")
                running_loss = 0
                model.train()
    
        #validation
        validation_total = 0
        validation_loss = 0
        validation_accuracy = 0
        count=0
    
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                validation_loss += criterion(logps, labels).item()
                
                ps = torch.exp(logps)
                equality = (labels.data == ps.max(dim=1)[1])
                validation_accuracy += equality.type(torch.FloatTensor).mean()
                
                count += 1
            print(f"Validation for Epoch {epoch + 1}/{epochs}")
            print("-------------------------------------------")
            print(f"Validation Loss {validation_loss/count:.3f}")
            if validation_accuracy > 0:
                print(f"Validation accuracy: {(validation_accuracy/count) * 100:.2f}%")
            
    print("Training Done")
    print("Setting directory to save checkpoint")
    
    model.class_to_idx = training_datasets.class_to_idx
    modelstate = {
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
        'arch': in_arg.arch
    }

    torch.save(modelstate, in_arg.save_dir+'/savemodel.pth')
    
# Call to main function to run the program
if __name__ == "__main__":
    main()    
    