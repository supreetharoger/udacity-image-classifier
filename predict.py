import json
import torch
import os

from get_predict_args import get_predict_args
from PIL import Image
from torchvision import models
from torchvision import transforms

def main():
    in_arg = get_predict_args()
    
    #Selct GPU or CPU device
    if in_arg.choose_gpu and not torch.cuda.is_available(): 
        print(f"GPU is not enabled in your platform. Enable GPU")
        exit(1)
        
    device = torch.device("cuda:0" if in_arg.choose_gpu and torch.cuda.is_available() else "cpu")
    print(f"Device used: {device} ")
    
    if not os.path.exists(in_arg.category_names):
        print(f"Category name json file - {in_arg.category_names} does not exist")
        exit(1)

    with open(in_arg.category_names, 'r') as f:
       cat_to_name = json.load(f) 
    
    checkpoint_model = load_model(device, in_arg.checkpoint)
    
#The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
    top_probability, top_classes = predict(in_arg.path_to_image, checkpoint_model, in_arg.top_k)
    
    label = top_classes[0]
    probability = top_probability[0]
    
    #prints the most likely image class and it's associated probability
    print("Prediction class")
    print("------------------------------------")
    print(f'The most likely image class: {cat_to_name[label]}')
    print(f'Associated Probability: {probability*100:.2f}% \n')
    
    #Top K classes along with associated probabilities
    print("Top k classes")
    print("-------------------------------------")
    for i in range(len(top_probability)):
        print(f"{cat_to_name[top_classes[i]]} : {top_probability[i]*100:.2f}%")
    
# Write a function that loads a checkpoint and rebuilds the model
def load_model(device, file='savemodel.pth'):
    modelstate = torch.load(file, map_location=lambda storage, loc: storage)
    model = models.__dict__[modelstate['arch']](pretrained=True)
    model.classifier = modelstate['classifier']
    model.load_state_dict = modelstate['state_dict']
    model.optimizer_dict = modelstate['optimizer_dict']
    model.class_to_idx = modelstate['class_to_idx']
    model.to(device)
    return model

#Predict the class (or classes) of an image using a trained deep learning model.
def predict(image_path, model, topk=3):
    model.eval()
    model.cpu()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    # Disabling gradient calculation 
    with torch.no_grad():
        output = model.forward(image)
        top_probability, top_labels = torch.topk(output, topk)
        
        # Calculate the exponentials
        top_probability = top_probability.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_probability.numpy()[0], mapped_classes

#Scales, crops, and normalizes a PIL image for a PyTorch model,
#returns an Numpy array
def process_image(image):    
    means = [0.485, 0.456, 0.406]
    deviation = [0.229, 0.224, 0.225]
    
    pil_image = Image.open(image)
    pil_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, deviation)])
    pil_image = pil_transforms(pil_image)
    return pil_image

# Call to main function to run the program
if __name__ == "__main__":
    main()  