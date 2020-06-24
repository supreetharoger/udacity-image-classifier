# AI Programming with Python Project

Image classifier built with PyTorch, then converted it into a command line application

# Part 1 - Development Notebook

Run Image Classifier Project.pynb using command:
jupyter notebook

# Part 2 - Command Line Application

Train.py successfully trains a network on a dataset of images

* Basic usage: python train.py data_directory
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
* Choose architecture: python train.py data_dir --arch "vgg16"
  vgg and densenet are supported in this project
* Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units [3136, 784] --epochs 5
* Use GPU for training: python train.py data_dir --gpu

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: python predict.py /path/to/image checkpoint
* Return top KK most likely classes: python predict.py input checkpoint --top_k 3
* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
* Use GPU for inference: python predict.py input checkpoint --gpu
