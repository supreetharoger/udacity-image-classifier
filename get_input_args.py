#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: SUPREETHA
# DATE CREATED: 19 June 2020                              
# REVISED DATE: 
# PURPOSE: Command Line Arguments:
#     1. Image Folder - data_directory
#     2. CNN Model Architecture as --arch with default value 'vgg16'
#     3. Save directory - --save_dir , default - current directory
#     4. GPU - choose GPU , default - False
#     5. Argument group - Learning rate - default - 0.001, epochs - 3, hidden units - [3136, 784]
#
##
# Imports python modules
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    #Basic usage: python train.py data_directory
    parser.add_argument('data_directory', action='store')
    
    #Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    parser.add_argument('--save_dir', dest="save_dir", type=str, default=".", help="Set directory to save checkpoints")
    
    #Choose architecture: python train.py data_dir --arch "vgg13"
    parser.add_argument('--arch', dest="arch", type=str, default='vgg16', help='Choose CNN model architecture')
    
    #Use GPU for training: python train.py data_dir --gpu
    parser.add_argument('--gpu', dest="choose_gpu", action="store_true", help="Choose GPU", default=False )
    
    #category names
    parser.add_argument('--category_names', help="Use mapping of categories to real names", dest="category_names" ,default="cat_to_name.json", type=str)
    
    #Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--learning_rate', dest="learning_rate", default=0.001, type=float, help='Learning rate')
    hyperparameters.add_argument('--hidden_units', dest="hidden_units", default=[3136, 784], type=int, help="Hidden units")
    hyperparameters.add_argument('--epochs', dest="epochs", default=2, type=int, help='epochs')
    
    return parser.parse_args()
    