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
#Imports python modules

import argparse

def get_predict_args():
    parser = argparse.ArgumentParser()
    
    #Basic usage: python predict.py /path/to/image checkpoint
    parser.add_argument('path_to_image', help="Path to image file")
    
    #Checkpoint file
    parser.add_argument('checkpoint', help="Path to checkpoint file")
    
    #Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    parser.add_argument('--top_k', help="Return top KK most likely classes", default=3 , type=int, dest="top_k")
    
    #Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    parser.add_argument('--category_names', help="Use mapping of categories to real names", dest="category_names" ,default="cat_to_name.json", type=str)
    
    #Use GPU for inference: python predict.py input checkpoint --gpu
    parser.add_argument('--gpu', help="Choose GPU", dest="choose_gpu", default=False,  action="store_true")
    
    return parser.parse_args()

    
    
    
    
    