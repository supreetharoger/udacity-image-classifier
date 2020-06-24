#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/print_functions_for_lab_checks.py
#                                                                             
# PROGRAMMER: Supreetha                                                  
# DATE CREATED: 22 June 2020
# REVISED DATE:             <=(Date Revised - if any)                         
# PURPOSE: check your code after programming each function.
#
##
import os

def check_command_line_arguments(in_arg):
    print(os.path.isdir(in_arg.data_directory))
    if in_arg is None:
        print("get_input_args is not defined")
    elif not os.path.isdir(in_arg.data_directory):
        print(f"Directory {in_arg.data_directory} not found")
        exit(1)
    elif not os.path.isdir(in_arg.save_dir):
        print(f"Directory {in_arg.save_dir} not found. Creating dir")
        os.makedirs(in_arg.save_dir)
    else:
        print("Command Line Arguments: \n save-dir =", in_arg.save_dir, 
              "\n arch =", in_arg.arch, " \n gpu=", in_arg.choose_gpu)