import sys
import os
from shutil import copyfile
from torch.backends import cudnn
import random
import numpy as np
import nibabel as nibabel
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings
import nibabel as nib

deep_learning_model_path = '../Healthy_Human_Brain_Model/deep_learning_model/'
model_weights_path = '../Healthy_Human_Brain_Model/saved_model_weights/'
inference_path = '../Test_retest_data_complete/NatureBME_share_nonContrast/'
inference_output_path = '../Newly_generated_prediction/'

sys.path.append(deep_learning_model_path)
from solver import Solver
from data_loader import get_loader_inference

class config():   
    # model hyper-parameters
    t = int(3) # t for Recurrent step of R2U_Net or R2AttU_Net'
    img_ch = int(1)
    output_ch = int(1)
    num_epochs = int(50)
    num_workers = int(1)
    mode = 'inference'
    
    model_type = 'ResAttU_Net' # 'U_Net/R2U_Net/AttU_Net/R2AttU_Net/ResAttU_Net'
    optimizer_choice = 'SGD'
    initialization = 'NA'
    UnetLayer = 6 # This is only implemented in ResAttU_Net
    first_layer_numKernel = 64 # How many kernels in the first convolutional layer? Will be halfed every layer downward.
    which_unet = '18'
    inference_filename_start_end = [0, -19]
    
    initial_lr = float(0.1)  # initial learning rate
    adaptive_lr = True # whether or not to use adaptive learning rate
    loss_function_name = 'CVPR_Adaptive_loss' # MSE/SmoothL1/CVPR_Adaptive_loss
    batch_size = int(4)

    cuda_idx = int(0)

    current_model_saving_path = model_weights_path
    inference_input_folder = inference_path
    inference_slice_dimension = 2
    current_inference_output_path = inference_output_path

for this_directory in [config.current_inference_output_path]:
    if not os.path.exists(this_directory):
        os.makedirs(this_directory)

cudnn.benchmark = True

if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net', 'ResAttU_Net']:
    print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    print('Your input for model_type was %s'%model_type)
if config.loss_function_name not in ['SmoothL1', 'MSE', 'SmoothL1WithKL', 'CVPR_Adaptive_loss']:
    print('ERROR!! loss_function should be selected in MSE/SmoothL1/SmoothL1withKL')

inference_loader = get_loader_inference(inference_input_folder = config.inference_input_folder,
                                        batch_size = int(1),
                                        inference_slice_dimension = config.inference_slice_dimension,
                                        num_workers = config.num_workers,
                                        shuffle = False)

solver = Solver(config, None, None, None, inference_loader)

warnings.filterwarnings(action = 'once')

solver.inference(config.which_unet)