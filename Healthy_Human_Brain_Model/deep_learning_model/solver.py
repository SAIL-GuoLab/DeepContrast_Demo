import os
from glob import glob
import numpy as np
import nibabel as nib
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, KLDivLoss
import torch.nn.functional as Ff
from torch.nn.utils import clip_grad_norm_
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, ResAttU_Net, init_weights
import csv
from scipy import ndimage
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

torch.manual_seed(202001)
torch.cuda.manual_seed_all(202001)
np.random.seed(202001)

class Solver(object):
	def __init__(self, config, train_loader, validation_loader, test_loader, inference_loader = None):
		"""Initialize our deep learning model."""
		# Data loader
		self.train_loader = train_loader
		self.validation_loader = validation_loader
		self.test_loader = test_loader
		self.inference_loader = inference_loader

		# Models
		self.unet = None
		self.optimizer_choice = config.optimizer_choice
		self.initialization = config.initialization
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.UnetLayer = config.UnetLayer
		self.first_layer_numKernel = config.first_layer_numKernel
		self.device = torch.device('cuda: %d' % config.cuda_idx)
		if 'inference_filename_start_end' in config.__dict__:
			self.inference_filename_start_end = config.inference_filename_start_end

		# Hyper-parameters
		self.initial_lr = config.initial_lr
		self.current_lr = config.initial_lr
		if 'loss_function_lr' in config.__dict__:
			self.loss_function_lr = config.loss_function_lr
		if 'adaptive_lr' in config.__dict__:
			self.adaptive_lr = config.adaptive_lr

		if 'clipped_gradient' in config.__dict__:
			if config.clipped_gradient != 'NA':
				self.clipped_gradient = config.clipped_gradient

		# Loss Function
		if config.loss_function_name == 'MSE':
			self.loss_function_name = 'MSE'
		elif config.loss_function_name == 'SmoothL1':
			self.loss_function_name = 'SmoothL1'
		elif config.loss_function_name == 'CVPR_Adaptive_loss':
			self.loss_function_name = 'CVPR_Adaptive_loss'

		# Training settings
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size

		# Path
		self.current_model_saving_path = config.current_model_saving_path
		if 'inference_input_folder' in config.__dict__:
			self.inference_input_folder = config.inference_input_folder
			self.current_inference_output_path = config.current_inference_output_path
			self.inference_slice_dimension = config.inference_slice_dimension

		self.mode = config.mode

		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build our deep learning model."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch = 1, output_ch = 1, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch = 1, output_ch = 1, t = self.t, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch = 1, output_ch = 1, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch = 1, output_ch = 1, t = self.t, first_layer_numKernel = self.first_layer_numKernel)
		elif self.model_type == 'ResAttU_Net':
			self.unet = ResAttU_Net(UnetLayer = self.UnetLayer, img_ch = 1, output_ch = 1, first_layer_numKernel = self.first_layer_numKernel)

		if self.initialization != 'NA':
			init_weights(self.unet, init_type = self.initialization)
		self.unet.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))
		
	def to_data(self, x):
		"""Convert tensor to numpy."""
		if torch.cuda.is_available():
			x = x.cpu().detach().numpy()
		return x

	def inference(self, which_unet = 'best', save_prediction_nifti = True):
		"""Use the deep learning model to generate prediction on unseen data."""

		#===================================== Inferece ==================================#
		#=================================================================================#
		self.build_model()

		if str(which_unet).isdigit() == True:
			all_unet_path = os.path.join(self.current_model_saving_path, '%s-%s-%.4f-%s-%d-%s%s.pkl' %(self.model_type, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, 'epoch', str(which_unet).zfill(2)))
			self.unet.load_state_dict(torch.load(all_unet_path))
		else:
			print('Input argument which_unet is invalid. Has to be "best" or "last" or an integer representing the epoch.')

		# Make the directory for the current prediction.
		if not os.path.exists(self.current_inference_output_path):
			os.makedirs(self.current_inference_output_path)

		print('The inference scans will be saved at: ', self.current_inference_output_path)

		# We only allow saving the gado uptake ground truth (GT) if 'save_prediction_nifti' is True.
		if save_prediction_nifti == False and save_GT_nifti == True:
			save_prediction_nifti = True
			print('The parameter "save_prediction_nifti" is set to "True". We only allow saving the GT when we are also saving the predictions.')

		self.unet.train(False)
		self.unet.eval()
		length = 0
        
		inference_input_paths = list(np.sort(glob(self.inference_input_folder + '*.nii.gz')))

		with torch.no_grad():
			for batch, (pre_image) in enumerate(tqdm(self.inference_loader)):
				# Check the batch size of the inference loader. We only support "1".
				inference_batch_size = pre_image.shape[0]
				assert inference_batch_size == 1
				del inference_batch_size

				# Create a list to store the number of slices for each scan in the inference folder.
				slices_by_scan = []
				for current_scan_path in inference_input_paths:
					slices_by_scan.append(nib.load(current_scan_path).shape[self.inference_slice_dimension]);

				# Find the current scan index in the list of all scans.
				current_scan_index_in_path = np.where(np.cumsum(slices_by_scan) > batch)[0][0]

				# Find the current slice index in the current scan.
				if current_scan_index_in_path == 0:
					current_slice_index_in_scan = batch
				else:
					current_slice_index_in_scan = batch - np.cumsum(slices_by_scan)[current_scan_index_in_path - 1]

				# Find the corresponding image filename to name the generated nifti file.
				corresponding_pre_scan_path = inference_input_paths[current_scan_index_in_path]
				filename_with_extension = corresponding_pre_scan_path.split('/')[-1]

				filename = filename_with_extension[self.inference_filename_start_end[0]:self.inference_filename_start_end[1]]

				# Find the desired dimension.
				corresponding_pre_nifti = nib.load(corresponding_pre_scan_path)
				if self.inference_slice_dimension == 0:
					desired_length = corresponding_pre_nifti.shape[1]; desired_width = corresponding_pre_nifti.shape[2]
				elif self.inference_slice_dimension == 1:
					desired_length = corresponding_pre_nifti.shape[0]; desired_width = corresponding_pre_nifti.shape[2]
				elif self.inference_slice_dimension == 2:
					desired_length = corresponding_pre_nifti.shape[0]; desired_width = corresponding_pre_nifti.shape[1]

				# Initialize an empty matrix to store the current prediction scan if we would like to save the scans in nifti format.
				# Also find the affine and header information from the corresponding pre-gado nifti file.
				if (save_prediction_nifti == True) and (current_slice_index_in_scan == 0):
					corresponding_pre_nifti = nib.load(corresponding_pre_scan_path)
					current_scan_affine = corresponding_pre_nifti.affine
					current_scan_header = corresponding_pre_nifti.header
					if self.inference_slice_dimension == 0:
						current_prediction_scan = np.float32(np.zeros((slices_by_scan[current_scan_index_in_path], desired_length, desired_width)))
					elif self.inference_slice_dimension == 1:
						current_prediction_scan = np.float32(np.zeros((desired_length, slices_by_scan[current_scan_index_in_path], desired_width)))
					elif self.inference_slice_dimension == 2:
						current_prediction_scan = np.float32(np.zeros((desired_length, desired_width, slices_by_scan[current_scan_index_in_path])))
					del corresponding_pre_nifti

				pre_image = pre_image.to(self.device)

				if self.img_ch == 1:
					pre_image = pre_image[:, np.newaxis, :, :]
				else:
					pre_image = pre_image.transpose(1, 3); pre_image = pre_image.transpose(2, 3)

				# During the inference phase, we don't conduct any quantitative evaluation.
				Prediction = self.unet(pre_image)
				np_image = np.squeeze(Prediction.cpu().detach().numpy())

				# Crop the resulting image.
				if np_image.shape != (desired_length, desired_width):
					length_difference = desired_length - np_image.shape[0]; width_difference = desired_width - np_image.shape[1]
					if length_difference > 0:
						pad_top = int(np.round(length_difference/2)); pad_bottom = length_difference - pad_top
						np_image = np.pad(np_image, ((pad_top, pad_bottom), (0, 0)), 'edge')
					elif length_difference < 0:
						crop_top = int(np.round(-length_difference/2)); crop_bottom = -length_difference - crop_top 
						np_image = np_image[crop_top:-crop_bottom, :]
					if width_difference > 0:
						pad_left = int(np.round(width_difference/2)); pad_right = width_difference - pad_left
						np_image = np.pad(np_image, ((0, 0), (pad_left, pad_right)), 'edge')
					elif width_difference < 0:
						crop_left = int(np.round(-width_difference/2)); crop_right = -width_difference - crop_left
						np_image = np_image[:, crop_left:-crop_right]
				assert np_image.shape == (desired_length, desired_width)

				# Add the prediction images to the prediction scan if appropriate if we would like to have nifti output.
				if save_prediction_nifti == True:
					if self.inference_slice_dimension == 0:
						current_prediction_scan[current_slice_index_in_scan, :, :] = np_image
					elif self.inference_slice_dimension == 1:
						current_prediction_scan[:, current_slice_index_in_scan, :] = np_image
					elif self.inference_slice_dimension == 2:
						current_prediction_scan[:, :, current_slice_index_in_scan] = np_image

					# Also, save the current prediction scan once the scan is completely filled.
					if current_slice_index_in_scan == slices_by_scan[current_scan_index_in_path] - 1:
						current_prediction_scan_nifti = nib.Nifti1Image(current_prediction_scan, current_scan_affine, current_scan_header)
						nib.save(current_prediction_scan_nifti, self.current_inference_output_path + filename + '_GBCApredicted.nii.gz')

					del pre_image, Prediction
					# Empty cache to free up memory at the end of each batch.
					torch.cuda.empty_cache()