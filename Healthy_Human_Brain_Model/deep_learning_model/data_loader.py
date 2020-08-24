import os
import random
from glob import glob
from random import shuffle
import numpy as np
import nibabel as nib
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from sklearn.feature_extraction import image as sklearn_image
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
np.random.seed(2019)
random.seed(2019)

class NiftiDataset_inference(data.Dataset):
	def __init__(self, inference_input_folder, desired_dimension, inference_slice_dimension):
		"""Initializes nifti file paths and preprocessing module."""
        
		# Define the directories where the pre-gado nifti scans are stored.
		self.inference_input_folder = inference_input_folder

		# Define the target image dimension.
		self.desired_dimension = desired_dimension

		# Define the target slice dimension. I.e., which dimension of the scan is the slice dimension.
		self.inference_slice_dimension = inference_slice_dimension

		# Grab all the files in these directories. 
		self.inference_input_paths = list(np.sort(glob(inference_input_folder + '*.nii.gz')))

		# Report the number of files in the pre-gado directories.
		print('Input inference nifti file count: {}'.format(len(self.inference_input_paths)))

		# Record how many slices each nifti scan contain in a list.
		self.slices_by_scan = []
		for current_scan_path in self.inference_input_paths:
			self.slices_by_scan.append(nib.load(current_scan_path).shape[self.inference_slice_dimension]);
        
	def __getitem__(self, index):
		"""Reads images from a nifti file, preprocesses it and returns."""
		# Here we need to provide the correct slice according to the index.
		# (e.x., index == 100 means we need to provide the 101st slice in the folder).

		# Every nifti file not necessarily contain the same number of slices.
		# We would need to check and assign the current slice index to the pre-gado nifti file where it comes from.
        
		# Find the current scan index in the list of all scan paths.
		current_scan_index_in_path = np.where(np.cumsum(self.slices_by_scan) > index)[0][0]
		# Find the pre-gado file path corresponding to the current scan slice.
		inference_input_path = self.inference_input_paths[current_scan_index_in_path]

		# Load the pre-gado nifti scans.
		pre_nifti_scan = nib.load(inference_input_path).get_fdata()

		# Find the current slice index in the current scan.
		if current_scan_index_in_path == 0:
			current_slice_index_in_scan = index
		else:
			current_slice_index_in_scan = index - np.cumsum(self.slices_by_scan)[current_scan_index_in_path - 1]

		# Extract the current pre-gado images (aka. slices) from the nifti scans.
		if self.inference_slice_dimension == 0:
			pre_nifti_image = np.array(pre_nifti_scan[current_slice_index_in_scan, :, :]).astype(np.float32)
		elif self.inference_slice_dimension == 1:
			pre_nifti_image = np.array(pre_nifti_scan[:, current_slice_index_in_scan, :]).astype(np.float32)
		elif self.inference_slice_dimension == 2:
			pre_nifti_image = np.array(pre_nifti_scan[:, :, current_slice_index_in_scan]).astype(np.float32)
		
		# Pad the image to the correct dimension.
		if pre_nifti_image.shape != self.desired_dimension:
			length_difference = self.desired_dimension[0] - pre_nifti_image.shape[0]; width_difference = self.desired_dimension[1] - pre_nifti_image.shape[1]
			if length_difference > 0:
				pad_top = int(np.round(length_difference/2)); pad_bottom = length_difference - pad_top
				pre_nifti_image = np.pad(pre_nifti_image, ((pad_top, pad_bottom), (0, 0)), 'edge')
			elif length_difference < 0:
				crop_top = int(np.round(-length_difference/2)); crop_bottom = -length_difference - crop_top 
				pre_nifti_image = pre_nifti_image[crop_top:-crop_bottom, :]
			if width_difference > 0:
				pad_left = int(np.round(width_difference/2)); pad_right = width_difference - pad_left
			elif width_difference < 0:
				crop_left = int(np.round(-width_difference/2)); crop_right = -width_difference - crop_left 
				pre_nifti_image = pre_nifti_image[:, crop_left:-crop_right]
			assert pre_nifti_image.shape == self.desired_dimension

		# Extremely important!!!! Here we normalize the pre-gado and post-gado images based on the range of the pre-gado scan,
		pre_nifti_scan_max = np.asarray(pre_nifti_scan).max().astype(np.float32)
		pre_nifti_image_normalized = pre_nifti_image/pre_nifti_scan_max

		return pre_nifti_image_normalized

	def __len__(self):
		"""Returns the total number of nifti images."""
		return sum(self.slices_by_scan)

def get_loader_inference(inference_input_folder, batch_size, inference_slice_dimension, num_workers = 1, desired_dimension = (352, 352), shuffle = False):
	"""Builds and returns Dataloader."""

	dataset = NiftiDataset_inference(inference_input_folder = inference_input_folder, desired_dimension = desired_dimension, inference_slice_dimension = inference_slice_dimension)
	data_loader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
	return data_loader