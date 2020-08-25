import sys
import os
from shutil import copyfile
from torch.backends import cudnn
import random
import numpy as np
import nibabel as nibabel
from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import rank
from sklearn import linear_model
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import warnings
import nibabel as nib
import csv
import pandas as pd

new_prediction_path = '../Newly_generated_prediction/'
old_prediction_path = '../Test_retest_data_complete/NatureBME_share_GBCApredicted/'

all_new_predictions = list(np.sort(glob(new_prediction_path + '*.nii.gz')))
all_old_predictions = list(np.sort(glob(old_prediction_path + '*.nii.gz')))

print('There are a total of %s new prediction scans and %s old predictions scans' % (len(all_new_predictions), len(all_old_predictions)))
assert len(all_new_predictions) == len(all_old_predictions)

print('Let''s check if they are identical.')

for scan_index in range(len(all_new_predictions)):
    current_new_prediction_filename = all_new_predictions[scan_index]
    current_old_prediction_filename = all_old_predictions[scan_index]
    assert current_new_prediction_filename.split('/')[-1].split('_GBCApredicted')[0] == \
           current_old_prediction_filename.split('/')[-1].split('_GBCApredicted')[0]
    subject_ID = current_new_prediction_filename.split('/')[-1].split('_GBCApredicted')[0]
    print('Checking subject %s' % subject_ID)

    current_new_prediction = np.float32(nib.load(current_new_prediction_filename).get_fdata())
    current_old_prediction = np.float32(nib.load(current_old_prediction_filename).get_fdata())

    total_voxels_different = np.sum(abs(current_new_prediction - current_old_prediction) > 1e-8)
    print('Number of different voxels = %s.' % total_voxels_different)
    if total_voxels_different == 0:
        print('The two scans are identical.')