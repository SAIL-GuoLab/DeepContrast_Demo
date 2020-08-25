import sys
import os
import sys
import os
import random
import numpy as np
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
import warnings
import nibabel as nib
from skimage.measure import compare_ssim as ssim
from scipy.stats import pearsonr, spearmanr, sem
from tqdm import tqdm

print('Hello. Here we are performing some qualitative evalution for the test-retest reliability dataset.')
print('\nFor example we can display some sample scans for visual inspection.')

data_folder = '../Test_retest_data_complete/'
nonContrast_folder = data_folder +  'NatureBME_share_nonContrast/'
GBCAuptake_folder = data_folder + 'NatureBME_share_GBCAuptake/'
GBCApredicted_folder = data_folder + 'NatureBME_share_GBCApredicted/'
brain_mask_folder = data_folder + 'NatureBME_share_brainMask/'

nonContrast_baseline_scans = list(np.sort(glob(nonContrast_folder + '*baseline*.nii.gz')))
nonContrast_followup_scans = list(np.sort(glob(nonContrast_folder + '*followup*.nii.gz')))
GBCAuptake_baseline_scans = list(np.sort(glob(GBCAuptake_folder + '*baseline*.nii.gz')))
GBCAuptake_followup_scans = list(np.sort(glob(GBCAuptake_folder + '*followup*.nii.gz')))
GBCApredicted_baseline_scans = list(np.sort(glob(GBCApredicted_folder + '*baseline*.nii.gz')))
GBCApredicted_followup_scans = list(np.sort(glob(GBCApredicted_folder + '*followup*.nii.gz')))
brain_mask_scans = list(np.sort(glob(brain_mask_folder + '*.nii.gz')))

print('...Scans found.')
print('# of non-contrast baseline scans: ', len(nonContrast_baseline_scans), \
      '\n# of non-contrast followup scans: ', len(nonContrast_followup_scans), \
      '\n# of GBCA-uptake baseline scans: ', len(GBCAuptake_baseline_scans), \
      '\n# of GBCA-uptake followup scans: ', len(GBCAuptake_followup_scans), \
      '\n# of GBCA-predicted baseline scans: ', len(GBCApredicted_baseline_scans), \
      '\n# of GBCA-predicted followup scans: ', len(GBCApredicted_followup_scans), \
      '\n# of brain mask scans: ', len(brain_mask_scans))

sample_index = np.random.choice(len(nonContrast_baseline_scans), 1)[0]
print('Random subject: ', nonContrast_baseline_scans[sample_index].split('/')[-1].split('_baseline')[0])

sample_nonContrast_baseline = np.float32(nib.load(nonContrast_baseline_scans[sample_index]).get_fdata())
sample_nonContrast_followup = np.float32(nib.load(nonContrast_followup_scans[sample_index]).get_fdata())
sample_GBCAuptake_baseline = np.float32(nib.load(GBCAuptake_baseline_scans[sample_index]).get_fdata())
sample_GBCAuptake_followup = np.float32(nib.load(GBCAuptake_followup_scans[sample_index]).get_fdata())
sample_GBCApredicted_baseline = np.float32(nib.load(GBCApredicted_baseline_scans[sample_index]).get_fdata())
sample_GBCApredicted_followup = np.float32(nib.load(GBCApredicted_followup_scans[sample_index]).get_fdata())
sample_brain_mask = np.float32(nib.load(brain_mask_scans[sample_index]).get_fdata())

# Normalize the non-contrast scan to range [0, 1].
sample_nonContrast_baseline = sample_nonContrast_baseline / sample_nonContrast_baseline.max()
sample_nonContrast_followup = sample_nonContrast_followup / sample_nonContrast_followup.max()

# Choose slice for display.
slice_for_display = np.shape(sample_brain_mask)[2] * 48 // 100 # The 45% location along the coronal direction.

plt.rcParams['figure.figsize'] = [10, 8]

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(sample_nonContrast_baseline * sample_brain_mask)[:, :, slice_for_display], clim = [0, 1], cmap = 'gray')
plt.axis('off'); title_object = plt.title('Baseline, non-contrast'); plt.setp(title_object, color = 'w')
plt.subplot(2, 3, 2)
plt.imshow(np.rot90(sample_GBCAuptake_baseline * sample_brain_mask)[:, :, slice_for_display], clim = [0, 0.22], cmap = 'jet')
plt.axis('off'); title_object = plt.title('Baseline, GBCA-uptake'); plt.setp(title_object, color = 'w')
plt.subplot(2, 3, 3)
plt.imshow(np.rot90(sample_GBCApredicted_baseline * sample_brain_mask)[:, :, slice_for_display], clim = [0, 0.22], cmap = 'jet')
plt.axis('off'); title_object = plt.title('Baseline, GBCA-predicted'); plt.setp(title_object, color = 'w')
plt.subplot(2, 3, 4)
plt.imshow(np.rot90(sample_nonContrast_followup * sample_brain_mask)[:, :, slice_for_display], clim = [0, 1], cmap = 'gray')
plt.axis('off'); title_object = plt.title('Followup, non-contrast'); plt.setp(title_object, color = 'w')
plt.subplot(2, 3, 5)
plt.imshow(np.rot90(sample_GBCAuptake_followup * sample_brain_mask)[:, :, slice_for_display], clim = [0, 0.22], cmap = 'jet')
plt.axis('off'); title_object = plt.title('Followup, GBCA-uptake'); plt.setp(title_object, color = 'w')
plt.subplot(2, 3, 6)
plt.imshow(np.rot90(sample_GBCApredicted_followup * sample_brain_mask)[:, :, slice_for_display], clim = [0, 0.22], cmap = 'jet')
plt.axis('off'); title_object = plt.title('Followup, GBCA-predicted'); plt.setp(title_object, color = 'w')

plt.tight_layout()
plt.savefig('Test_retest_sample_visual_inspection.png', facecolor = 'black')