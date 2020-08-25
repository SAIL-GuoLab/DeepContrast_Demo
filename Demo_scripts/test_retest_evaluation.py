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

print('Hello. Here we are performing the quantitative evalution for the test-retest reliability dataset.')
print('\n\nWe will compute the following things.')
print('\nFirst, we will compute the four metrics (PSNR, P.R, S.R, SSIM) between the corresponding baseline and followup ground truth GBCA-uptake maps.')
print('Then, we will compute the four metrics (PSNR, P.R, S.R, SSIM) between the corresponding baseline and followup deep-learning GBCA-predicted maps.')
print('Lastly, we will compute the four metrics (PSNR, P.R, S.R, SSIM) between the corresponding GBCA-uptake and GBCA-predicted maps. (Not reported in our manuscript.)')

print('\nThe first demonstrates the clinical-level test-retest reliability of the GBCA contrast enhancement.')
print('The second demonstrates the test-retest reliability of the proposed deep-learning approach.')
print('The third confirms that the deep-learning predictions are similiar to the respective ground truth.')
print('\n*Kind note 1: The baseline and followup scans of the same person are registered together using affine deformation.', \
    'Therefore we find it acceptable to share the same brain mask and tissue label map for the same subject.')
print('\n*Kind note 2: We will separately report the results for whole brain, white matter, gray matter, and CSF.')

print('\nGetting ready...')

def compute_PSNR(Prediction_vector, Target_vector):
    MSE = ((Prediction_vector - Target_vector)**2).sum() / Target_vector.size
    MaxI = np.max(Target_vector) - np.min(Target_vector)
    PSNR = 10 * np.log10(MaxI**2 / MSE)
    return PSNR

data_folder = '../Test_retest_data_complete/'
GBCAuptake_folder = data_folder + 'NatureBME_share_GBCAuptake/'
GBCApredicted_folder = data_folder + 'NatureBME_share_GBCApredicted/'
brain_mask_folder = data_folder + 'NatureBME_share_brainMask/'
tissue_label_folder = data_folder + 'NatureBME_share_tissueLabel/'

GBCAuptake_baseline_scans = list(np.sort(glob(GBCAuptake_folder + '*baseline*.nii.gz')))
GBCAuptake_followup_scans = list(np.sort(glob(GBCAuptake_folder + '*followup*.nii.gz')))
GBCApredicted_baseline_scans = list(np.sort(glob(GBCApredicted_folder + '*baseline*.nii.gz')))
GBCApredicted_followup_scans = list(np.sort(glob(GBCApredicted_folder + '*followup*.nii.gz')))
brain_mask_scans = list(np.sort(glob(brain_mask_folder + '*.nii.gz')))
tissue_label_scans = list(np.sort(glob(tissue_label_folder + '*.nii.gz')))

print('...Scans found.')
print('# of GBCA-uptake baseline scans: ', len(GBCAuptake_baseline_scans), \
      '\n# of GBCA-uptake followup scans: ', len(GBCAuptake_followup_scans), \
      '\n# of GBCA-predicted baseline scans: ', len(GBCApredicted_baseline_scans), \
      '\n# of GBCA-predicted followup scans: ', len(GBCApredicted_followup_scans), \
      '\n# of brain mask scans: ', len(brain_mask_scans), \
      '\n# of tissule label scans: ', len(tissue_label_scans))

print('Computing the four metrics over the whole brain.')
GBCAuptake_PSNR_WB, GBCAuptake_PR_WB, GBCAuptake_SR_WB, GBCAuptake_SSIM_WB = [], [], [], []
GBCApredicted_PSNR_WB, GBCApredicted_PR_WB, GBCApredicted_SR_WB, GBCApredicted_SSIM_WB = [], [], [], []
uptake_predicted_PSNR_WB, uptake_predicted_PR_WB, uptake_predicted_SR_WB, uptake_predicted_SSIM_WB = [], [], [], []

for subject_index in tqdm(range(len(GBCAuptake_baseline_scans))):
    brain_mask_scan = np.int16(nib.load(brain_mask_scans[subject_index]).get_fdata())    
    bbox_max = np.max(np.where(brain_mask_scan == 1), axis = 1)
    bbox_min = np.min(np.where(brain_mask_scan == 1), axis = 1)
    
    GBCAuptake_scan_baseline = np.float32(nib.load(GBCAuptake_baseline_scans[subject_index]).get_fdata())
    GBCApredicted_scan_baseline = np.float32(nib.load(GBCApredicted_baseline_scans[subject_index]).get_fdata())
    GBCAuptake_scan_followup = np.float32(nib.load(GBCAuptake_followup_scans[subject_index]).get_fdata())
    GBCApredicted_scan_followup = np.float32(nib.load(GBCApredicted_followup_scans[subject_index]).get_fdata())

    GBCAuptake_PR_WB.append(pearsonr(GBCAuptake_scan_baseline[brain_mask_scan == 1], GBCAuptake_scan_followup[brain_mask_scan == 1])[0])
    GBCAuptake_SR_WB.append(spearmanr(GBCAuptake_scan_baseline[brain_mask_scan == 1], GBCAuptake_scan_followup[brain_mask_scan == 1])[0])
    GBCAuptake_PSNR_WB.append(compute_PSNR(GBCAuptake_scan_baseline[brain_mask_scan == 1], GBCAuptake_scan_followup[brain_mask_scan == 1]))
    GBCAuptake_SSIM_WB.append(ssim((GBCAuptake_scan_baseline * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCAuptake_scan_followup * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    
    GBCApredicted_PR_WB.append(pearsonr(GBCApredicted_scan_baseline[brain_mask_scan == 1], GBCApredicted_scan_followup[brain_mask_scan == 1])[0])
    GBCApredicted_SR_WB.append(spearmanr(GBCApredicted_scan_baseline[brain_mask_scan == 1], GBCApredicted_scan_followup[brain_mask_scan == 1])[0])
    GBCApredicted_PSNR_WB.append(compute_PSNR(GBCApredicted_scan_baseline[brain_mask_scan == 1], GBCApredicted_scan_followup[brain_mask_scan == 1]))
    GBCApredicted_SSIM_WB.append(ssim((GBCApredicted_scan_baseline * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))

    uptake_predicted_PR_WB.append(pearsonr(GBCAuptake_scan_baseline[brain_mask_scan == 1], GBCApredicted_scan_baseline[brain_mask_scan == 1])[0])
    uptake_predicted_SR_WB.append(spearmanr(GBCAuptake_scan_baseline[brain_mask_scan == 1], GBCApredicted_scan_baseline[brain_mask_scan == 1])[0])
    uptake_predicted_PSNR_WB.append(compute_PSNR(GBCAuptake_scan_baseline[brain_mask_scan == 1], GBCApredicted_scan_baseline[brain_mask_scan == 1]))
    uptake_predicted_SSIM_WB.append(ssim((GBCAuptake_scan_baseline * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_baseline * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    uptake_predicted_PR_WB.append(pearsonr(GBCAuptake_scan_followup[brain_mask_scan == 1], GBCApredicted_scan_followup[brain_mask_scan == 1])[0])
    uptake_predicted_SR_WB.append(spearmanr(GBCAuptake_scan_followup[brain_mask_scan == 1], GBCApredicted_scan_followup[brain_mask_scan == 1])[0])
    uptake_predicted_PSNR_WB.append(compute_PSNR(GBCAuptake_scan_followup[brain_mask_scan == 1], GBCApredicted_scan_followup[brain_mask_scan == 1]))
    uptake_predicted_SSIM_WB.append(ssim((GBCAuptake_scan_followup * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * brain_mask_scan)[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))


print('Computing the four metrics over the white matter.')
GBCAuptake_PSNR_WM, GBCAuptake_PR_WM, GBCAuptake_SR_WM, GBCAuptake_SSIM_WM = [], [], [], []
GBCApredicted_PSNR_WM, GBCApredicted_PR_WM, GBCApredicted_SR_WM, GBCApredicted_SSIM_WM = [], [], [], []
uptake_predicted_PSNR_WM, uptake_predicted_PR_WM, uptake_predicted_SR_WM, uptake_predicted_SSIM_WM = [], [], [], []

for subject_index in tqdm(range(len(GBCAuptake_baseline_scans))):
    tissue_label_scan = np.int16(nib.load(tissue_label_scans[subject_index]).get_fdata())    
    bbox_max = np.max(np.where(tissue_label_scan == 3), axis = 1)
    bbox_min = np.min(np.where(tissue_label_scan == 3), axis = 1)
    
    GBCAuptake_scan_baseline = np.float32(nib.load(GBCAuptake_baseline_scans[subject_index]).get_fdata())
    GBCApredicted_scan_baseline = np.float32(nib.load(GBCApredicted_baseline_scans[subject_index]).get_fdata())
    GBCAuptake_scan_followup = np.float32(nib.load(GBCAuptake_followup_scans[subject_index]).get_fdata())
    GBCApredicted_scan_followup = np.float32(nib.load(GBCApredicted_followup_scans[subject_index]).get_fdata())

    GBCAuptake_PR_WM.append(pearsonr(GBCAuptake_scan_baseline[tissue_label_scan == 3], GBCAuptake_scan_followup[tissue_label_scan == 3])[0])
    GBCAuptake_SR_WM.append(spearmanr(GBCAuptake_scan_baseline[tissue_label_scan == 3], GBCAuptake_scan_followup[tissue_label_scan == 3])[0])
    GBCAuptake_PSNR_WM.append(compute_PSNR(GBCAuptake_scan_baseline[tissue_label_scan == 3], GBCAuptake_scan_followup[tissue_label_scan == 3]))
    GBCAuptake_SSIM_WM.append(ssim((GBCAuptake_scan_baseline * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCAuptake_scan_followup * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    
    GBCApredicted_PR_WM.append(pearsonr(GBCApredicted_scan_baseline[tissue_label_scan == 3], GBCApredicted_scan_followup[tissue_label_scan == 3])[0])
    GBCApredicted_SR_WM.append(spearmanr(GBCApredicted_scan_baseline[tissue_label_scan == 3], GBCApredicted_scan_followup[tissue_label_scan == 3])[0])
    GBCApredicted_PSNR_WM.append(compute_PSNR(GBCApredicted_scan_baseline[tissue_label_scan == 3], GBCApredicted_scan_followup[tissue_label_scan == 3]))
    GBCApredicted_SSIM_WM.append(ssim((GBCApredicted_scan_baseline * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))

    uptake_predicted_PR_WM.append(pearsonr(GBCAuptake_scan_baseline[tissue_label_scan == 3], GBCApredicted_scan_baseline[tissue_label_scan == 3])[0])
    uptake_predicted_SR_WM.append(spearmanr(GBCAuptake_scan_baseline[tissue_label_scan == 3], GBCApredicted_scan_baseline[tissue_label_scan == 3])[0])
    uptake_predicted_PSNR_WM.append(compute_PSNR(GBCAuptake_scan_baseline[tissue_label_scan == 3], GBCApredicted_scan_baseline[tissue_label_scan == 3]))
    uptake_predicted_SSIM_WM.append(ssim((GBCAuptake_scan_baseline * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_baseline * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    uptake_predicted_PR_WM.append(pearsonr(GBCAuptake_scan_followup[tissue_label_scan == 3], GBCApredicted_scan_followup[tissue_label_scan == 3])[0])
    uptake_predicted_SR_WM.append(spearmanr(GBCAuptake_scan_followup[tissue_label_scan == 3], GBCApredicted_scan_followup[tissue_label_scan == 3])[0])
    uptake_predicted_PSNR_WM.append(compute_PSNR(GBCAuptake_scan_followup[tissue_label_scan == 3], GBCApredicted_scan_followup[tissue_label_scan == 3]))
    uptake_predicted_SSIM_WM.append(ssim((GBCAuptake_scan_followup * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * (tissue_label_scan == 3))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))


print('Computing the four metrics over the gray matter.')
GBCAuptake_PSNR_GM, GBCAuptake_PR_GM, GBCAuptake_SR_GM, GBCAuptake_SSIM_GM = [], [], [], []
GBCApredicted_PSNR_GM, GBCApredicted_PR_GM, GBCApredicted_SR_GM, GBCApredicted_SSIM_GM = [], [], [], []
uptake_predicted_PSNR_GM, uptake_predicted_PR_GM, uptake_predicted_SR_GM, uptake_predicted_SSIM_GM = [], [], [], []

for subject_index in tqdm(range(len(GBCAuptake_baseline_scans))):
    tissue_label_scan = np.int16(nib.load(tissue_label_scans[subject_index]).get_fdata())    
    bbox_max = np.max(np.where(tissue_label_scan == 2), axis = 1)
    bbox_min = np.min(np.where(tissue_label_scan == 2), axis = 1)
    
    GBCAuptake_scan_baseline = np.float32(nib.load(GBCAuptake_baseline_scans[subject_index]).get_fdata())
    GBCApredicted_scan_baseline = np.float32(nib.load(GBCApredicted_baseline_scans[subject_index]).get_fdata())
    GBCAuptake_scan_followup = np.float32(nib.load(GBCAuptake_followup_scans[subject_index]).get_fdata())
    GBCApredicted_scan_followup = np.float32(nib.load(GBCApredicted_followup_scans[subject_index]).get_fdata())

    GBCAuptake_PR_GM.append(pearsonr(GBCAuptake_scan_baseline[tissue_label_scan == 2], GBCAuptake_scan_followup[tissue_label_scan == 2])[0])
    GBCAuptake_SR_GM.append(spearmanr(GBCAuptake_scan_baseline[tissue_label_scan == 2], GBCAuptake_scan_followup[tissue_label_scan == 2])[0])
    GBCAuptake_PSNR_GM.append(compute_PSNR(GBCAuptake_scan_baseline[tissue_label_scan == 2], GBCAuptake_scan_followup[tissue_label_scan == 2]))
    GBCAuptake_SSIM_GM.append(ssim((GBCAuptake_scan_baseline * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCAuptake_scan_followup * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    
    GBCApredicted_PR_GM.append(pearsonr(GBCApredicted_scan_baseline[tissue_label_scan == 2], GBCApredicted_scan_followup[tissue_label_scan == 2])[0])
    GBCApredicted_SR_GM.append(spearmanr(GBCApredicted_scan_baseline[tissue_label_scan == 2], GBCApredicted_scan_followup[tissue_label_scan == 2])[0])
    GBCApredicted_PSNR_GM.append(compute_PSNR(GBCApredicted_scan_baseline[tissue_label_scan == 2], GBCApredicted_scan_followup[tissue_label_scan == 2]))
    GBCApredicted_SSIM_GM.append(ssim((GBCApredicted_scan_baseline * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))

    uptake_predicted_PR_GM.append(pearsonr(GBCAuptake_scan_baseline[tissue_label_scan == 2], GBCApredicted_scan_baseline[tissue_label_scan == 2])[0])
    uptake_predicted_SR_GM.append(spearmanr(GBCAuptake_scan_baseline[tissue_label_scan == 2], GBCApredicted_scan_baseline[tissue_label_scan == 2])[0])
    uptake_predicted_PSNR_GM.append(compute_PSNR(GBCAuptake_scan_baseline[tissue_label_scan == 2], GBCApredicted_scan_baseline[tissue_label_scan == 2]))
    uptake_predicted_SSIM_GM.append(ssim((GBCAuptake_scan_baseline * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_baseline * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    uptake_predicted_PR_GM.append(pearsonr(GBCAuptake_scan_followup[tissue_label_scan == 2], GBCApredicted_scan_followup[tissue_label_scan == 2])[0])
    uptake_predicted_SR_GM.append(spearmanr(GBCAuptake_scan_followup[tissue_label_scan == 2], GBCApredicted_scan_followup[tissue_label_scan == 2])[0])
    uptake_predicted_PSNR_GM.append(compute_PSNR(GBCAuptake_scan_followup[tissue_label_scan == 2], GBCApredicted_scan_followup[tissue_label_scan == 2]))
    uptake_predicted_SSIM_GM.append(ssim((GBCAuptake_scan_followup * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * (tissue_label_scan == 2))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))


print('Computing the four metrics over the CSF.')
GBCAuptake_PSNR_CSF, GBCAuptake_PR_CSF, GBCAuptake_SR_CSF, GBCAuptake_SSIM_CSF = [], [], [], []
GBCApredicted_PSNR_CSF, GBCApredicted_PR_CSF, GBCApredicted_SR_CSF, GBCApredicted_SSIM_CSF = [], [], [], []
uptake_predicted_PSNR_CSF, uptake_predicted_PR_CSF, uptake_predicted_SR_CSF, uptake_predicted_SSIM_CSF = [], [], [], []

for subject_index in tqdm(range(len(GBCAuptake_baseline_scans))):
    tissue_label_scan = np.int16(nib.load(tissue_label_scans[subject_index]).get_fdata())    
    bbox_max = np.max(np.where(tissue_label_scan == 1), axis = 1)
    bbox_min = np.min(np.where(tissue_label_scan == 1), axis = 1)
    
    GBCAuptake_scan_baseline = np.float32(nib.load(GBCAuptake_baseline_scans[subject_index]).get_fdata())
    GBCApredicted_scan_baseline = np.float32(nib.load(GBCApredicted_baseline_scans[subject_index]).get_fdata())
    GBCAuptake_scan_followup = np.float32(nib.load(GBCAuptake_followup_scans[subject_index]).get_fdata())
    GBCApredicted_scan_followup = np.float32(nib.load(GBCApredicted_followup_scans[subject_index]).get_fdata())

    GBCAuptake_PR_CSF.append(pearsonr(GBCAuptake_scan_baseline[tissue_label_scan == 1], GBCAuptake_scan_followup[tissue_label_scan == 1])[0])
    GBCAuptake_SR_CSF.append(spearmanr(GBCAuptake_scan_baseline[tissue_label_scan == 1], GBCAuptake_scan_followup[tissue_label_scan == 1])[0])
    GBCAuptake_PSNR_CSF.append(compute_PSNR(GBCAuptake_scan_baseline[tissue_label_scan == 1], GBCAuptake_scan_followup[tissue_label_scan == 1]))
    GBCAuptake_SSIM_CSF.append(ssim((GBCAuptake_scan_baseline * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCAuptake_scan_followup * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    
    GBCApredicted_PR_CSF.append(pearsonr(GBCApredicted_scan_baseline[tissue_label_scan == 1], GBCApredicted_scan_followup[tissue_label_scan == 1])[0])
    GBCApredicted_SR_CSF.append(spearmanr(GBCApredicted_scan_baseline[tissue_label_scan == 1], GBCApredicted_scan_followup[tissue_label_scan == 1])[0])
    GBCApredicted_PSNR_CSF.append(compute_PSNR(GBCApredicted_scan_baseline[tissue_label_scan == 1], GBCApredicted_scan_followup[tissue_label_scan == 1]))
    GBCApredicted_SSIM_CSF.append(ssim((GBCApredicted_scan_baseline * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))

    uptake_predicted_PR_CSF.append(pearsonr(GBCAuptake_scan_baseline[tissue_label_scan == 1], GBCApredicted_scan_baseline[tissue_label_scan == 1])[0])
    uptake_predicted_SR_CSF.append(spearmanr(GBCAuptake_scan_baseline[tissue_label_scan == 1], GBCApredicted_scan_baseline[tissue_label_scan == 1])[0])
    uptake_predicted_PSNR_CSF.append(compute_PSNR(GBCAuptake_scan_baseline[tissue_label_scan == 1], GBCApredicted_scan_baseline[tissue_label_scan == 1]))
    uptake_predicted_SSIM_CSF.append(ssim((GBCAuptake_scan_baseline * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_baseline * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))
    uptake_predicted_PR_CSF.append(pearsonr(GBCAuptake_scan_followup[tissue_label_scan == 1], GBCApredicted_scan_followup[tissue_label_scan == 1])[0])
    uptake_predicted_SR_CSF.append(spearmanr(GBCAuptake_scan_followup[tissue_label_scan == 1], GBCApredicted_scan_followup[tissue_label_scan == 1])[0])
    uptake_predicted_PSNR_CSF.append(compute_PSNR(GBCAuptake_scan_followup[tissue_label_scan == 1], GBCApredicted_scan_followup[tissue_label_scan == 1]))
    uptake_predicted_SSIM_CSF.append(ssim((GBCAuptake_scan_followup * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]], \
                    (GBCApredicted_scan_followup * (tissue_label_scan == 1))[bbox_min[0]: bbox_max[0], bbox_min[1]: bbox_max[1], bbox_min[2]: bbox_max[2]]))


# Report the numbers.
print('\n\n*****The four-metric report for test-retest reliability of ground truth GBCA-uptake over the whole brain.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCAuptake_PSNR_WB), sem(GBCAuptake_PSNR_WB)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_PR_WB), sem(GBCAuptake_PR_WB)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SR_WB), sem(GBCAuptake_SR_WB)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SSIM_WB), sem(GBCAuptake_SSIM_WB)))

print('*****The four-metric report for test-retest reliability of GBCA-predicted over the whole brain.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCApredicted_PSNR_WB), sem(GBCApredicted_PSNR_WB)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_PR_WB), sem(GBCApredicted_PR_WB)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SR_WB), sem(GBCApredicted_SR_WB)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SSIM_WB), sem(GBCApredicted_SSIM_WB)))

print('*****The four-metric report for uptake-predicted concordance over the whole brain. (Not reported in our manuscript). *****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(uptake_predicted_PSNR_WB), sem(uptake_predicted_PSNR_WB)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_PR_WB), sem(uptake_predicted_PR_WB)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SR_WB), sem(uptake_predicted_SR_WB)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SSIM_WB), sem(uptake_predicted_SSIM_WB)))

print('\n\n*****The four-metric report for test-retest reliability of ground truth GBCA-uptake over the white matter.*****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCAuptake_PSNR_WM), sem(GBCAuptake_PSNR_WM)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_PR_WM), sem(GBCAuptake_PR_WM)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SR_WM), sem(GBCAuptake_SR_WM)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SSIM_WM), sem(GBCAuptake_SSIM_WM)))

print('*****The four-metric report for test-retest reliability of GBCA-predicted over the white matter.*****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCApredicted_PSNR_WM), sem(GBCApredicted_PSNR_WM)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_PR_WM), sem(GBCApredicted_PR_WM)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SR_WM), sem(GBCApredicted_SR_WM)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SSIM_WM), sem(GBCApredicted_SSIM_WM)))

print('*****The four-metric report for uptake-predicted concordance over the white matter. (Not reported in our manuscript). *****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(uptake_predicted_PSNR_WM), sem(uptake_predicted_PSNR_WM)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_PR_WM), sem(uptake_predicted_PR_WM)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SR_WM), sem(uptake_predicted_SR_WM)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SSIM_WM), sem(uptake_predicted_SSIM_WM)))

print('\n\n*****The four-metric report for test-retest reliability of ground truth GBCA-uptake over the gray matter.*****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCAuptake_PSNR_GM), sem(GBCAuptake_PSNR_GM)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_PR_GM), sem(GBCAuptake_PR_GM)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SR_GM), sem(GBCAuptake_SR_GM)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SSIM_GM), sem(GBCAuptake_SSIM_GM)))

print('*****The four-metric report for test-retest reliability of GBCA-predicted over the gray matter.*****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCApredicted_PSNR_GM), sem(GBCApredicted_PSNR_GM)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_PR_GM), sem(GBCApredicted_PR_GM)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SR_GM), sem(GBCApredicted_SR_GM)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SSIM_GM), sem(GBCApredicted_SSIM_GM)))

print('*****The four-metric report for uptake-predicted concordance over the gray matter. (Not reported in our manuscript). *****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(uptake_predicted_PSNR_GM), sem(uptake_predicted_PSNR_GM)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_PR_GM), sem(uptake_predicted_PR_GM)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SR_GM), sem(uptake_predicted_SR_GM)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SSIM_GM), sem(uptake_predicted_SSIM_GM)))

print('\n\n*****The four-metric report for test-retest reliability of ground truth GBCA-uptake over the CSF.*****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCAuptake_PSNR_CSF), sem(GBCAuptake_PSNR_CSF)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_PR_CSF), sem(GBCAuptake_PR_CSF)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SR_CSF), sem(GBCAuptake_SR_CSF)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCAuptake_SSIM_CSF), sem(GBCAuptake_SSIM_CSF)))

print('*****The four-metric report for test-retest reliability of GBCA-predicted over the CSF.*****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(GBCApredicted_PSNR_CSF), sem(GBCApredicted_PSNR_CSF)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_PR_CSF), sem(GBCApredicted_PR_CSF)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SR_CSF), sem(GBCApredicted_SR_CSF)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(GBCApredicted_SSIM_CSF), sem(GBCApredicted_SSIM_CSF)))

print('*****The four-metric report for uptake-predicted concordance over the CSF. (Not reported in our manuscript). *****')
print('*****Please note that SSIM is less informative when the volume is partially hollow as in this case.*****')
print('PSNR: %.2f \xb1 %.2f' % (np.mean(uptake_predicted_PSNR_CSF), sem(uptake_predicted_PSNR_CSF)))
print('P.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_PR_CSF), sem(uptake_predicted_PR_CSF)))
print('S.R: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SR_CSF), sem(uptake_predicted_SR_CSF)))
print('SSIM: %.3f \xb1 %.3f' % (np.mean(uptake_predicted_SSIM_CSF), sem(uptake_predicted_SSIM_CSF)))


# Plot the results in bar plot.
GBCAuptake_PSNR_WB_means, GBCApredicted_PSNR_WB_means = np.mean(GBCAuptake_PSNR_WB), np.mean(GBCApredicted_PSNR_WB)
GBCAuptake_PSNR_WB_sem, GBCApredicted_PSNR_WB_sem = sem(GBCAuptake_PSNR_WB), sem(GBCApredicted_PSNR_WB)

GBCAuptake_ThreeMetrics_WB_means = [np.mean(GBCAuptake_PR_WB), np.mean(GBCAuptake_SR_WB), np.mean(GBCAuptake_SSIM_WB)]
GBCApredicted_ThreeMetrics_WB_means = [np.mean(GBCApredicted_PR_WB), np.mean(GBCApredicted_SR_WB), np.mean(GBCApredicted_SSIM_WB)]
GBCAuptake_ThreeMetrics_WB_sem = [sem(GBCAuptake_PR_WB), sem(GBCAuptake_SR_WB), sem(GBCAuptake_SSIM_WB)]
GBCApredicted_ThreeMetrics_WB_sem = [sem(GBCApredicted_PR_WB), sem(GBCApredicted_SR_WB), sem(GBCApredicted_SSIM_WB)]

plt.rcParams['figure.figsize'] = [12, 3]
grid = plt.GridSpec(1, 5)
ax1 = plt.subplot(grid[0, 0])
plot_index = np.arange(1)
plot_width = 0.4
rects1 = plt.bar(plot_index - plot_width / 2, GBCAuptake_PSNR_WB_means, plot_width, yerr = GBCAuptake_PSNR_WB_sem, capsize = 8, label = 'GBCAuptake', color = 'gray')
rects1 = plt.bar(plot_index + plot_width / 2, GBCApredicted_PSNR_WB_means, plot_width, yerr = GBCApredicted_PSNR_WB_sem, capsize = 8, label = 'Estimated Gd-Uptake', color = 'firebrick')
plt.xticks([])
plt.yticks([0, 10, 20, 35])
ax1.margins(0.25, 0.1)
ax1.set_ylim([0, 36])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_tick_params(labelsize = 30)

ax2 = plt.subplot(grid[0, 2:])
plot_index = np.arange(3)
rects1 = plt.bar(plot_index - plot_width / 2, GBCAuptake_ThreeMetrics_WB_means, plot_width, yerr = GBCAuptake_ThreeMetrics_WB_sem, capsize = 8, label = 'GBCAuptake', color = 'gray')
rects1 = plt.bar(plot_index + plot_width / 2, GBCApredicted_ThreeMetrics_WB_means, plot_width, yerr = GBCApredicted_ThreeMetrics_WB_sem, capsize = 8, label = 'Estimated Gd-Uptake', color = 'firebrick')
plt.xticks([])
plt.yticks([0, 0.5, 1])
ax2.margins(0.05, 0.1)
ax2.set_ylim([0, 1])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_tick_params(labelsize = 30)

plt.tight_layout()
plt.savefig('Test_retest_sample_bar_plot_for_whole_brain.png', facecolor = 'white')