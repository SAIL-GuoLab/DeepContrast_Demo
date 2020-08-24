# DeepContrast_Demo
This is the repository that accompanies the submission **Deep Learning Substitutes Gadolinium in Detecting Functional and Structural Brain Lesions with MRI**, *under review at Nature Biomedical Engineering*.

## Overview of this repository
```
DeepContrast_Demo
    ├── Test_retest_data_complete
    |   ├── (*) NatureBME_share_nonContrast
    |   ├── (*) NatureBME_share_GBCAuptake
    |   ├── (*) NatureBME_share_GBCApredicted
    |   ├── (*) NatureBME_share_brainMask
    |   └── (*) NatureBME_share_tissueLabel
    |
    ├── Healthy_Human_Brain_Model
    |   ├── deep_learning_model
    |   |   ├── data_loader.py
    |   |   ├── network.py
    |   |   └── solver.py
    |   |
    |   └── saved_model_weights
    |       └──ResAttU_Net-SGD-0.1000-CVPR_Adaptive_loss-4-epoch18.pkl
    |
    ├── Demo_scripts
    |   ├── generate_new_predictions.py
    |   ├── verify_old_new_predictions_identical.py
    |   ├── visual_inspection.py
    |   └── test_retest_evaluation.py
    |
    └── Newly_generated_prediction
```
**Test_retest_data_complete** contains the 


## Authors
Chen Liu, Nanyan Zhu, Dipika Sikka, Xinyang Feng, Haoran Sun, Xueqing Liu, Sabrina Gjerswold-Selleck, Hong-Jian Wei, Pavan S. Upadhyayula, Angeliki Mela, Peter D. Canoll, Cheng-Chia Wu, Andrew F. Laine, Jeffrey A. Lieberman, Frank A. Provenzano, Scott A. Small, Jia Guo, for the Alzheimer’s Disease Neuroimaging Initiative.

**Chen Liu and Nanyan Zhu contributed equally to this work and are joint first authors.**

**Correspondance: Jia Guo (jg3400@columbia.edu).**

## Link to Manuscript
To be added.

## Code availability
The trained Healthy Human Brain Model, alongside the test-retest reliability dataset (n = 11, each with two test-retest acquisitions) with both non-contrast scans and ground truth GBCA-uptake maps, is available on GitHub (link to be announced). The scripts that predict GBCA-uptake maps from non-contrast scans, as well as the script performing quantitative evaluations, are included. All code and data (except for those from public datasets) are proprietary and managed by the Columbia Technology Ventures Office of Intellectual Property. The custom training code and large-scale datasets are not publicly available.

## Data availability
The authors declare that all data supporting the results in this study are available from the corresponding author J.G. upon reasonable request, after permission from the Columbia Technology Ventures Office of Intellectual Property.
