# DeepContrast_Demo
This is the repository that accompanies the submission **Deep Learning Substitutes Gadolinium in Detecting Functional and Structural Brain Lesions with MRI**, *under review at Nature Biomedical Engineering*.

Once missing files and model weights are downloaded from Google Drive and placed at the correct locations, the files and scripts shared in this repository shall be enough for an experienced deep learning researcher to replicate the results we reported in the test-retest reliability study.

Please note that the code was developed on Linux, and it may require some adjustments if you intend to run it on a different operating system. For example, on Linux the path format uses '/' as a separator while on Windows it uses '\\' instead.


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
    ├── Newly_generated_prediction
    |
    └── Environment_setup
        └── DeepContrast.yml
```
### Explanations of the folders:
**Test_retest_data_complete** is supposed to contain all data from the test-retest reliability dataset. However, as GitHub has a strict data upload limit of 100 MB, we have to omit the actual data files from this repository, and instead make these files available upon request on Google Drive. Currently the Chief Editor and the reviewers will have access to the link pointing to the Google Drive folder.

**Healthy_Human_Brain_Model** contains the backbone of the Healthy Human Brain Model (both the architecture and the trained model weights) introduced in our manuscript. Currently only the customized testing code is made available. Again, since the model weights (543 MB) exceeds the file size limit, we have to keep the "saved_model_weights" folder empty and only share that over Google Drive.

**Demo_scripts** contain the four sample scripts to demonstrate the model.

**Newly_generated_prediction** is an empty folder and will be filled with new predictions once "./Demo_scripts/generate_new_predictions.py" is executed.

**Environment_setup** contains the anaconda configuration file "DeepContrast.yml" with which one can quickly configure an environment suitable to run our scripts. If it doesn't work on your machine, you would probably need to manually install the required packages (and anaconda as well if not installed).

*More detailed descriptions can be found in the respective folders.*

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
