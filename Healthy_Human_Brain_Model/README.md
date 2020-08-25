## Healthy_Human_Brain_Model
This contains the testing code (or more accurately, the inference code) for the Healthy Human Brain Model reported in our manuscript. The model architecture and the implementation of the inference procedure are included in "./deep_learning_model/". The actual scripts to run are located in "../Demo_scripts/" in the parent folder of the this folder.

### deep_learning_model
This folder contains 3 files, "data_loader.py", "network.py" and "solver.py".

The first is the data loader to appropriately handle the NifTI data (read and organize the non-contrast scan for the deep learning model).

The second is the network architecture (where in fact only the Residual Attention U-Net is used in our case).

The third is the main script that define how inference (using the trained network on a new standalone dataset unused for training and not necessarily has ground truth) is performed. In our case we wrote the inference code such that the predicted scans will be saved in NifTI format. 

### saved_model_weights
This folder is supposed to contain one file, "ResAttU_Net-SGD-0.1000-CVPR_Adaptive_loss-4-epoch18.pkl". However, that file is too large for GitHub and can only be accessed through [Google Drive](https://drive.google.com/drive/folders/1l5GU6E0iCHbs24ZNzN6uIQgbQud1DZ3e?usp=sharing) as described previously. Once you download the files, please make sure you place that file in the correct location (inside "./saved_model_weights/") once you downloaded it.
