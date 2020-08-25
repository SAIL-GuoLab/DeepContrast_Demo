## Demo_scripts

This folder contains the key scripts we organized for demonstration of how to reproduce some results described in our manuscript. We will briefly describe what each of these four scripts does.

Before executing any of these scripts, you would need to download the missing scans and model weights using the Google Drive link you requested (for now we are prioritizing this service to the editors and reviewers), and place the downloaded folders in the correct locations which shall be clearly depicted in the ["Overview of this repository"](https://github.com/SAIL-GuoLab/DeepContrast_Demo#overview-of-this-repository) section. Moreover, you would need to configure an anaconda environment as described [here](https://github.com/SAIL-GuoLab/DeepContrast_Demo#preparing-the-anaconda-environment-to-execute-the-code) and [here](https://github.com/SAIL-GuoLab/DeepContrast_Demo/tree/master/Environment_setup). Lastly, once everything is done and as the last prepratory step, you can activate the environment.

```
conda activate DeepContrast
```

### generate_new_predictions.py

This script runs the trained Healthy Human Brain Model to generate GBCA-predicted maps using the non-contrast scans as the input. Specifically,
1. It follows the processes specified in the solver file (../Healthy_Human_Brain_Model/deep_learning_model/solver.py).
2. It loads the model weights (../saved_model_weights/ResAttU_Net-SGD-0.1000-CVPR_Adaptive_loss-4-epoch18.pkl) into the Residual Attention U-Net implemented in the network file (../Healthy_Human_Brain_Model/deep_learning_model/network.py)
3. It grabs the non-contrast scans (../Test_retest_data_complete/NatureBME_share_nonContrast/EVERYTHING.nii.gz) and loads them into the model using the data loader defined in (../Healthy_Human_Brain_Model/deep_learning_model/data_loader.py).
4. It uses the model and weights to generate GBCA-predicted maps and store them in (../Newly_generated_prediction/).

The resulting scans shall be identical to those in (../Test_retest_data_complete/NatureBME_share_GBCApredicted/EVERYTHING.nii.gz)

Execution:
```
python generate_new_predictions.py
```

### verify_old_new_predictions_identical.py

This script, as a supplement to the previous script, verifies that the newly generated GBCA-predicted scans are indeed the same as what we generated before using the same inputs and same model. It loads the old and new scans and confirm that not a single voxel is different in each of the scans.

Execution:
```
python verify_old_new_predictions_identical.py
```

Partial results (displayed in the terminal) when running the code above:

![Partial_output_verify_old_new_predictions_identical.PNG](https://github.com/SAIL-GuoLab/DeepContrast_Demo/blob/master/misc/Partial_output_verify_old_new_predictions_identical.PNG)

**!!! Note !!!** Currently, we set the tolerance to declare corresponding voxels to be the same at 1e-8 (which is millions of times smaller than either the dynamic range of these scans or the natural variance among voxels). Due to hardware differences (different computers may exhibit slightly different rounding errors), this tolerance may be too strict if you test the model on your device. We suggest that if the program fails to declare the old and new predictions to be identical you may try a slightly larger tolerance, such as 1e-6.

### visual_inspection.py

This script qualitatively displays the performances of the proposed deep learning approach, to visually showcase the higher between-acquisition coherence of the GBCA-predicted maps over the GBCA-uptake maps. Each run of the script randomly selects a subject and produces a figure with 2x3 subplots using scans from the same subject.

Execution:
```
python visual_inspection.py
```

Sample figure produced when running the code above:

![Test_retest_sample_visual_inspection.png](https://github.com/SAIL-GuoLab/DeepContrast_Demo/blob/master/misc/Test_retest_sample_visual_inspection.png)


### test_retest_evaluation.py

This script performs the quantitative evalution of the test-retest reliability. The numbers calculated and their implications are quite thoroughly described in the script outputs.

Please note that here we not only generate the same numbers we reported (test-retest reliability between two consecutive acquisitions, for GBCA-uptake maps and GBCA-predicted maps), but also print out a metric not included in our manuscript but still worth showing here (concordance between GBCA-uptake maps and GBCA-predicted maps).

With both of these in place, we demonstrate that our deep learning approach not only has great test-retest reliability, but also truly carries GBCA-equivalent information (instead of producing consistent/identical yet meaningless graphs).

Execution:
```
python test_retest_evaluation.py
```

Partial results (displayed in the terminal) when running the code above:

![Partial_output_test_retest_evaluation.PNG](https://github.com/SAIL-GuoLab/DeepContrast_Demo/blob/master/misc/Partial_output_test_retest_evaluation.PNG)

Sample bar plot saved when running the code above (Shall be the same as the first row in Figure 3c of our manuscript, except that it does not have asterisks plotted):

![Test_retest_sample_bar_plot_for_whole_brain.png](https://github.com/SAIL-GuoLab/DeepContrast_Demo/blob/master/misc/Test_retest_sample_bar_plot_for_whole_brain.png)
