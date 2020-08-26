## Environment_setup

### How to configure the environment?

<details><summary> If you already have Anaconda installed. </summary>
<p>
If you already have anaconda installed, it's great! You will only need to run the following command in your command line / terminal / bash, after navigating to this folder.

```
conda env create -f DeepContrast.yml
```

Anaconda shall be able to configure the environment correctly.

<details><summary> If it works. </summary>
<p>
  Congrats! Nothing else to say.
</p>
</details>

<details><summary> If it doesn't work. </summary>
<p>
You may need to manually install the packages.

You have the following options.

<details><summary> 1. [RECOMMENDED] As a shortcut, you can try the following commands. We tested it on a Windows computer to reproduce an environment that can run the scripts. </summary>
<p>

```
conda create -n DeepContrast

conda activate DeepContrast
conda install python=3.7 numpy scipy scikit-image scikit-learn seaborn -c anaconda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install nibabel tqdm -c conda-forge
```

Remember to hit 'y' followed by 'Enter' / 'Return' to allow installation of packages.

</p>
</details>

2. The exhaustive (but maybe unnecessary) list, directly exported from the environment where we developed the model, can be found in "DeepContrast.yml".

</p>
</details>

</p>
</details>


<details><summary> If you do not have Anaconda installed. </summary>
<p>
  You can refer to [this tutorial](https://github.com/RnR-2018/Deep-learning-with-PyTorch-and-GCP/tree/master/Step01_manage_anaconda_on_GCP#step-by-step-instructions) we wrote for a deep learning course. The step-by-step instructions provides a brief guide on how to download and set up anaconda as well as how to prepare environments and install individual packages. The specific packages to be downloaded for our purpose (in environment "DeepContrast") will be different from what listed in that tutorial for the course. Specifically, the required packages can be found here.
</p>
</details>
