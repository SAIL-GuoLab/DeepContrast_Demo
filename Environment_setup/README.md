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
  
  <details><summary> 1. The exhaustive (but maybe unnecessary) list, directly exported from the environment where we developed the model, can be found here. </summary>
<p>
  
```
channels:
  - simpleitk
  - bioconda
  - anaconda
  - conda-forge
  - defaults
dependencies:
  - _pytorch_select=0.2=gpu_0
  - alabaster=0.7.12=py37_0
  - argh=0.26.2=py37_0
  - asn1crypto=1.2.0=py37_0
  - astroid=2.3.3=py37_0
  - atomicwrites=1.3.0=py37_1
  - attrs=19.3.0=py_0
  - autopep8=1.4.4=py_0
  - babel=2.7.0=py_0
  - backcall=0.1.0=py37_0
  - blas=1.0=mkl
  - bleach=3.1.0=py37_0
  - ca-certificates=2020.4.5.2=hecda079_0
  - certifi=2020.4.5.2=py37hc8dfbb8_0
  - cffi=1.13.2=py37h2e261b9_0
  - chardet=3.0.4=py37_1003
  - cloudpickle=1.2.2=py_0
  - cryptography=2.8=py37h1ba5d50_0
  - cudatoolkit=10.0.130=0
  - cudnn=7.6.5=cuda10.0_0
  - cycler=0.10.0=py37_0
  - cytoolz=0.10.1=py37h7b6447c_0
  - dask-core=2.9.0=py_0
  - dbus=1.13.12=h746ee38_0
  - decorator=4.4.1=py_0
  - defusedxml=0.6.0=py_0
  - diff-match-patch=20181111=py_0
  - docutils=0.15.2=py37_0
  - entrypoints=0.3=py37_0
  - expat=2.2.6=he6710b0_0
  - fastdtw=0.2.0=py_1
  - flake8=3.7.9=py37_0
  - fontconfig=2.13.0=h9420a91_0
  - freetype=2.9.1=h8a8886c_1
  - future=0.18.2=py37_0
  - git=2.23.0=pl526hacde149_0
  - glib=2.56.2=hd408876_0
  - gmp=6.1.2=hb3b607b_0
  - gst-plugins-base=1.14.0=hbbd80ab_1
  - gstreamer=1.14.0=hb453b48_1
  - h5py=2.9.0=py37h7918eee_0
  - hdf5=1.10.4=hb1b8bf9_0
  - icu=58.2=h211956c_0
  - idna=2.8=py37_0
  - imageio=2.6.1=py37_0
  - imagesize=1.1.0=py37_0
  - importlib_metadata=1.3.0=py37_0
  - intel-openmp=2019.5=281
  - intervaltree=3.0.2=py_0
  - ipykernel=5.1.3=py37h39e3cac_0
  - ipython=7.10.2=py37h39e3cac_0
  - ipython_genutils=0.2.0=py37_0
  - ipywidgets=7.5.1=py_0
  - isort=4.3.21=py37_0
  - jedi=0.14.1=py37_0
  - jeepney=0.4.1=py_0
  - jinja2=2.10.3=py_0
  - joblib=0.14.1=py_0
  - jpeg=9b=habf39ab_1
  - json5=0.8.5=py_0
  - jsonschema=3.2.0=py37_0
  - jupyter=1.0.0=py37_7
  - jupyter_client=5.3.4=py37_0
  - jupyter_console=6.0.0=py37_0
  - jupyter_core=4.6.1=py37_0
  - jupyterlab=1.2.4=pyhf63ae98_0
  - jupyterlab_server=1.0.6=py_0
  - keyring=20.0.0=py37_0
  - kiwisolver=1.1.0=py37he6710b0_0
  - krb5=1.16.4=h173b8e3_0
  - lazy-object-proxy=1.4.3=py37h7b6447c_0
  - libcurl=7.67.0=h20c2e04_0
  - libedit=3.1.20181209=hc058e9b_0
  - libffi=3.2.1=h4deb6c0_3
  - libgcc-ng=9.1.0=hdf63c60_0
  - libgfortran-ng=7.3.0=hdf63c60_0
  - libpng=1.6.37=hbc83047_0
  - libsodium=1.0.16=h1bed415_0
  - libspatialindex=1.9.3=he6710b0_0
  - libssh2=1.8.2=h1ba5d50_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - libtiff=4.1.0=h2733197_0
  - libuuid=1.0.3=h1bed415_2
  - libxcb=1.13=h1bed415_1
  - libxml2=2.9.9=hea5a465_1
  - markupsafe=1.1.1=py37h7b6447c_0
  - matplotlib=3.1.1=py37h5429711_0
  - matplotlib-base=3.1.3=py37hef1b27d_0
  - mccabe=0.6.1=py37_1
  - mistune=0.8.4=py37h7b6447c_0
  - mkl=2019.5=281
  - mkl-service=2.3.0=py37he904b0f_0
  - mkl_fft=1.0.15=py37ha843d7b_0
  - mkl_random=1.1.0=py37hd6b4f25_0
  - more-itertools=8.0.2=py_0
  - nbconvert=5.6.1=py37_0
  - nbformat=4.4.0=py37_0
  - ncurses=6.1=he6710b0_1
  - networkx=2.4=py_0
  - nibabel=3.0.0=py_0
  - nilearn=0.6.2=pyh5ca1d4c_0
  - ninja=1.9.0=py37hfd86e86_0
  - notebook=6.0.2=py37_0
  - numpy=1.17.4=py37hc1035e2_0
  - numpy-base=1.17.4=py37hde5b4d6_0
  - numpydoc=0.9.1=py_0
  - olefile=0.46=py37_0
  - openssl=1.1.1g=h516909a_0
  - packaging=19.2=py_0
  - pandas=0.25.3=py37he6710b0_0
  - pandoc=2.2.3.2=0
  - pandocfilters=1.4.2=py37_1
  - parso=0.5.2=py_0
  - pathtools=0.1.2=py_1
  - patsy=0.5.1=py37_0
  - pcre=8.43=he6710b0_0
  - perl=5.26.2=h14c3975_0
  - pexpect=4.7.0=py37_0
  - pickleshare=0.7.5=py37_0
  - pillow=6.2.1=py37h34e0f95_0
  - pip=20.0.2=py37_1
  - pluggy=0.13.1=py37_0
  - prometheus_client=0.7.1=py_0
  - prompt_toolkit=2.0.9=py37_0
  - psutil=5.6.7=py37h7b6447c_0
  - ptyprocess=0.6.0=py37_0
  - pycodestyle=2.5.0=py37_0
  - pycparser=2.19=py37_0
  - pydicom=1.3.0=py_0
  - pydocstyle=4.0.1=py_0
  - pyflakes=2.1.1=py37_0
  - pygments=2.5.2=py_0
  - pylint=2.4.4=py37_0
  - pympler=0.7=py_0
  - pyopenssl=19.1.0=py37_0
  - pyparsing=2.4.5=py_0
  - pyqt=5.9.2=py37h22d08a2_1
  - pyrsistent=0.15.6=py37h7b6447c_0
  - pysocks=1.7.1=py37_0
  - python=3.7.5=h0371630_0
  - python-dateutil=2.8.1=py_0
  - python-jsonrpc-server=0.3.2=py_0
  - python-language-server=0.31.2=py37_0
  - python_abi=3.7=1_cp37m
  - pytorch=1.3.1=cuda100py37h53c1284_0
  - pytz=2019.3=py_0
  - pywavelets=1.1.1=py37h7b6447c_0
  - pyxdg=0.26=py_0
  - pyyaml=5.2=py37h7b6447c_0
  - pyzmq=18.1.0=py37he6710b0_0
  - qdarkstyle=2.7=py_0
  - qt=5.9.7=h5867ecd_1
  - qtawesome=0.6.0=py_0
  - qtconsole=4.6.0=py_0
  - qtpy=1.9.0=py_0
  - readline=7.0=h7b6447c_5
  - requests=2.22.0=py37_1
  - rope=0.14.0=py_0
  - rtree=0.8.3=py37_0
  - scikit-image=0.15.0=py37he6710b0_0
  - scikit-learn=0.22=py37hd81dba3_0
  - scipy=1.3.2=py37h7c811a0_0
  - seaborn=0.9.0=pyh91ea838_1
  - secretstorage=3.1.1=py37_0
  - send2trash=1.5.0=py37_0
  - setuptools=42.0.2=py37_0
  - simpleitk=1.2.4=py37hf484d3e_0
  - sip=4.19.13=py37he6710b0_0
  - six=1.13.0=py37_0
  - snowballstemmer=2.0.0=py_0
  - sortedcontainers=2.1.0=py37_0
  - sphinx=2.3.0=py_0
  - sphinxcontrib-applehelp=1.0.1=py_0
  - sphinxcontrib-devhelp=1.0.1=py_0
  - sphinxcontrib-htmlhelp=1.0.2=py_0
  - sphinxcontrib-jsmath=1.0.1=py_0
  - sphinxcontrib-qthelp=1.0.2=py_0
  - sphinxcontrib-serializinghtml=1.1.3=py_0
  - spyder=4.0.0=py37_0
  - spyder-kernels=1.8.1=py37_0
  - sqlite=3.30.1=h7b6447c_0
  - statsmodels=0.10.1=py37hdd07704_0
  - terminado=0.8.3=py37_0
  - testpath=0.4.4=py_0
  - tk=8.6.8=hbc83047_0
  - toolz=0.10.0=py_0
  - torchvision=0.4.2=cuda100py37hecfc37a_0
  - tornado=6.0.3=py37h7b6447c_0
  - tqdm=4.43.0=py_0
  - traitlets=4.3.3=py37_0
  - ujson=1.35=py37h14c3975_0
  - urllib3=1.25.7=py37_0
  - watchdog=0.9.0=py37_1
  - wcwidth=0.1.7=py37_0
  - webencodings=0.5.1=py37_1
  - wheel=0.33.6=py37_0
  - widgetsnbextension=3.5.1=py37_0
  - wrapt=1.11.2=py37h7b6447c_0
  - wurlitzer=2.0.0=py37_0
  - xlrd=1.2.0=py37_0
  - xz=5.2.4=h14c3975_4
  - yaml=0.1.7=h96e3832_1
  - yapf=0.28.0=py_0
  - zeromq=4.3.1=he6710b0_3
  - zipp=0.6.0=py_0
  - zlib=1.2.11=h7b6447c_3
  - zstd=1.3.7=h0b5b093_0
  - pip:
    - absl-py==0.9.0
    - click==7.1.1
    - mpmath==1.1.0
    - pytorch-msssim==0.1
    - robust-loss-pytorch==0.0.2
    - torch-dct==0.1.5
    - torchio==0.13.16
```
</p>
</details>



<details><summary> 2. As a shortcut, you can try the following commands. We tested it on a Windows computer to reproduce an environment that can run the scripts. </summary>
<p>

```
conda create -n DeepContrast

conda activate DeepContrast
conda install python=3.7 numpy scipy scikit-image scikit-learn -c anaconda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install nibabel -c conda-forge
```

</p>
</details>
  
</p>
</details>


</p>
</details>


<details><summary> If you do not have Anaconda installed. </summary>
<p>
  You can refer to [this tutorial!](https://github.com/RnR-2018/Deep-learning-with-PyTorch-and-GCP/tree/master/Step01_manage_anaconda_on_GCP).
</p>
</details>
