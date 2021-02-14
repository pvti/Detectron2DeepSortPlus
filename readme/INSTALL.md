# Installation


The code was tested on Ubuntu 18.04, with [Anaconda](https://www.anaconda.com/download) Python 3.7, CUDA 10.2, and [PyTorch]((http://pytorch.org/)) v1.4.
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create --name d2dp python=3.7
    ~~~
    And activate the environment.
    
    ~~~
    conda activate d2dp
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch torchvision -c pytorch
    ~~~
    

2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ~~~

3. Clone this repo:

    ~~~
    d2dp_ROOT=/path/to/clone/d2dp
    git clone --recursive https://github.com/pvtien96/Detectron2DeepSortPlus $d2dp_ROOT

    You can manually install the [submodules](../.gitmodules) if you forget `--recursive`.

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Install Detectron2
   
   ~~~
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ~~~
  Issues can be solved at [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
    
6. Download pertained models from [Model_zoo](MODEL_ZOO.md) and move them to `$d2dp_ROOT/models/`.
