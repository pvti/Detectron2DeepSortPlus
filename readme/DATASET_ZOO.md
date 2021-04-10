# Dataset Zoo
We provide several relevant datasets for training and evaluating the Joint Detection and Embedding (JDE) model. 
Annotations are provided in a unified format. If you want to use these datasets, please **follow their licenses**, 
and if you use any of these datasets in your research, please cite the original work.
## Data Format
1. [MICARehab]() 

    The MICARehab dataset has the following structure:
    ```
    MICARehab 
    ├── GH010383_8_3221_3956_2/
    │   ├── 0000.png
    │   ├── ...
    │   ├── 000N.png
    │   ├── GH010383_8_3221_3956_2.avi
    │   ├── gt
    │   │   └── gt.txt
    │   ├── seqinfo.ini
    │   ├── via_export_json.json
    ├── ...
    ├── GH0XXXX_X_XXXX_XXXX_X/
    ```

    This dataset contains 32 videos labelled via [VIA](robots.ox.ac.uk/~vgg/software/via/). 

    Every video has a corresponding annotation text 'via_export_json.json', both are in the same folder.

    More detail can be found in the paper:
    ```
    @misc{tien2021d2dp,
      author =       {Van-Tien Pham and Thanh-Hai Tran and Hai Vu},
      title =        {Detection and tracking hand from FPV: benchmarks and challenges on rehabilitation exercises dataset},
      howpublished = {\url{https://github.com/pvtien96/Detectron2DeepSortPlus}},
      year =         {2021}
    }
    ```

2. [EgoHands](http://vision.soic.indiana.edu/projects/egohands/)
```
    @InProceedings{Bambach_2015_ICCV,
    author = {Bambach, Sven and Lee, Stefan and Crandall, David J. and Yu, Chen},
    title = {Lending A Hand: Detecting Hands and Recognizing Activities in Complex Egocentric Interactions},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {December},
    year = {2015}
    }
```
 3. [Georgia Tech Egocentric Activity Datasets](http://cbs.ic.gatech.edu/fpv/)
 ```
    @misc{li2020eye,
          title={In the Eye of the Beholder: Gaze and Actions in First Person Video}, 
          author={Yin Li and Miao Liu and James M. Rehg},
          year={2020},
          eprint={2006.00626},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
```
