# Dataset Zoo
We provide several relevant datasets for training and evaluating the Detectron2DeepsortPlus models. 
Annotations are provided in a unified format. If you want to use these datasets, please **follow their licenses**, 
and if you use any of these datasets in your research, please cite the original work.
## Data Format
1. [MICARehab](https://drive.google.com/file/d/1ICEgkyGkPQRTa7eY1gMkbMxx-XFVQTk3/view?usp=sharing) 

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
    
    Frames from each video are also available in the same folder. The 'gt.txt' contains groundtruth in the [MOT16](https://motchallenge.net/data/MOT16/) format.
    
    We provide 2 ways to access labels: 
    
    + **VIA format: 'via_export_json.json'**
    
        Annotation of each image is as follow:
    
    ```
      {
      "0000.png1585642": {
        "filename": "0000.png",
        "size": 1585642,
        "regions": [
          {
            "shape_attributes": {
              "name": "polygon",
              "all_points_x": [ 1248, 1915, 1915, 1248 ],
              "all_points_y": [ 1089, 1089, 1430, 1430 ]
            },
            "region_attributes": { "category_id": "1" }
          }
        ],
        "file_attributes": {}
      },

    ```
    For detection labels, a bounding box can be represented by the 2 points: top-left and bottom-right. Top-left coordinate is (x_min, y_min) while bottom-right is (x_max, y_max) from the "all_points_x" and "all_points_y".
    
    For tracking labels, id of a hand is the "category_id".
    
    Code for visualizing the groundtruth in this way is at [visualize_gt.py](../visualize_gt.py).
    
    For example, 
    
    ```
    python visualize_gt.py --input /path/to/GH010373_6_3150_4744/ --display True --out_vid out_vid.avi
    ```
    will show the groundtruth for the input video and also save this to out_vid.avi as follow:
    
    <img src="GH010373_6_3150_4744_groundtruth.gif" />
    
    The white polygon can be a rectangle or sometime a true polygon represented for the hand's mask as follow:
    
    <img src="GH010383_5_462_968_1_groundtruth.gif" />
    
    Full 32 original quality groundtruth videos are uploaded to this [youtube link](https://youtube.com/playlist?list=PLWBYzJD_wkfs6mab6b8lE1otKp9yqdSoO).
    
    + **MOT16 format: 'gt.txt'.**
    
    This is saved in simple comma-separated value (CSV) files. Each line represents one object instance and contains 9 values.
    
    | Position | Name                | Description                                                                           |
    |----------|---------------------|---------------------------------------------------------------------------------------|
    | 1        | Frame number        | Indicate at which frame the object is present                                         |
    | 2        | Identity number     | Each hand trajectory is identified by a unique ID                                     |
    | 3        | Bounding box left   | Coordinate of the top-left corner of the pedestrian bounding box                      |
    | 4        | Bounding box top    | Coordinate of the top-left corner of the pedestrian bounding box                      |
    | 5        | Bounding box width  | Width in pixels of the hand bounding box                                              |
    | 6        | Bounding box height | Height in pixels of the hand bounding box                                             |
    | 7        | Confidence score    | It acts as a flag whether the entry is to be considered (1) or ignored (0)            |
    | 8        | Class               | Indicates the type of object annotated. In this thesis, always (1)                    |
    | 9        | Visibility          | Visibility ratio, a number between 0 and 1 that says how much of the hand is visible. |
    
    An example of such an annotation file is:
    
    ```
    1, 1, 1672, 763, 248, 245, 1, 1 ,1
    1, 2, 1253, 426, 156, 200, 1, 1 ,1
    2, 1, 1668, 774, 252, 234, 1, 1, 1
    ```
    
    In this case, there are 2 hand in the first frame of the sequence, with identity tags 1, 2 and 1 hand in the second frame with identity tags 1.
    
    This format is useful for evaluating MOT16 metrics with [py-motmetrics](https://github.com/cheind/py-motmetrics).
    
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
