# Hand detection, segmentation and tracking from egocentric vision
![](readme/proposedFramework.png)
![Alt Text](https://media.giphy.com/media/MCjhfIlUY9udz9yOuS/giphy.gif) \
Contact: [pvtien96@gmail.com](mailto:pvtien96@gmail.com). Discussions are welcome!

## Abstract
Egocentric vision is an emerging field of computer vision characterized by the acquisition video from the first person perspective in which hand is essential in the execution of activity and portraying its trajectory is the principal cue for action recognition. We develop a fully automatic tracking by detection pipeline that extracts hands positions and identities in consequence frames. The proposed framework consists of state of the art detectors from RCNN and YOLO family models combined with the SORT or DeepSORT for tracking task. This paper aims to explore how the stand alone performance of the object detection algorithm correlates with overall performance of a tracking by detection system. Results reports that the capacity of the object detection algorithm is highly indicative of the overall performance. Further, this work also shows how the use of visual descriptors in the tracking stage can reduce the number of identity switches and thereby increase performance of the whole system. We also presents a new egocentric hand tracking dataset Micand32 for future researches.

## Main results

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## License

Code is released under the [Apache 2.0 license](LICENSE).

## Citing

If you use this work in your research or wish to refer to the results, please use the following BibTeX entry.

```BibTeX
@misc{tien2021d2dp,
  author =       {Van-Tien Pham and Thanh-Hai Tran and Hai Vu},
  title =        {Hand detection, segmentation and tracking from egocentric vision},
  howpublished = {\url{https://github.com/pvtien96/Detectron2DeepSortPlus}},
  year =         {2021}
}
```
