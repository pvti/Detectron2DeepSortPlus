# Getting Started with Detectron2DeepSortPlus
After downloading models, please refer 'dt2ds.py' for RCNN models and 'yl2ds' for yolo models.
    For example:  
 ~~~
    python dt2ds.py -h 
~~~

will generate:

~~~
usage: dt2ds.py [-h] [--input INPUT] [--config-file FILE] [--confidence-threshold CONFIDENCE_THRESHOLD] [--region_based REGION_BASED] [--tracker TRACKER]
                [--deepsort_checkpoint DEEPSORT_CHECKPOINT] [--max_dist MAX_DIST] [--nms_max_overlap NMS_MAX_OVERLAP] [--display DISPLAY] [--fps FPS] [--out_vid OUT_VID]
                [--use_cuda USE_CUDA] [--out_txt OUT_TXT] [--opts ...]

Detectron2 to (Deep)SORT demo

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path to input video
  --config-file FILE    path to detectron2 config file
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Minimum score for instance predictions to be shown
  --region_based REGION_BASED
                        1 if track on hand region only. ThanhHai's recommendation
  --tracker TRACKER     tracker type, sort or deepsort
  --deepsort_checkpoint DEEPSORT_CHECKPOINT
                        Cosine metric learning model checkpoint
  --max_dist MAX_DIST   Max cosine distance
  --nms_max_overlap NMS_MAX_OVERLAP
                        Non-max suppression threshold
  --display DISPLAY     Streaming frames to display
  --fps FPS             Output video Frame Per Second
  --out_vid OUT_VID     Output video
  --use_cuda USE_CUDA   Use GPU if true, else use CPU only
  --out_txt OUT_TXT     Write tracking results in MOT16 format to file seqtxt2write. To evaluate using pymotmetrics
  --opts ...            Modify config options using the command-line 'KEY VALUE' pairs

~~~

Model inference example:

~~~
    python dt2ds.py --input ./in_vid.avi --config-file --config-file ../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --tracker sort --out_vid ./out.vid.avi --opts MODEL.WEIGHTS ./fasterrcnnr50fpn3x.pth
~~~
