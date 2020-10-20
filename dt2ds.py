import argparse
import os
import time
from distutils.util import strtobool
import cv2
import numpy as np
from tqdm import tqdm

#from detectron2_dt import detectron2

from sort import Sort
from deep_sort import DeepSort

from util import draw_bboxes, draw_detections


from detectron2.utils.logger import setup_logger
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def detectron2(im, args, predictor):
    #predictor = DefaultPredictor(setup_cfg(args))
    predictions = predictor(im)
    boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
    scores = predictions["instances"].scores.cpu().numpy()
    dets = []
    for (box, score) in zip(boxes, scores):
        t, l, b, r = box
        dets.append([t, l, b, r, score])
    if os.path.basename(args.config_file).split('_')[0] == 'mask':
        predict_masks = predictions["instances"].pred_masks
        masks = predict_masks.cpu().numpy()
        temp = np.zeros_like(im[:, :, 0])
        for i in range(len(predict_masks)):
            predict_mask_i = predict_masks[i]
            temp += np.array(predict_mask_i.to("cpu").numpy()).astype(np.uint8)
        region = im.copy()
        region[temp == 0] = 0
        region[temp!= 0] = im[temp != 0]
        return dets, np.array(masks), region
    return dets, [], []

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.merge_from_list(args.opts)
    return cfg


def main():
    args = get_parser().parse_args()
    if args.display:
        cv2.namedWindow("out_vid", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("out_vid", 960, 720)
    sort = Sort()
    deepsort = DeepSort(args.deepsort_checkpoint, nms_max_overlap=args.nms_max_overlap, use_cuda=bool(strtobool(args.use_cuda)))
    assert os.path.isfile(args.input), "Error: path error, input file not found"
    
    if args.out_vid:
        out_vid = cv2.VideoWriter(
            filename=args.out_vid,
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=args.fps,
            frameSize=(1920, 1440),
        )
    
    if args.out_txt:
        out_txt = open(args.out_txt, "w+")
    total_counter = [0]*1000
    inp_vid = cv2.VideoCapture(args.input)
    num_frames = int(inp_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    predictor = DefaultPredictor(setup_cfg(args))
    for frameID in tqdm(range(num_frames)):
        ret, im = inp_vid.read()
        start = time.time()
        dets, masks, region = detectron2(im, args, predictor)
        if args.region_based:
            im = region
        if args.tracker == 'sort':
            if len(dets):
                dets = np.array(dets)
            else:
                dets = np.empty((0,5))
            outputs = sort.update(dets)
            outputs = np.array([element.clip(min=0) for element in outputs]).astype(int)
        else:
            if len(dets):
                ccwh_boxes = []
                for det in dets:
                    ccwh_boxes.append([(det[0]+det[2])/2, (det[1]+det[3])/2, det[2]-det[0], det[3]-det[1]])  
                ccwh_boxes = np.array(ccwh_boxes)
                confidences = np.ones(len(dets))
                outputs, __ = deepsort.update(ccwh_boxes, confidences, im)
            else:
                outputs = []
        current_counter = []
        if len(outputs):
            tlbr_boxes = outputs[:, :4]
            identities = current_counter = outputs[:, -1]
            ordered_identities = []
            for identity in identities:
                if not total_counter[identity]:
                    total_counter[identity] = max(total_counter) + 1
                ordered_identities.append(total_counter[identity])
            im = draw_bboxes(im, tlbr_boxes, ordered_identities, binary_masks=masks)
            if args.out_txt:
                for i in range(len(ordered_identities)):
                    tlbr = tlbr_boxes[i]
                    line = [frameID+1, ordered_identities[i], tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1], 1, 1, 1]
                    out_txt.write(",".join(str(item) for item in line) + "\n")
        
        end = time.time()
        im = cv2.putText(im, "Frame ID: "+str(frameID+1), (20,30), 0, 5e-3 * 200, (0,255,0), 2) 
        time_fps = "Time: {}s, fps: {}".format(round(end - start, 2), round(1 / (end - start), 2))            
        im = cv2.putText(im, time_fps,(20, 60), 0, 5e-3 * 200, (0,255,0), 3)      
        im = cv2.putText(im, os.path.basename(args.config_file) + ' ' + args.tracker, (20, 90), 0, 5e-3*200, (0,255,0), 3) 
        im = cv2.putText(im, "Current Hand Counter: "+str(len(current_counter)),(20, 120), 0, 5e-3 * 200, (0,255,0), 2)
        im = cv2.putText(im, "Total Hand Counter: "+str(max(total_counter)), (20, 150), 0, 5e-3 * 200, (0,255,0), 2)
        if args.display:
            cv2.imshow("out_vid", im)
            cv2.waitKey(1)
        if args.out_vid:
            out_vid.write(im)
        frameID+=1

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 to (Deep)SORT demo")
    parser.add_argument("--input", 
         type=str,
         default='/media/data3/EgoCentric_Nafosted/micand26/gt/',
         help='path to input video', 
    )
    parser.add_argument(
        "--config-file",
        default="../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to detectron2 config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--region_based",
        type=int,
        default=0,
        help="1 if track on hand region only. ThanhHai's recommendation",
    )
    parser.add_argument("--tracker",
        type=str,
        default='sort',
        help='tracker type, sort or deepsort',
    )
    parser.add_argument("--deepsort_checkpoint",
        type=str,
        default="deep_sort/deep/checkpoint/ckpt.t7",
        help='Cosine metric learning model checkpoint',
    )
    parser.add_argument(
        "--max_dist",
        type=float, 
        default=0.3,
        help="Max cosine distance",
    )
    parser.add_argument("--nms_max_overlap",
        type=float,
        default=0.5,
        help='Non-max suppression threshold',
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=False,
        help="Streaming frames to display",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video Frame Per Second",
    )
    parser.add_argument(
        "--out_vid", 
        type=str, 
        default="output_video.avi",
        help="Output video",
    )
    parser.add_argument(
        "--use_cuda", 
        type=str, 
        default="True",
        help="Use GPU if true, else use CPU only",
    )
    parser.add_argument(
        "--out_txt",
        type=str,
        default="output_txt.txt",
        help="Write tracking results in MOT16 format to file seqtxt2write. To evaluate using pymotmetrics",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    main()
