import argparse
import os
import time
from distutils.util import strtobool
import cv2
import json
import numpy as np
from tqdm import tqdm

from sort import Sort
from deep_sort import DeepSort

from util import draw_bboxes, draw_polys

def main():
    args = get_parser().parse_args()
    if args.display:
        cv2.namedWindow("out_vid", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("out_vid", 960, 720)
    sort = Sort()
    deepsort = DeepSort(args.deepsort_checkpoint, nms_max_overlap=args.nms_max_overlap, use_cuda=bool(strtobool(args.use_cuda)))
    assert os.path.isfile(os.path.join(args.input, 'via_export_json.json')), "Error: path error, via_export_json.json not found"
    '''
    if args.out_vid:
        out_vid = cv2.VideoWriter(
            filename=args.out_vid,
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=args.fps,
            frameSize=(1920, 1440),
        )
    '''
    if args.out_txt:
        out_txt = open(args.out_txt, "w+")

    total_counter = [0]*1000
    json_file = os.path.join(args.input, 'via_export_json.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)
    for idx, v in tqdm(enumerate(imgs_anns.values()), total=len(imgs_anns.values())):
        filename = os.path.join(args.input, v["filename"])
        annos = v["regions"]        
        polys = []
        dets = []
        for anno in annos:
            region_attributes = anno["region_attributes"]
            if not region_attributes:
                break
            anno = anno["shape_attributes"]
            if anno["name"] != "polygon":
                break
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = np.array([[x, y] for x, y in zip(px, py)], np.int32).reshape((-1,1,2))
            if int(region_attributes["category_id"]):
                dets.append([np.min(px), np.min(py), np.max(px), np.max(py), 1])
                polys.append(poly)
        start = time.time()
        im = cv2.imread(filename)
        current_counter = []
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
        if len(outputs):
            tlbr_boxes = outputs[:, :4]
            identities = current_counter = outputs[:, -1]
            ordered_identities = []
            for identity in identities:
                if not total_counter[identity]:
                    total_counter[identity] = max(total_counter) + 1
                ordered_identities.append(total_counter[identity])
            im = draw_bboxes(im, tlbr_boxes, ordered_identities, binary_masks=[])
            if args.out_txt:
                for i in range(len(ordered_identities)):
                    tlbr = tlbr_boxes[i]
                    line = [idx+1, ordered_identities[i], tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1], 1, 1, 1]
                    out_txt.write(",".join(str(item) for item in line) + "\n")
        end = time.time()
        im = draw_polys(im, polys)
        im = cv2.putText(im, "Frame ID: "+str(idx), (20,20), 0, 5e-3 * 200, (0,255,0), 2) 
        time_fps = "Time: {}s, fps: {}".format(round(end - start, 2), round(1 / (end - start), 2))            
        im = cv2.putText(im, time_fps,(20, 60), 0, 5e-3 * 200, (0,255,0), 3)      
        im = cv2.putText(im, 'Groundtruth2'+args.tracker, (20, 100), 0, 5e-3*200, (0,255,0), 3) 
        im = cv2.putText(im, "Current Hand Counter: "+str(len(current_counter)),(20, 140), 0, 5e-3 * 200, (0,255,0), 2)
        im = cv2.putText(im, "Total Hand Counter: "+str(max(total_counter)), (20, 180), 0, 5e-3 * 200, (0,255,0), 2)
        if args.display:
            cv2.imshow("out_vid", im)
            cv2.waitKey(1)
        '''
        if args.out_vid:
            out_vid.write(im)
        '''
            
def get_parser():
    parser = argparse.ArgumentParser(description="Grounttruth to (Deep)SORT demo")
    parser.add_argument("--input", 
         type=str,
         default='/media/data3/EgoCentric_Nafosted/micand26/gt/',
         help='path to input folder contains detection groundtruth', 
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
    return parser

if __name__ == "__main__":
    main()
