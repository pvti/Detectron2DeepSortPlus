import argparse
import os
import time
from distutils.util import strtobool

import cv2

from deep_sort import DeepSort
from util import draw_bboxes, draw_detections
import json
import numpy as np

class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)


        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        self.total_counter = [0]*1000

    def __enter__(self):
        assert os.path.isfile(os.path.join(self.args.path, 'via_export_json.json')), "Error: path error, via_export_json.json not found"


        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 1, (1920, 1440))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        #while self.vdo.grab():
        json_file = os.path.join(self.args.path, 'via_export_json.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)
        for idx, v in enumerate(imgs_anns.values()):
            filename = os.path.join(self.args.path, v["filename"])
            annos = v["regions"]
            bbox_xcycwh, cls_conf, cls_ids, binary_masks = [], [], [], []
            for anno in annos:
                region_attributes = anno["region_attributes"]
                if not region_attributes:
                    break
                anno = anno["shape_attributes"]
                if anno["name"] != "polygon":
                    break
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                x0, y0, x1, y1 = np.min(px), np.min(py), np.max(px), np.max(py) 
                #for all 9 class of micand, get hand only
                if int(region_attributes["category_id"])==1:
                    cls_ids.append(0)
                    #cls_ids.append(int(region_attributes["category_id"])-1)
                    bbox_xcycwh.append([(x1+x0)/2, (y1+y0)/2, (x1-x0), (y1-y0)])
                    cls_conf.append(1)
 
            start = time.time()
            im = cv2.imread(filename)
            print('----------------------------------------------DEMO started-----------------------------------------------')            
            print(bbox_xcycwh, cls_conf, cls_ids, binary_masks)
            bbox_xcycwh, cls_conf, cls_ids, binary_masks = np.array(bbox_xcycwh), np.array(cls_conf), np.array(cls_ids), np.array(binary_masks)
            current_counter = []
            print(bbox_xcycwh)
            if len(bbox_xcycwh):
                mask = cls_ids == 0 # select class hand
                bbox_xcycwh = bbox_xcycwh[mask]
                cls_conf = cls_conf[mask]
                
                #draw detections after NMS, white box
                
                outputs, detections = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                im = draw_detections(detections, im)
                print('++++++++++++++++++++++++++++++++++++++ outputs of deepsort.update', outputs)
                if len(outputs):
                    bbox_xyxy = outputs[:, :4]
                    identities = current_counter = outputs[:, -1]
                    ordered_identities = []
                    for identity in identities:
                        if not self.total_counter[identity]:
                            self.total_counter[identity] = max(self.total_counter) + 1
                        ordered_identities.append(self.total_counter[identity])                                       
                    im = draw_bboxes(im, bbox_xyxy, ordered_identities, binary_masks)
                    #nums = "len(bbox_xyxy): {}, len(identities): {}, len(binary_masks): {}".format(len(bbox_xyxy), len(identities), len(binary_masks))
                    #im = cv2.putText(im, nums, (150, 150), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
                    
            end = time.time()
            time_fps = "time: {}s, fps: {}".format(round(end - start, 2), round(1 / (end - start), 2))            
            im = cv2.putText(im, "Total Hand Counter: "+str(max(self.total_counter)), (int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
            im = cv2.putText(im, "Current Hand Counter: "+str(len(current_counter)),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
            im = cv2.putText(im, time_fps,(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
            if self.args.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(im)
            # exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", 
            type=str,
            default='/media/data3/EgoCentric_Nafosted/non_skip/train/',
            help='path to folder contains detection groundtruth',
    )
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
