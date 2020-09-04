import argparse
import os
import time
from distutils.util import strtobool

import cv2

from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes, draw_detections


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2(args)
        self.deepsort = DeepSort(args.deepsort_checkpoint, nms_max_overlap=args.nms_max_overlap, use_cuda=use_cuda)
        self.total_counter = [0]*1000

    def __enter__(self):
        assert os.path.isfile(self.args.video_path), "Error: path error"
        self.vdo.open(self.args.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 1, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            print('----------------------------------------------DEMO started-----------------------------------------------')            
            bbox_xcycwh, cls_conf, cls_ids, cls_masks, region = self.detectron2.detect(im)
            #print('bbox_xcycwh, cls_conf, cls_ids, cls_masks', bbox_xcycwh, cls_conf, cls_ids, cls_masks)
            
            if args.region_based:
                im = region
            #if bbox_xcycwh is not None:
            current_counter = []
            if len(bbox_xcycwh):
                mask = cls_ids == 0 # select class person
                #print('mask>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', mask)

                bbox_xcycwh = bbox_xcycwh[mask]

                print('bbox_xcycwh', bbox_xcycwh)
                
                #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^cls_conf', cls_conf)
                cls_conf = cls_conf[mask]
                #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^cls_masks[mask]', cls_conf[mask])
                binary_masks = cls_masks[mask]
                #binary_masks = cls_masks
                
                #draw detections after NMS, white box
                
                outputs, detections = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                print('++++++++++++++++++++++++++++++++++++++ outputs of deepsort.update', outputs)
                if len(outputs):
                    bbox_xyxy = outputs[:, :4]
                    #print("+++++++++++++++++++++++++++++++++++++bbox_xyxy, bbox_xyxy_detectron2", bbox_xyxy, bbox_xyxy_detectron2)
                    identities = current_counter = outputs[:, -1]
                    #print("+++++++++++++++++++++++++++++++++++++identities", identities)
                    ordered_identities = []
                    for identity in identities:
                        if not self.total_counter[identity]:
                            self.total_counter[identity] = max(self.total_counter) + 1
                        ordered_identities.append(self.total_counter[identity])      
                    im = draw_detections(detections, im)                                 
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
    parser.add_argument("--video_path",
            type=str,
            default='/media/data3/EgoCentric_Nafosted/non_skip/train/',
            help='path to input video',
    )
    parser.add_argument(
        "--config-file",
        default="/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/configs/COCO-InstanceSegemntation/mask_r_cnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to detectron2 config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--nms_max_overlap",
            type=float,
            default=0.5,
            help='Non-max suppression threshold',
    )
 
    parser.add_argument("--region_based", type=bool, default=False, help="True if track on hand region only. ThanhHai's recommendation")
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
