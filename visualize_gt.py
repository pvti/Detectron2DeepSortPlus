import argparse
import os
import time
from distutils.util import strtobool
import cv2
import json
import numpy as np
from tqdm import tqdm

from util import draw_bboxes, draw_polys

def main():
    args = get_parser().parse_args()
    if args.display:
        cv2.namedWindow("out_vid", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("out_vid", 960, 720)
    assert os.path.isfile(os.path.join(args.input, 'via_export_json.json')), "Error: path error, via_export_json.json not found"
    if args.out_vid:
        out_vid = cv2.VideoWriter(
            filename=args.out_vid,
            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
            fps=args.fps,
            frameSize=(1920, 1440),
        )
    if args.out_txt:
        out_txt = open(args.out_txt, "w+")

    total_counter = 0
    json_file = os.path.join(args.input, 'via_export_json.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)
    for idx, v in tqdm(enumerate(imgs_anns.values()), total=len(imgs_anns.values())):
        filename = os.path.join(args.input, v["filename"])
        annos = v["regions"]        
        polys = []
        dets = []
        tlbr_boxes = []
        identities = []
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
                tlbr_boxes.append([np.min(px), np.min(py), np.max(px), np.max(py)])
                identities.append(int(region_attributes["category_id"]))
                polys.append(poly)
        im = cv2.imread(filename)
        im = draw_bboxes(im, tlbr_boxes, identities, binary_masks=[])
        if args.out_txt:
            for i in range(len(identities)):
                tlbr = tlbr_boxes[i]
                line = [idx+1, identities[i], tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1], 1, 1, 1]
                out_txt.write(",".join(str(item) for item in line) + "\n")
                if identities[i]>total_counter:
                    total_counter = identities[i]
        im = draw_polys(im, polys)
        im = cv2.putText(im, "Frame ID: "+str(idx), (20,40), 0, 5e-3 * 200, (0,255,0), 2) 
        im = cv2.putText(im, 'Detection & tracking Groundtruth', (20, 80), 0, 5e-3*200, (0,255,0), 3) 
        im = cv2.putText(im, "Current Hand Counter: "+str(len(identities)),(20, 120), 0, 5e-3 * 200, (0,255,0), 2)
        im = cv2.putText(im, "Total Hand Counter: "+str(total_counter), (20, 160), 0, 5e-3 * 200, (0,255,0), 2)
        if args.display:
            cv2.imshow("out_vid", im)
            cv2.waitKey(1)
        if args.out_vid:
            out_vid.write(im)
            
def get_parser():
    parser = argparse.ArgumentParser(description="Visualizing MOT Groundtruth")
    parser.add_argument("--input", 
         type=str,
         default='/media/data3/EgoCentric_Nafosted/micand26/gt/',
         help='path to input folder contains detection groundtruth', 
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
    "--out_txt",
    type=str,
    default="output_txt.txt",
    help="Write tracking results in MOT16 format to file seqtxt2write. To evaluate using pymotmetrics",
    )
    return parser

if __name__ == "__main__":
    main()
