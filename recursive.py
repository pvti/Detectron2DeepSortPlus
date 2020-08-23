import os
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="inference vids")
    parser.add_argument("--root_path", help="Root path")
    return parser

def main():
    args = get_parser().parse_args()
    vids = open('/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/tienpv13/vids_in_micand.txt', 'r+')
    line = True
    while line:
        line = vids.readline().rstrip()
        output = line.split('/', 8)[8]
        command = 'python3 demo_detectron2_deepsort.py ' + line + ' --config-file /home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml ' + '--ignore_display ' + ' --save_path ' + output + ' --opts MODEL.WEIGHTS /home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/projects/Thesis/output/model_final.pth'
        print(command)
        os.system(command)

    return 0

if __name__ == "__main__":
    main()
