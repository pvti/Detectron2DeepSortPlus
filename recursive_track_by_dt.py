#recusively inference on groundtruth of detection
import os
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="inference vids")
    parser.add_argument("--root_path", help="Root path")
    return parser

def main():
    args = get_parser().parse_args()
    directories = os.listdir(args.root_path)
    for directory in directories:
        file_name = directory + '.avi'
        input_vid = os.path.join(args.root_path, directory) + '/' + file_name
        output = os.path.join(args.root_path, file_name)
        command = 'python3 demo_detectron2_deepsort.py ' + input_vid + ' --config-file /home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml ' + '--ignore_display ' + ' --save_path ' + output + ' --opts MODEL.WEIGHTS /home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/projects/Thesis/output/model_final.pth'
        print(command)
        os.system(command)

    return 0

if __name__ == "__main__":
    main()
