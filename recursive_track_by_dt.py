#recusively inference on groundtruth of detection
import os
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="inference vids")
    parser.add_argument("--root_path", default="/media/data3/EgoCentric_Nafosted/non_skip/train/", help="Root path")
    return parser

def main():
    args = get_parser().parse_args()
    directories = os.listdir(args.root_path)
    for directory in directories:
        file_name = directory + '.avi'
        input_vid = os.path.join(args.root_path, directory) + '/' + file_name
        output = os.path.join(args.root_path, file_name)
        command = 'python3 track_by_dt.py --video_path ' + input_vid + ' --config-file /home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --region_based True --ignore_display --save_path ' + output + ' --seqtxt2write /media/data3/EgoCentric_Nafosted/mot/test/' + directory + '.txt'+ ' --opts MODEL.WEIGHTS /home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/projects/Thesis/output/model_final.pth'
        print(command)
        os.system(command)

    return 0

if __name__ == "__main__":
    main()
