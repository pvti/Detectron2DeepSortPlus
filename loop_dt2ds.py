#recusively inference detectron2
import os
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="inference vids")
    parser.add_argument(
        "--root_path",
        default="/media/data3/EgoCentric_Nafosted/non_skip/train/",
        help="Root path",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="Config file",
    )
    parser.add_argument(
        "--region_based",
        type=str,
        default='False',
        help="Region based",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default='/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/projects/Thesis/output/model_final.pth',
        help="Path to model weights",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="sort",
        help="sort or deepsort",
    )
    parser.add_argument(
        "--out_vids",
        type=str,
        default='/media/data3/EgoCentric_Nafosted/non_skip/out_vids/',
        help='Path to folder to save output videos',
    )
    parser.add_argument(
        "--out_txts",
        type=str,
        default='/media/data3/EgoCentric_Nafosted/mot/test/',
        help='Path to folder to save output sequence texts',
    )
    return parser

def main():
    args = get_parser().parse_args()
    directories = os.listdir(args.root_path)
    for i in tqdm(range(len(directories))):
        directory = directories[i]
        file_name = directory + '.avi'
        input_vid = os.path.join(args.root_path, directory) + '/' + file_name
        out_vid = os.path.join(args.out_vids, file_name)
        out_txt = os.path.join(args.out_txts, directory+'.txt')
        command = 'python3 dt2ds.py --input ' + input_vid + ' --config-file ' + args.config_file + ' --region_based ' + args.region_based +  ' --out_vid ' + out_vid + ' --out_txt ' +  out_txt + ' --opts MODEL.WEIGHTS ' + args.model_weights 
        print(command)
        os.system(command)

    return 0

if __name__ == "__main__":
    main()
