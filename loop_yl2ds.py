#recusively inference yolov5
import os
from tqdm import tqdm
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="inference vids")
    parser.add_argument(
        "--root_path",
        default="/media/data3/EgoCentric_Nafosted/micand26/gt/",
        help="Root path",
    )    
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
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
        command = 'python3 yl2ds.py --input ' + input_vid + ' --tracker '+ args.tracker + ' --out_vid ' + out_vid + ' --out_txt ' +  out_txt + ' --weights ' + args.weights 
        print(command)
        os.system(command)

    return 0

if __name__ == "__main__":
    main()
