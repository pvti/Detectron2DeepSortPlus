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
        path = os.path.join(args.root_path, directory) 
        output = os.path.join(args.root_path, file_name)
        command = 'python3 track_only.py --path ' + path + ' --ignore_display ' + ' --save_path ' + output
        print(command)
        os.system(command)

    return 0

if __name__ == "__main__":
    main()
