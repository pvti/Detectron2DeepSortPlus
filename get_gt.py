import argparse
import os
import cv2
from tqdm import tqdm

#for verify_load_data
import numpy as np 
import random
from detectron2.utils.visualizer import Visualizer

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

import scipy.io as sio
import json 

#for evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

def setup_cfg(args):
    #load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #initialize a pretrained weights
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2 #num-gpus=1
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Mica datasets for Detectron2")
    parser.add_argument(
        "--config-file",
        default="../../configs/Base-RCNN-FPN.yaml",
        metavar="FILE",
        help="path to config file",
        )
    parser.add_argument(
        "--mica-dir",
        default="/media/data3/EgoCentric_Nafosted/",
        help="path to mica dataset"
        )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_hand_dicts(img_dir):
    directories = os.listdir(img_dir)
    dataset_dicts = []
    for directory in directories:
        directory = os.path.join(img_dir, directory)
        print(directory)
        json_file = os.path.join(directory, "via_export_json.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)
    

        for idx, v in enumerate(imgs_anns.values()):
            record = {}
            filename = os.path.join(directory, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
          
            annos = v["regions"]
            objs = []
            #for _, anno in annos.items():
            for anno in annos:
                #assert not anno["region_attributes"]
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
    
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": int(region_attributes["category_id"])-1,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def verify_load_data(args, num_samples, mica_metadata):
    dataset_dicts = get_hand_dicts(args.mica_dir + "train")
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=mica_metadata, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        vis = visualizer.draw_dataset_dict(d)
        #cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])        
        print(d["file_name"])
        print("/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/projects/Mica/samples/" + d["file_name"].split("/")[7])
        cv2.imwrite("/home/minhkv/tienpv_DO_NOT_REMOVE/detectron2/projects/Mica/samples/" + d["file_name"].split("/")[7], vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return

def save_gt(args, mica_metadata):
    dataset_dicts = get_hand_dicts(args.mica_dir + "train")
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=mica_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(d["file_name"], vis.get_image()[:, :, ::-1])
    return 
        
def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    for d in ["train", "val"]:
        DatasetCatalog.register("mica_" + d, lambda d=d: get_hand_dicts(args.mica_dir + d))
        MetadataCatalog.get("mica_" + d).set(thing_classes=['hand'])

    #verify_load_data(args, num_samples=100, mica_metadata=MetadataCatalog.get("mica_train"))
    save_gt(args, mica_metadata=MetadataCatalog.get("mica_train"))
    return 0

if __name__ == "__main__":
    main()
