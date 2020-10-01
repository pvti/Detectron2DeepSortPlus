from detectron2.utils.logger import setup_logger
import numpy as np
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def detectron2(im, args):
    predictor = DefaultPredictor(setup_cfg(args))
    predictions = predictor(im)
    boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
    scores = predictions["instances"].scores.cpu().numpy()
    dets = []
    for (box, score) in zip(boxes, scores):
        t, l, b, r = box
        dets.append([t, l, b, r, score])
    if os.path.basename(args.config_file).split('_')[0] == 'mask':
        predict_masks = predictions["instances"].pred_masks
        masks = predict_masks.cpu().numpy()
        temp = np.zeros_like(im[:, :, 0])
        for i in range(len(predict_masks)):
            predict_mask_i = predict_masks[i]
            temp += np.array(predict_mask_i.to("cpu").numpy()).astype(np.uint8)
        region = im.copy()
        region[temp == 0] = 0
        region[temp!= 0] = im[temp != 0]
        return dets, np.array(masks), region
    return dets, [], []

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.merge_from_list(args.opts)
    return cfg

