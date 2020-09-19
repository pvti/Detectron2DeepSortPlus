from detection_helpers import *
from models.experimental import attempt_load

class Yolov5Detector:
    def __init__(self, weights, img_size=(640, 640), conf_thresh=0.4, iou_thresh=0.5, device='cpu', agnostic_nms=False):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device)
        self.agnostic_nms = agnostic_nms
        self.model = attempt_load(weights, map_location=self.device)
    @torch.no_grad()
    def detect(self, im):
        img, __, __ = letterbox(im, new_shape=self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(np.expand_dims(img, 0))
        img = torch.from_numpy(img).float().to(self.device).div(255)

        dets = self.model(img)[0].cpu()
        dets = non_max_suppression(dets, self.conf_thresh, self.iou_thresh, self.agnostic_nms)
        det=dets[0]
        if det is not None and len(det):
            det = det[det[:, 0].argsort()]
            det[:,:4] = scale_coords(det[:, :4], img.shape[2:], im.shape).round()
        else:
            det = []
        return det

if __name__ == '__main__':
    import cv2
    detector = Yolov5Detector(weights='weights/best.pt', device='cuda:0')
    im = cv2.imread('/home/n/micand26/gt/GH010374_6_4944_6241_1/0000.png')
    detector.detect(im)
