import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best (1).pt')
    model.val(data='datasets/yunshu/yunshu.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )