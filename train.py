import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8改進/yunshud.yaml')   #模型配置文件
    # model.load('yolov8n.pt') # loading pretrain weights               #预训练权重
    model.train(data='yolov8改進/datasets/guandao/guandao.yaml',
                task='detect',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                #patience=15,
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )

#python train.py --yaml ultralytics/cfg/models/v8/yolov8-attention.yaml --info