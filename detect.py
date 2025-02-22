import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import cv2
import os
if __name__ == '__main__':
    model = YOLO('best (1).pt') # select your model.pt path
    # model.predict(source='autodl-fs/yolov8/datasets/yunshu/images/val2017',
    #               imgsz=640,
    #               project='runs/detect',
    #               name='exp',
    #               save=True,
    #               # conf=0.2,
    #               # visualize=True # visualize model features maps
    #             )
    results = model.predict(
    source='autodl-fs/yolov8/datasets/yunshu/images/val2017/1698830609042_20231101172329A247_png.rf.f1aa64b164e11b444dc7ba0e1f145c7f.jpg',
    project='autodl-fs/yolov8/runs/detect',
    name='exp',
    conf=0.25,
    iou=0.45,
    save=True
)

# Graduated alerts and plotting information on pictures
for i, result in enumerate(results):
    img = result.orig_img  # Acquiring the original image
    names = model.names  # Get category name
    for box in result.boxes:  # Iterate through each detection box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates of the detection box
        confidence = box.conf[0]  # Getting Confidence
        cls_index = int(box.cls[0])  # Get category index
        cls_name = names[cls_index]  # Get category name by index
        area = (x2 - x1) * (y2 - y1)  # Calculate the detection frame area

        # Determining the level of warning
        if area > 10000 and confidence > 0.5:
            warning = "serve"
            color = (0, 0, 255)  # red
        elif area > 5000:
            warning = "mid"
            color = (0, 255, 255)  # yellow
        else:
            warning = "light"
            color = (0, 255, 0)  # green

        # Mapping of bounding boxes and early warning messages
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Drawing the bounding box
        cv2.putText(
            img, 
            f"{cls_name} | {warning} | {confidence:.2f}",  # Text content
            (x1, y1 - 10),  # Position (above the upper left corner of the detection frame)
            cv2.FONT_HERSHEY_SIMPLEX,  # calligraphic style
            1,  # font size
            color,  # font color
            2,  # thickness of font
            cv2.LINE_AA  # antialiasing
        )

    # Save the resultant image to the specified path
    save_path = os.path.join('autodl-fs/yolov8/runs/detect', f"result_{i + 1}.jpg")
    cv2.imwrite(save_path, img)  # Save results
    print(f"The results have been saved to: {save_path}")
