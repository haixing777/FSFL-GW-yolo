import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import cv2
import os
if __name__ == '__main__':
    model = YOLO('best (1).pt') # select your model.pt path
    # model.predict(source='autodl-fs/yolov8改進/datasets/yunshu/images/val2017',
    #               imgsz=640,
    #               project='runs/detect',
    #               name='exp',
    #               save=True,
    #               # conf=0.2,
    #               # visualize=True # visualize model features maps
    #             )
    results = model.predict(
    source='autodl-fs/yolov8改進/datasets/yunshu/images/val2017/1698830609042_20231101172329A247_png.rf.f1aa64b164e11b444dc7ba0e1f145c7f.jpg',
    project='autodl-fs/yolov8改進/runs/detect',
    name='exp',
    conf=0.25,
    iou=0.45,
    save=True
)

# 分级预警和在图片上绘制信息
for i, result in enumerate(results):
    img = result.orig_img  # 获取原始图像
    names = model.names  # 获取类别名称
    for box in result.boxes:  # 遍历每个检测框
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框的坐标
        confidence = box.conf[0]  # 获取置信度
        cls_index = int(box.cls[0])  # 获取类别索引
        cls_name = names[cls_index]  # 根据索引获取类别名称
        area = (x2 - x1) * (y2 - y1)  # 计算检测框面积

        # 判断预警级别
        if area > 10000 and confidence > 0.5:
            warning = "serve"
            color = (0, 0, 255)  # 红色
        elif area > 5000:
            warning = "mid"
            color = (0, 255, 255)  # 黄色
        else:
            warning = "light"
            color = (0, 255, 0)  # 绿色

        # 绘制边界框和预警信息
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # 绘制边界框
        cv2.putText(
            img, 
            f"{cls_name} | {warning} | {confidence:.2f}",  # 文本内容
            (x1, y1 - 10),  # 位置（检测框左上角上方）
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            1,  # 字体大小
            color,  # 字体颜色
            2,  # 字体粗细
            cv2.LINE_AA  # 抗锯齿线
        )

    # 保存结果图像到指定路径
    save_path = os.path.join('autodl-fs/yolov8改進/runs/detect', f"result_{i + 1}.jpg")
    cv2.imwrite(save_path, img)  # 保存结果
    print(f"结果已保存到: {save_path}")
