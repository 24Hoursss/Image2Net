import os

os.environ['WANDB_MODE'] = 'disabled'

from ultralytics import YOLO

# model = YOLO('yolo11m-pose.pt')
model = YOLO(r'runs/pose/train4/weights/last.pt')
# model = YOLO('yolov8.pt')

if __name__ == '__main__':
    model.train(data='data_pose.yaml', epochs=300, imgsz=1024, task='pose', device='0,1,2,4', batch=200, box=7.5,
                pose=15, fliplr=0.5, flipud=0.5, dropout=0.3)
