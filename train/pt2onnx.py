from ultralytics import YOLO

# 载入pytorch模型
# model = YOLO(r'C:\Users\PC\Desktop\eda 2024\runs\detect\train12\weights\last.pt')
model = YOLO(r'../checkpoints/best.pt')

# 导出模型
model.export(format='onnx')