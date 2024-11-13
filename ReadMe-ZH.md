## 使用方法

`sh ~/submission/build.sh`

请自行安装`dgl`和`pytorch`库。

- 可以直接使用 main.py

   ```
   python main.py
   ```

- `from main import solution`

   ```python
   import os
   from main import solution
   load_dir = r'C:\Users\PC\Desktop\public\images'
   save_dir = r'C:\Users\PC\Desktop\public\generate'
   for file in os.listdir(load_dir):
      save_name = file.replace('.png', '.txt')
      result = solution(os.path.join(load_dir, file))
      with open(os.path.join(save_dir, save_name), 'w') as f:
         f.write(str(result))
    ```

  函数 'solution' 可以接受的参数为： str(image path), byte(import by pickle), Pillow.Image<br>
  函数 'solution' 会以字典形式返回结果

- 使用main.ipynb运行单独图像及获取每部可视化效果

- 使用preprocess/image2net.py可视化文件夹 'load_dir' 中的所有图像，并将图片保存到文件夹 'results'
   ```
   export PYTHONPATH=~/{Your Workspace}
   cd ~/{Your Workspace}/preprocess
   python image2net.py
   ```
  
## 数据集

https://github.com/24Hoursss/Circuit-Dataset

## 算法流程

1. **导入**

2. **初始化模型**
    - YOLO模型
    - GCN模型

3. **初始化图像：Core.ImageManager.Manager**
    - 使用PIL格式输入
    - 过滤灰色背景
    - 增强图像中的黑色区域
    - 转换为灰度图像
    - 使用大津阈值法将灰度图像转换为二值图像
    - 将图像转换为名为'image_cv'的数组

4. **YOLO检测：Core.Detection.Detection**
    - 使用YOLOv8m-pose检测组件和姿态
    - 放大所有边界框和特定组件框
    - 确定电路线宽
    - 调整器件端口至边缘最近的点并在边缘随机选择作为yolo模型没预测出来的点
    - 将YOLO结果转换为Core.interface.Node和Core.interface.Point中定义的自定义格式
    - 确定模式（最终代码中不会使用）：0 - 无交叉或拐角，1（默认）- 无交叉，2 - 存在交叉

5. **拐角检测：Core.Corner.Corner**
    - 使用算法检测拐角：['Hough', 'Harris', 'Shi-Tomasi', 'FAST', 'ORB', 'AGAST', 'SIFT', 'SURF', 'BRISK', 'MSER']
    - 过滤YOLO检测出来的框内的拐角
    - 使用DBSCAN聚类拐角
    - 验证拐角：允许90度折线；过滤直线和四向拐角

6. **构建图：Core.Connection.Connect**
    - 计算拐角之间的连接：
        - 检查直线。然后检查该线是否跨越不允许跨越的框
        - 检查斜线。然后检查该线是否不跨越所有框
        - 允许在重叠区域交叉内通过直线和斜线
    - 使用双向BFS构建图
    - 输出网表

7. **电路类型分类：Core.GCN.GCNClassifier**
    - 将网表解析为图
    - 使用GCN模型进行分类（支持同构图和异构图两种形式）

8. **输出网表和电路类型**
    - 迭代重命名和删除组件：'VDD'和'GND'

## 流程图

![flow chart](flow%20chart.png)

## 需求

- numpy
- matplotlib
- torch
- dgl
- networkx
- dgl
- numba
- opencv_python
- Pillow
- easyocr（不会使用，用于OCR）
- scikit_learn
- ultralytics

## 文件描述

1. checkpoints/*: YOLO和GCN模型

2. Core/*.py:
    - Connection: 查找检测到的拐角和组件引脚的连接
    - Corner: 检测拐角
    - Detection: 使用yolo-pose检测组件
    - DeviceType: 确定组件名称
    - ExceptionCase: 捕获异常时返回特定情况
    - GCN: 训练/验证电路类型
    - GCNModel: GCN和GCNHetero模型
    - ImageManager: 处理输入图像
    - interface: 确定节点和点接口
    - ocr: 检测图像中的文字（不会使用）
    - RunTime: 函数'cal_time'用于计算运行时间
    - Utils: 可视化函数

3. main.ipynb: 整个工作流程的单个可视化

4. GCNTrain/*.txt: GCNTrain文件

5. model/yolo/*: YOLO参数

6. preprocess/*.py
    - group: 组合组件和关键点
    - image2net: 将算法应用或处理图像分类为用于GCNTrain的网

7. results/*.png: 结果的可视化

8. train/*.py:
    - GCNTrain: 训练GCN
    - pt2onnx: 将pt转换为onnx
    - tranYOLO: 训练YOLO