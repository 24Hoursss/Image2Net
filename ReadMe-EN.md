## Usage

`sh ~/submission/build.sh`

Please install package `dgl`, `pytorch` by yourself.

- You can use main.py directly

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

  Accepted TYPE for function 'solution': str(image path), byte(import by pickle), Pillow.Image<br>
  Function 'solution' will return result as 'dict' format

- Use main.ipynb for single detail image

- Use preprocess/image2net.py for visualization all images in folder 'load_dir' and save pictures to folder 'results'
   ```
   export PYTHONPATH=~/{Your Workspace}
   cd ~/{Your Workspace}/preprocess
   python image2net.py
   ```

## Dataset

https://github.com/24Hoursss/Circuit-Dataset

## Algorith flow chart

1. **Import**

2. **Initialize Models**
    - YOLO Model
    - GCN Model

3. **Initialize Image: Core.ImageManager.Manager**
    - Use PIL format input
    - Filter out gray background
    - Enhance black regions in the image
    - Convert to grayscale
    - Convert grayscale image to binary using Otsu's thresholding
    - Convert image to array named 'image_cv'

4. **YOLO Detection: Core.Detection.Detection**
    - Detect components and pose using YOLOv8m-pose
    - Enlarge all bounding boxes and specific component boxes
    - Determine circuit line width
    - Adjust pose points to nearest edge points and identify non-existent points
    - Convert YOLO results to custom format defined in Core.interface.Node & Core.interface.Point
    - Determine mode (will not be used in final code): 0 - no cross or corner, 1 (default) - no cross, 2 - existing
      cross.

5. **Corner Detection: Core.Corner.Corner**
    - Detect corners using supported
      algorithms: ['Hough', 'Harris', 'Shi-Tomasi', 'FAST', 'ORB', 'AGAST', 'SIFT', 'SURF', 'BRISK', 'MSER']
    - Filter corners within boxes
    - Cluster corners using DBSCAN
    - Validate corners: allow 90-degree polylines; filter straight lines and four-way corners

6. **Build Graph: Core.Connection.Connect**
    - Calculate connections between corners:
        - Straight line will be checked. Then check if this line inside Boxes which is not allowed to cross
        - Diagonal line will be checked. Then check if this line not inside all the Boxes
        - Overlap area is allowed to both Straight line & Diagonal line to be crossed
    - Build graph using bidirectional BFS
    - Output netlist

7. **Circuit Type Classification: Core.GCN.GCNClassifier**
    - Parse netlist into graph
    - Use GCN model for classification

8. **Output both netlist and circuit type**
    - Iter Rename & Delete component: 'VDD' & 'GND'

## Requirement

- numpy
- matplotlib
- torch
- dgl
- networkx
- dgl
- numba
- opencv_python
- Pillow
- easyocr (will not be used)
- scikit_learn
- ultralytics

## File description

1. checkpoints/*: Yolo & GCN models

2. Core/*.py:
    - Connection: Find connections of detected corner & component pin
    - Corner: Detect corner
    - Detection: Detect component using yolo-pose
    - DeviceType: Determine component names
    - ExceptionCase: Return specific case when catch exception
    - GCN: Train/Val circuit type
    - GCNModel: GCN & GCNHetero model
    - ImageManager: Process input image
    - interface: Determine Node & Point interface
    - ocr: Detect word in image (will not be used)
    - RunTime: Function 'cal_time' to calculate running time
    - Utils: Visualization functions

3. main.ipynb: Single visualization of whole work flow

4. GCNTrain/*.txt: GCNTrain Files

5. model/yolo/*: Yolo parameters

6. preprocess/*.py
    - group: Group component & key-points
    - image2net: Apply algorithm or process image classification to net which will be used to GCNTrain

7. results/*.png: Visualization the results

8. train/*.py:
    - GCNTrain: Train GCN
    - pt2onnx: Convert pt to onnx
    - tranYOLO: Train YOLO