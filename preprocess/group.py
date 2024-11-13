import os
import json
from copy import deepcopy
import shutil
from Core.DeviceType import DeviceTypeYolo

DeviceTypeSecondary = {}
for i in DeviceTypeYolo.values():
    for key, value in i.items():
        if key != 'default':
            DeviceTypeSecondary[key] = value


def point_inside_box(point, box):
    x, y = point
    x1, y1 = box[0]
    x2, y2 = box[2]
    return (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1)


def calculate_iou(box1, box2):
    x11, y11 = box1[0]
    x12, y12 = box1[2]
    x21, y21 = box2[0]
    x22, y22 = box2[2]

    xi1 = max(min(x11, x12), min(x21, x22))
    yi1 = max(min(y11, y12), min(y21, y22))
    xi2 = min(max(x11, x12), max(x21, x22))
    yi2 = min(max(y11, y12), max(y21, y22))

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = abs((x12 - x11) * (y12 - y11))
    box2_area = abs((x22 - x21) * (y22 - y21))

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0


_dir = r'../process/4'
save_dir = r'../process/5'

for file in os.listdir(_dir):
    if not file.endswith('.json'):
        shutil.copy(os.path.join(_dir, file), os.path.join(save_dir, file))
        continue

    with open(os.path.join(_dir, file), "r", encoding="utf-8") as f:
        data = json.load(f)

    result = deepcopy(data)
    boxes = []
    points = []
    _group = max(list([i['group_id'] if i['group_id'] else float('-Inf') for i in data['shapes']])) + 1
    _group = 0 if _group == float('-Inf') else _group

    # First pass: identify boxes and points
    for index, shape in enumerate(result['shapes']):
        if len(shape['points']) == 4:
            if shape['group_id'] is None:
                shape['group_id'] = _group
                _group += 1
            boxes.append((index, shape['points'], shape['label']))
        elif len(shape['points']) == 1:
            points.append((index, shape['points'][0]))

    # Check for overlapping boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i][1], boxes[j][1])
            if iou > 0.8 and boxes[i][2]:
                print(f"File '{file}' has overlapping boxes with IoU: {iou}")
                print(f"Box 1 - Index: {boxes[i][0]}, Label: {boxes[i][2]}")
                print(f"Box 2 - Index: {boxes[j][0]}, Label: {boxes[j][2]}")
                # input()

    # Second pass: assign points to boxes
    for point_index, point in points:
        for box_index, box, box_label in boxes:
            if point_inside_box(point, box):
                if result['shapes'][point_index]['label'] in DeviceTypeSecondary[box_label]:
                    result['shapes'][point_index]['group_id'] = result['shapes'][box_index]['group_id']
                    break
                else:
                    print(file, result['shapes'][point_index]['label'], box_label, DeviceTypeSecondary[box_label])
                    # input()

    # Save the modified JSON file
    with open(os.path.join(save_dir, file), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
