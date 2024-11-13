from ultralytics import YOLO
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Core.interface import Node, Point
import numpy as np
import random
from copy import deepcopy
from Core.DeviceType import DeviceTypeSecondary
from math import ceil
import torch

os.environ['WANDB_MODE'] = 'disabled'


class Detection:
    def __init__(self, model_path, *args, **kwargs):
        self.model_path = model_path
        self.model = YOLO(model_path, task='pose', *args, **kwargs)
        self.line_width = 5

    def predict(self, img_path=None, image=None, conf=0.6, iou=0.1, draw=True, *args, **kwargs):
        self.line_width = 5
        image = Image.open(img_path).convert('RGB') if img_path else image
        results = self.model.predict(image, conf=0.3, iou=iou, *args, **kwargs)[0].cpu()
        # results.plot(show=True)

        results = deepcopy(results)

        # results = self.scale_boxes(results, 1.05, image)
        results = self.add_box(results, 3, image)

        parameters = {
            'cross': -3,
            'corner': -4
        }

        results = self.specific_enlarge(results, parameters, image)

        parameters = {
            'corner': 0.8
        }

        results = self.specific_enlarge(results, parameters, image)

        results = self.filter_conf(results, conf)

        if draw:
            self.plot(image, results)
        return results

    def specific_enlarge(self, results, parameters, image):
        # 对指定器件框进行放大
        for result in results:
            names_detection = result.names
            classes = result.boxes.cls
            name = names_detection[int(classes)]
            if name in parameters:
                if abs(parameters[name]) >= 1:
                    result.boxes.xyxy[0] = self.add_box_single(result.boxes.xyxy, parameters[name], image)
                else:
                    result.boxes.xyxy[0] = self.scale_boxes_single(result.boxes.xyxy, parameters[name], image)
        return results

    def fix_overlap(self, results):
        num_boxes = len(results)
        for i in range(num_boxes):
            names_detection = results[i].names
            name1 = names_detection[int(results[i].boxes.cls)]
            for j in range(i + 1, num_boxes):
                name2 = names_detection[int(results[j].boxes.cls)]
                if {name1, name2} == {'corner', 'cross'} and self.boxes_overlap(results[i].boxes.xyxy[0],
                                                                                results[j].boxes.xyxy[0]):
                    overlap_ratio = self.calculate_overlap_ratio(results[i].boxes.xyxy[0], results[j].boxes.xyxy[0])
                    if overlap_ratio <= 0.1:
                        self.adjust_boxes(results[i].boxes.xyxy[0], results[j].boxes.xyxy[0])

        return results

    @staticmethod
    def filter_conf(results, conf):
        filtered_boxes = []

        for i, result in enumerate(results):
            names_detection = result.names
            classes = result.boxes.cls
            confidence = result.boxes.conf
            name = names_detection[int(classes)]

            if name == 'corner' or float(confidence) >= conf:
                filtered_boxes.append(result)

        return filtered_boxes

    @staticmethod
    def boxes_overlap(box1, box2):
        return not (box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3])

    @staticmethod
    def calculate_overlap_ratio(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        overlap_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        smaller_area = min(box1_area, box2_area)
        return overlap_area / smaller_area

    @staticmethod
    def adjust_boxes(box1, box2):
        overlap_x = min(box1[2], box2[2]) - max(box1[0], box2[0])
        overlap_y = min(box1[3], box2[3]) - max(box1[1], box2[1])

        if overlap_x > overlap_y:
            if box1[1] < box2[1]:
                box1[3] -= overlap_y / 2 + 1
                box2[1] += overlap_y / 2 + 1
            else:
                box1[1] += overlap_y / 2 + 1
                box2[3] -= overlap_y / 2 + 1
        else:
            if box1[0] < box2[0]:
                box1[2] -= overlap_x / 2 + 1
                box2[0] += overlap_x / 2 + 1
            else:
                box1[0] += overlap_x / 2 + 1
                box2[2] -= overlap_x / 2 + 1

    @staticmethod
    def scale_boxes(results, scale_factor, image):
        width, height = image.size
        boxes = results.boxes.xyxy.clone()

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        new_sizes = (boxes[:, 2:] - boxes[:, :2]) * scale_factor
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, :2] = centers - new_sizes / 2
        new_boxes[:, 2:] = centers + new_sizes / 2
        new_boxes[:, [0, 2]] = torch.clamp(new_boxes[:, [0, 2]], 0, width)
        new_boxes[:, [1, 3]] = torch.clamp(new_boxes[:, [1, 3]], 0, height)
        results.boxes.data[:, :4] = new_boxes

        return results

    @staticmethod
    def add_box(results, length, image):
        width, height = image.size
        boxes = results.boxes.xyxy

        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, :2] = boxes[:, :2] - length / 2
        new_boxes[:, 2:] = boxes[:, 2:] + length / 2
        new_boxes[:, [0, 2]] = torch.clamp(new_boxes[:, [0, 2]], 0, width)
        new_boxes[:, [1, 3]] = torch.clamp(new_boxes[:, [1, 3]], 0, height)
        results.boxes.data[:, :4] = new_boxes

        return results

    @staticmethod
    def add_box_single(boxes, length, image):
        width, height = image.size
        new_boxes = boxes.clone()

        new_boxes[:, :2] = boxes[:, :2] - length / 2
        new_boxes[:, 2:] = boxes[:, 2:] + length / 2
        new_boxes[:, [0, 2]] = torch.clamp(new_boxes[:, [0, 2]], 0, width)
        new_boxes[:, [1, 3]] = torch.clamp(new_boxes[:, [1, 3]], 0, height)

        return new_boxes

    @staticmethod
    def scale_boxes_single(boxes, scale_factor, image):
        width, height = image.size
        new_boxes = boxes.clone()

        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        x_max = boxes[:, 2]
        y_max = boxes[:, 3]

        centers_x = (x_min + x_max) / 2
        centers_y = (y_min + y_max) / 2

        new_widths = (x_max - x_min) * scale_factor
        new_heights = (y_max - y_min) * scale_factor

        new_boxes[:, 0] = torch.clamp(centers_x - new_widths / 2, 0, width)
        new_boxes[:, 1] = torch.clamp(centers_y - new_heights / 2, 0, height)
        new_boxes[:, 2] = torch.clamp(centers_x + new_widths / 2, 0, width)
        new_boxes[:, 3] = torch.clamp(centers_y + new_heights / 2, 0, height)

        return new_boxes

    @staticmethod
    def adjust_keypoints(index1, index2, keypoints, x1, x2, y1, y2):
        dx, dy = keypoints[index1][0], keypoints[index1][1]
        sx, sy = keypoints[index2][0], keypoints[index2][1]
        none_exist_point = None

        if int(dx) != 0 and int(dy) != 0 and int(sx) != 0 and int(sy) != 0:
            if abs(dx - sx) > abs(dy - sy):
                dist_0_left = abs(dx - x1)
                dist_0_right = abs(dx - x2)
                dist_1_left = abs(sx - x1)
                dist_1_right = abs(sx - x2)

                if dist_0_left + dist_1_right < dist_1_left + dist_0_right:
                    keypoints[index1][0] = x1
                    keypoints[index2][0] = x2
                else:
                    keypoints[index1][0] = x2
                    keypoints[index2][0] = x1
            else:
                dist_0_top = abs(dy - y1)
                dist_0_bottom = abs(dy - y2)
                dist_1_top = abs(sy - y1)
                dist_1_bottom = abs(sy - y2)

                if dist_0_top + dist_1_bottom < dist_1_top + dist_0_bottom:
                    keypoints[index1][1] = y1
                    keypoints[index2][1] = y2
                else:
                    keypoints[index1][1] = y2
                    keypoints[index2][1] = y1
        else:
            if int(dx) == 0 or int(dy) == 0:
                none_exist_point = 0
                # Move the second point to the nearest short edge and place the first point on the opposite edge
                if abs(x2 - x1) < abs(y2 - y1):  # x-direction is shorter
                    nearest_y = y1 if abs(sy - y1) < abs(sy - y2) else y2
                    opposite_y = y2 if nearest_y == y1 else y1
                    keypoints[index1][1] = opposite_y
                    keypoints[index1][0] = sx
                    keypoints[index2][1] = nearest_y
                else:  # y-direction is shorter
                    nearest_x = x1 if abs(sx - x1) < abs(sx - x2) else x2
                    opposite_x = x2 if nearest_x == x1 else x1
                    keypoints[index1][0] = opposite_x
                    keypoints[index1][1] = sy
                    keypoints[index2][0] = nearest_x
            else:
                none_exist_point = 1
                # Move the first point to the nearest short edge and place the second point on the opposite edge
                if abs(x2 - x1) < abs(y2 - y1):  # x-direction is shorter
                    nearest_y = y1 if abs(dy - y1) < abs(dy - y2) else y2
                    opposite_y = y2 if nearest_y == y1 else y1
                    keypoints[index2][1] = opposite_y
                    keypoints[index2][0] = dx
                    keypoints[index1][1] = nearest_y
                else:  # y-direction is shorter
                    nearest_x = x1 if abs(dx - x1) < abs(dx - x2) else x2
                    opposite_x = x2 if nearest_x == x1 else x1
                    keypoints[index2][0] = opposite_x
                    keypoints[index2][1] = dy
                    keypoints[index1][0] = nearest_x

        return keypoints, none_exist_point

    def fix(self, results, binary_image, move2closest=True, find_pose=True, is_draw=False, image=None):
        for result in results:
            names_detection = result.names
            classes = result.boxes.cls
            name = names_detection[int(classes)]

            if name in ['switch', 'switch-3', 'cross', 'corner']:
                continue

            boxes = result.boxes.xyxy[0]
            x1, y1, x2, y2 = map(int, boxes)

            keypoints = result.keypoints.xy[0].clone()
            keypoints_conf = result.keypoints.conf

            # Test
            # if name == 'pmos' or name == 'nmos':
            #     keypoints[2][0] = 0
            #     keypoints[2][1] = 0
            # print("*" * 30)
            # print(f"{name=}, {keypoints=}")

            if move2closest or find_pose:
                endpoints = self.find_endpoints(x1, y1, x2, y2, binary_image)
                assigned_keypoints = set()
                endpoint_used = [False] * len(endpoints)

                if name.startswith(('pmos', 'nmos')):
                    keypoints, none_exist_point = self.adjust_keypoints(0, 1, keypoints, x1, x2, y1, y2)
                    if none_exist_point is not None:
                        keypoints_conf[0][none_exist_point] = 0.99
                if name.endswith('bulk'):
                    keypoints, none_exist_point = self.adjust_keypoints(2, 3, keypoints, x1, x2, y1, y2)
                    if none_exist_point is not None:
                        keypoints_conf[0][none_exist_point + 2] = 0.99

                if move2closest:
                    distances = []
                    for i, (x, y) in enumerate(keypoints):
                        if int(x) == 0 and int(y) == 0:
                            continue
                        for j, (ex, ey) in enumerate(endpoints):
                            distance = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                            distances.append((distance, i, j))

                    distances.sort()

                    for _, kp_index, ep_index in distances:
                        if kp_index in assigned_keypoints or endpoint_used[ep_index]:
                            continue
                        assigned_keypoints.add(kp_index)
                        endpoint_used[ep_index] = True

                        keypoints[kp_index][0] = endpoints[ep_index][0]
                        keypoints[kp_index][1] = endpoints[ep_index][1]

                if find_pose:
                    device_length = len(DeviceTypeSecondary[name])
                    lefts = set(range(device_length)) - assigned_keypoints
                    for left in lefts:
                        available_endpoints = [ep for i, ep in enumerate(endpoints) if not endpoint_used[i]]
                        if available_endpoints:
                            closest_endpoint = random.choice(available_endpoints)
                            # print(f"{left=}, {closest_endpoint=}")
                            keypoints[left][0] = closest_endpoint[0]
                            keypoints[left][1] = closest_endpoint[1]
                            endpoint_used[endpoints.index(closest_endpoint)] = True
                            keypoints_conf[0][left] = 0.99

                # # 可视化每个对象的关键点和端点
                # cropped_binary_image = binary_image[y1:y2 + 1, x1:x2 + 1]
                # plt.imshow(cropped_binary_image, cmap='gray')
                # # 绘制端点
                # for ex, ey in endpoints:
                #     plt.scatter(ex - x1, ey - y1, c='red', s=30, marker='x')  # 红色 'x' 标记端点
                #
                # # 绘制关键点
                # for kx, ky in keypoints:
                #     if int(kx) == 0 and int(ky) == 0:
                #         continue
                #     plt.scatter(kx - x1, ky - y1, c='blue', s=30, marker='o')  # 蓝色 'o' 标记关键点
                #
                # plt.title(f"Endpoints and Keypoints for {name}")
                # plt.show()

            result.keypoints.xy[0] = keypoints

        results = self.fix_overlap(results)

        if is_draw and image is not None:
            self.plot(image, results)

        return results

    def find_endpoints(self, x1, y1, x2, y2, binary_image):
        cropped_binary_image = binary_image[y1:y2 + 1, x1:x2 + 1]
        endpoints = []

        for edge in ['top', 'bottom', 'left', 'right']:
            if edge == 'top':
                line = cropped_binary_image[0, :]
            elif edge == 'bottom':
                line = cropped_binary_image[-1, :]
            elif edge == 'left':
                line = cropped_binary_image[:, 0]
            elif edge == 'right':
                line = cropped_binary_image[:, -1]

            black_segments = self.find_black_segments(line, ceil(self.line_width / 2))

            for start, end in black_segments:
                center = (start + end) // 2
                if edge in ['top', 'bottom']:
                    y = 0 if edge == 'top' else cropped_binary_image.shape[0] - 1
                    endpoints.append((center, y))
                else:
                    x = 0 if edge == 'left' else cropped_binary_image.shape[1] - 1
                    endpoints.append((x, center))

        # plt.imshow(cropped_binary_image, cmap='gray')
        # for ex, ey in endpoints:
        #     plt.scatter(ex, ey, c='red', s=30, marker='x')
        # plt.title("Endpoints on Cropped Binary Image")
        # plt.show()

        return [(x1 + ex, y1 + ey) for ex, ey in endpoints]

    @staticmethod
    def find_black_segments(line, threshold_length):
        segments = []
        start = None

        for i, pixel in enumerate(line):
            if pixel == 0 and start is None:
                start = i
            elif pixel == 255 and start is not None:
                if i - start >= threshold_length:
                    segments.append((start, i - 1))
                start = None

        if start is not None and len(line) - start >= threshold_length:
            segments.append((start, len(line) - 1))

        return segments

    @staticmethod
    def find_closest_endpoint(point, endpoints):
        x, y = point
        closest_endpoint = None
        min_distance = float('inf')
        for ex, ey in endpoints:
            distance = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_endpoint = (ex, ey)
        return closest_endpoint

    def process_cross(self, result, binary_image):
        boxes = result.boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, boxes)
        node = Node(type='cross', xyxy=[x1, y1, x2, y2])

        endpoints = np.array(self.find_endpoints(x1, y1, x2, y2, binary_image))

        # plt.imshow(binary_image, cmap='gray')
        # for ex, ey in endpoints:
        #     plt.scatter(ex, ey, c='red', s=30, marker='x')
        # plt.title("Endpoints on Cropped Binary Image")
        # plt.show()

        if len(endpoints) < 4:
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)

            point1 = Point(x=(min_x + max_x) // 2, y=min_y, name='corner')
            point2 = Point(x=(min_x + max_x) // 2, y=max_y, name='corner')
            point3 = Point(x=min_x, y=(min_y + max_y) // 2, name='corner')
            point4 = Point(x=max_x, y=(min_y + max_y) // 2, name='corner')
        else:
            while len(endpoints) > 4:
                distances = np.array([
                    (i, j, np.linalg.norm(endpoints[i] - endpoints[j]))
                    for i in range(len(endpoints))
                    for j in range(i + 1, len(endpoints))
                ])

                min_dist_idx = np.argmin(distances[:, 2])
                i, j = int(distances[min_dist_idx][0]), int(distances[min_dist_idx][1])

                endpoints = np.delete(endpoints, j, axis=0)

            distances = np.array([
                (i, j, np.linalg.norm(endpoints[i] - endpoints[j]))
                for i in range(len(endpoints))
                for j in range(i + 1, len(endpoints))
            ])
            # print(f"{distances=}")
            max_dist_idx = np.argmax(distances[:, 2])
            # print(f"{max_dist_idx=}")
            i, j = int(distances[max_dist_idx][0]), int(distances[max_dist_idx][1])

            point1 = Point(x=endpoints[i][0], y=endpoints[i][1], name='corner')
            point2 = Point(x=endpoints[j][0], y=endpoints[j][1], name='corner')
            remaining_points = [k for k in range(len(endpoints)) if k != i and k != j]
            point3 = Point(x=endpoints[remaining_points[0]][0], y=endpoints[remaining_points[0]][1], name='corner')
            point4 = Point(x=endpoints[remaining_points[1]][0], y=endpoints[remaining_points[1]][1], name='corner')

        point1.nCorner = 3
        point2.nCorner = 3
        point3.nCorner = 3
        point4.nCorner = 3
        point1.add_connect(point2)
        point2.add_connect(point1)
        point3.add_connect(point4)
        point4.add_connect(point3)

        pose_result = [point1, point2, point3, point4]

        return [node], pose_result

    def process_switch(self, result, binary_image):
        return [], []

    def process_switch3(self, result, binary_image):
        return [], []

    def process_corner(self, result, binary_image):
        boxes = result.boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, boxes)

        points = []
        point = Point(x=(x1 + x2) // 2, y=(y1 + y2) // 2, name='corner')
        point.nCorner = 3
        points.append(point)

        endpoints = self.find_endpoints(x1, y1, x2, y2, binary_image)
        for x, y in endpoints:
            now = Point(x=x, y=y, name='corner')
            now.nCorner = 3
            for point in points:
                now.add_connect(point)
                point.add_connect(now)
            points.append(now)

        return [Node(xyxy=[x1, y1, x2, y2], type='corner')], points

    def results2custom(self, results, binary_image):
        custom_results = []
        pose_results = []
        process_methods = {
            'cross': self.process_cross,
            'switch': self.process_switch,
            'switch-3': self.process_switch3,
            'corner': self.process_corner
        }
        # 2 : 表示既没有 cross 也没有 corner
        # 1 default: 表示没有cross，即电路线都表示相连
        # 0 : 表示存在 cross，即电路线只有直线相连，cross跨导线
        mode = 1
        corner_flag = False

        for result in results:
            names_detection = result.names
            classes = result.boxes.cls
            class_index = int(classes)
            class_name = names_detection[class_index]

            if class_name == 'cross':
                mode = 0
            elif class_name == 'corner':
                corner_flag = True

            if class_name in process_methods:
                result_custom, result_pose = process_methods[class_name](result, binary_image)
                pose_results += result_pose
                custom_results += result_custom
                continue

            boxes = result.boxes.xyxy[0]
            keypoints = result.keypoints
            x1, y1, x2, y2 = boxes

            node = Node(type=class_name, xyxy=[x1, y1, x2, y2])

            poses = keypoints.xy[0]
            pose_names = DeviceTypeSecondary[class_name]
            length = len(pose_names)

            for i, (x, y) in enumerate(poses):
                if int(x) == 0 and int(y) == 0 or i > length - 1:
                    continue
                point_name = pose_names[i]
                point = Point(x=x, y=y, name=point_name, parent=node)
                pose_results.append(point)
                node.add_child(point)

            custom_results.append(node)

        if mode == 1 and not corner_flag:
            mode = 2

        return custom_results, pose_results, mode

    def plot(self, image, results):
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for result in results:
            names_detection = result.names
            boxes = result.boxes.xyxy[0]
            detection_conf = result.boxes.conf[0]
            classes = result.boxes.cls
            keypoints = result.keypoints
            name = names_detection[int(classes)]

            x1, y1, x2, y2 = boxes
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # plt.text(x1, y1, f'{name}: {detection_conf:.2f}', color='white', fontsize=8,
            #          bbox=dict(facecolor='red', alpha=0.5))

            poses = keypoints.xy[0]
            pose_name = DeviceTypeSecondary[name]
            pose_conf = keypoints.conf[0]
            for pose_index in range(len(pose_name)):
                x, y = poses[pose_index, :]
                if int(x) == 0 and int(y) == 0:
                    continue
                plt.scatter(x, y, c='blue', s=10)
                # plt.text(x, y, f'{pose_name[pose_index]}: {pose_conf[pose_index]:.2f}', color='blue', fontsize=8)

        plt.axis('off')
        plt.show()

    def find_line_widths(self, results, binary_image):
        widths = []

        for result in results:
            names_detection = result.names
            classes = result.boxes.cls
            class_index = int(classes)
            class_name = names_detection[class_index]
            if class_name in ['vdd', 'switch', 'switch-3', 'cross']:
                continue
            boxes = result.boxes.xyxy[0]
            x1, y1, x2, y2 = map(int, boxes)

            cropped_binary_image = binary_image[y1:y2 + 1, x1:x2 + 1]

            # plt.imshow(cropped_binary_image, cmap='gray')
            # plt.show()

            for edge in ['top', 'bottom', 'left', 'right']:
                if edge == 'top':
                    line = cropped_binary_image[0, :]
                elif edge == 'bottom':
                    line = cropped_binary_image[-1, :]
                elif edge == 'left':
                    line = cropped_binary_image[:, 0]
                elif edge == 'right':
                    line = cropped_binary_image[:, -1]

                black_segments = self.find_black_segments(line, 1)
                widths += [_line[1] - _line[0] + 1 for _line in black_segments]

        # print(widths)
        if widths:
            widths = list(filter(lambda x: x < 10, widths))
            widths.sort()
            low_idx = int(len(widths) * 0.025)
            high_idx = int(len(widths) * 0.975)
            if high_idx <= low_idx:
                high_idx = low_idx + 1
            filtered_widths = widths[low_idx:high_idx]
            if filtered_widths:
                average_width = int(np.mean(filtered_widths))
                self.line_width = min(max(average_width, 1), 15)
            else:
                self.line_width = 3
        else:
            self.line_width = 3

        #     self.line_width = average_width
        #     return average_width

        return self.line_width

    @staticmethod
    def calculate_line_widths(line):
        widths = []
        start = None

        for i, pixel in enumerate(line):
            if pixel == 0 and start is None:
                start = i
            elif pixel == 1 and start is not None:
                widths.append(i - start)
                start = None

        if start is not None:
            widths.append(len(line) - start)

        return widths


if __name__ == '__main__':
    from Core.Connection import *
    from Core.ImageManager import Manager
    from Core.Corner import Corner

    # image_manager = Manager(f'../public/case_12/image.png')
    image_manager = Manager(r'C:\Users\PC\Desktop\public\images\036.png')
    # image_manager = Manager(r'C:\Users\PC\Desktop\Circuit-Dataset\images\830.png')
    # image_manager = Manager('../cases/image2.png')

    # plt.imshow(image_manager.binary_image, cmap='gray')
    # plt.show()

    detector = Detection(model_path='../checkpoints/best-tuned.onnx')
    #
    results = detector.predict(image=image_manager.image, conf=0.5, iou=0.1, draw=True, imgsz=1024)
    line_width = detector.find_line_widths(results, image_manager.binary_image)
    print(f"{line_width=}")

    results = detector.fix(results, image_manager.binary_image, move2closest=True, find_pose=True, is_draw=True,
                           image=image_manager.image)
    results, pose, mode = detector.results2custom(results, image_manager.binary_image)

    corner_detector = Corner(image_manager=image_manager)
    corner_detector.plot(pose)
