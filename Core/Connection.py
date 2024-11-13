import numpy as np
from Core.interface import Point, Node, iter_rename
from typing import List
import matplotlib.pyplot as plt
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


def point_inside_box(px, py, bx1, by1, bx2, by2):
    return bx1 <= px <= bx2 and by1 <= py <= by2


def cross(vx1, vy1, vx2, vy2):
    return vx1 * vy2 - vy1 * vx2


def line_intersects_line(x1, y1, x2, y2, x3, y3, x4, y4):
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3

    denominator = cross(dx1, dy1, dx2, dy2)

    if denominator == 0:
        return False

    dx3, dy3 = x3 - x1, y3 - y1
    t1 = cross(dx3, dy3, dx2, dy2) / denominator
    t2 = cross(dx3, dy3, dx1, dy1) / denominator

    return 0 <= t1 <= 1 and 0 <= t2 <= 1


class Connect:
    def __init__(self, corner: List[Point], binary_image, results: List[Node], mode):
        # block_type = ['cross', 'corner', 'switch', 'switch-3', 'vdd', 'resistor', 'resistor2', 'inductor', 'gnd',
        #               'capacitor', 'diode', 'pnp', 'npn', 'current', 'Voltage_1', 'Voltage_2', 'nmos-bulk', 'nmos',
        #               'pmos-bulk', 'pmos']
        block_type = ['cross', 'switch', 'switch-3', 'current', 'Voltage_1', 'Voltage_2', 'resistor',
                      'resistor2', 'capacitor', 'diode', 'gnd']
        self.block_type = block_type
        self.corner = corner
        self.binary_image = binary_image
        self.block = [i for i in results if i.type in block_type]
        self.block_all = results
        self.mode = mode
        self.overlap: List[Node] = self.find_overlaps()

    def find_overlaps(self) -> List[Node]:
        overlaps = []
        for i, node1 in enumerate(self.block_all):
            for j, node2 in enumerate(self.block_all):
                if i >= j or {node1.type, node2.type} == {'cross', 'corner'}:
                    continue
                overlap_area = self.calculate_overlap(node1.xyxy, node2.xyxy)
                if overlap_area:
                    overlaps.append(Node(xyxy=overlap_area))
        return overlaps

    @staticmethod
    def calculate_overlap(box1: List[float], box2: List[float]) -> List[float]:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            overlap_area = [inter_x_min - 1, inter_y_min - 1, inter_x_max + 1, inter_y_max + 1]
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            if inter_area / min(box1_area, box2_area) < 0.2:
                return overlap_area
        return []

    def is_straight_line_connected(self, point1, point2, threshold=5):
        x1, y1 = int(point1.x), int(point1.y)
        x2, y2 = int(point2.x), int(point2.y)

        if abs(x1 - x2) <= threshold or abs(y1 - y2) <= threshold:
            return self.check_line(point1, point2)
        return False

    def is_polyline_connected(self, point1, point2):
        if self.mode != 2:
            return False

        x1, y1 = int(point1.x), int(point1.y)
        x2, y2 = int(point2.x), int(point2.y)

        possible_corners = [
            (x1, y2),
            (x2, y1)
        ]

        for cx, cy in possible_corners:
            if 0 <= cx < self.binary_image.shape[1] and 0 <= cy < self.binary_image.shape[0]:
                corner_point = Point(cx, cy)

                # print(f"{self.check_line(point1, corner_point)=}")
                # self.visualize_line(point1, corner_point)
                #
                # print(f"{self.check_line(corner_point, point2)=}")
                # self.visualize_line(corner_point, point2)

                if (self.check_line(point1, corner_point) and
                        self.check_line(corner_point, point2)):
                    return True

        return False

    def visualize_line(self, point1, point2, color='red'):
        plt.imshow(self.binary_image, cmap='gray')
        plt.plot([point1.x, point2.x], [point1.y, point2.y], color=color)
        plt.scatter([point1.x, point2.x], [point1.y, point2.y], color=color)
        plt.show()

    def is_diagonal_line_connected(self, point1, point2):
        x1, y1 = int(point1.x), int(point1.y)
        x2, y2 = int(point2.x), int(point2.y)
        # self.visualize_line(point1, point2)

        if abs(x1 - x2) <= 3 or abs(y1 - y2) <= 3:
            return False

        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            # Check if the current point in the image is black (0)
            if self.binary_image[y1][x1] != 0:
                return False

            # Check if we've reached the end point
            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        for node in self.overlap:
            bx1, by1, bx2, by2 = node.xyxy
            flag1 = point_inside_box(int(point1.x), int(point1.y), bx1, by1, bx2, by2)
            flag2 = point_inside_box(int(point2.x), int(point2.y), bx1, by1, bx2, by2)
            if ((flag1 and flag2) or
                    (flag1 and point2.parent and point2.parent.type not in self.block_type) or
                    (flag2 and point1.parent and point1.parent.type not in self.block_type)):
                return True

        for node in self.block_all:
            if self.diagonal_line_intersects_box_single(int(point1.x), int(point1.y), int(point2.x), int(point2.y),
                                                        node):
                return False

        return True

    def check_line(self, point1, point2):
        x1, y1 = int(point1.x), int(point1.y)
        x2, y2 = int(point2.x), int(point2.y)
        # print(f"{x1=}, {x2=}, {y1=}, {y2=}")
        # self.visualize_line(point1, point2)

        if abs(x2 - x1) < abs(y2 - y1):
            _yMin, _yMax = max(0, min(y2, y1)), min(self.binary_image.shape[0], (max(y2, y1) + 1))
            _x = (x1 + x2) // 2
            vertical_line = self.binary_image[_yMin:_yMax, _x]
            # print(f"{vertical_line=}\n{np.all(vertical_line == 0)=}")
            if self.is_line(vertical_line):
                # print(f'{self.line_intersects_box(_x, _yMin, _x, _yMax)=}')
                return not self.line_intersects_box(_x, _yMin, _x, _yMax, point1.parent, point2.parent)
            return False
        else:
            _y = (y1 + y2) // 2
            _xMin, _xMax = max(0, min(x2, x1)), min(self.binary_image.shape[1], (max(x2, x1) + 1))
            horizontal_line = self.binary_image[_y, _xMin: _xMax]
            # print(f"{horizontal_line=}\n{np.all(horizontal_line == 0)=}")
            if self.is_line(horizontal_line):
                # print(f"{self.line_intersects_box(_xMin, _y, _xMax, _y)=}")
                return not self.line_intersects_box(_xMin, _y, _xMax, _y, point1.parent, point2.parent)
            return False

    @staticmethod
    def is_line(line):
        # 防止有些图质量太差
        non_zero_count = np.count_nonzero(line)

        if len(line) > 20:
            return non_zero_count <= 1
        else:
            return non_zero_count == 0
        # return np.all(line == 0)

    def line_intersects_box(self, x1, y1, x2, y2, parent1, parent2):
        for node in self.overlap:
            bx1, by1, bx2, by2 = node.xyxy
            flag1 = point_inside_box(x1, y1, bx1, by1, bx2, by2)
            flag2 = point_inside_box(x2, y2, bx1, by1, bx2, by2)
            if ((flag1 and flag2) or
                    (flag1 and parent2 and parent2.type not in self.block_type) or
                    (flag2 and parent1 and parent1.type not in self.block_type)):
                return False
        for node in self.block:
            if self.line_intersects_box_single(x1, y1, x2, y2, node):
                return True
        return False

    @staticmethod
    def line_intersects_box_single(x1, y1, x2, y2, node):
        nx1, ny1, nx2, ny2 = node.xyxy
        # print(f"{node.xyxy=}")
        # print(f"{x1=}, {y1=}, {x2=}, {y2=}")

        if y1 == y2:
            if ny1 <= y1 <= ny2:
                return (nx1 + 1 < x1 < nx2 - 1) or (nx1 + 1 < x2 < nx2 - 1) or (x1 <= nx1 and x2 >= nx2)

        if x1 == x2:
            if nx1 <= x1 <= nx2:
                return (ny1 + 1 < y1 < ny2 - 1) or (ny1 + 1 < y2 < ny2 - 1) or (y1 <= ny1 and y2 >= ny2)

        return False

    @staticmethod
    def diagonal_line_intersects_box_single(x1, y1, x2, y2, node):
        # print(node.xyxy)
        # print(x1, y1, x2, y2)
        nx1, ny1, nx2, ny2 = node.xyxy

        if point_inside_box(x1, y1, nx1, ny1, nx2, ny2) and point_inside_box(x2, y2, nx1, ny1, nx2, ny2):
            return True

        return (line_intersects_line(x1, y1, x2, y2, nx1 + 1, ny1 + 1, nx2 - 1, ny1 + 1) or
                line_intersects_line(x1, y1, x2, y2, nx2 - 1, ny1 + 1, nx2 - 1, ny2 - 1) or
                line_intersects_line(x1, y1, x2, y2, nx2 - 1, ny2 - 1, nx1 + 1, ny2 - 1) or
                line_intersects_line(x1, y1, x2, y2, nx1 + 1, ny2 - 1, nx1 + 1, ny1 + 1))

    def is_connected(self, point1, point2, threshold=5):
        # print(f"{self.is_straight_line_connected(point1, point2, threshold)=}\n"
        #       f"{self.is_diagonal_line_connected(point1, point2)=}\n"
        #       f"{self.is_polyline_connected(point1, point2)=}")
        # print(f"{self.is_polyline_connected(point1, point2)=}")
        # if point1.parent and point1.name != 'Body' and point2.parent and point2.name != 'Body' and point1.parent is point2.parent:
        #     return False
        return (self.is_straight_line_connected(point1, point2, threshold) or
                self.is_diagonal_line_connected(point1, point2))

    def is_directly_connected(self, point1, point2, threshold=5):
        if not self.is_connected(point1, point2, threshold):
            return False

        x1, y1 = int(point1.x), int(point1.y)
        x2, y2 = int(point2.x), int(point2.y)

        for corner in self.corner:
            if corner == point1 or corner == point2:
                continue

            cx, cy = int(corner.x), int(corner.y)
            line_vec = np.array([x2 - x1, y2 - y1])
            point_vec = np.array([cx - x1, cy - y1])
            line_len = np.dot(line_vec, line_vec)
            if line_len == 0:
                continue

            projection = np.dot(point_vec, line_vec) / line_len
            if 0 <= projection <= 1:
                closest_point = np.array([x1, y1]) + projection * line_vec
                distance_to_line = np.linalg.norm(np.array([cx, cy]) - closest_point)
                if distance_to_line < threshold:
                    return False

        return True

    def build_graph(self, threshold=5, parallel_mode='none'):
        corners = self.corner

        if parallel_mode == 'process':
            # with multiprocessing.Pool(processes=64) as pool:
            #     args = [(i, j, threshold) for i in range(len(corners)) for j in range(i + 1, len(corners))]
            #     pool.starmap(self.process_pair, args)
            with ProcessPoolExecutor(max_workers=64) as executor:
                futures = []
                for i in range(len(corners)):
                    for j in range(i + 1, len(corners)):
                        futures.append(executor.submit(self.process_pair, i, j, threshold))
                for future in futures:
                    future.result()

        elif parallel_mode == 'thread':
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = []
                for i in range(len(corners)):
                    for j in range(i + 1, len(corners)):
                        futures.append(executor.submit(self.process_pair, i, j, threshold))
                for future in futures:
                    future.result()

        else:
            for i in range(len(corners)):
                for j in range(i + 1, len(corners)):
                    self.process_pair(i, j, threshold)
        return self.find_non_corner_connections(corners, parallel_mode=parallel_mode)

    def process_pair(self, i, j, threshold):
        corners = self.corner
        if corners[j].connect is corners[i].connect:
            return
        elif corners[j] in corners[i].connect:
            corners[i].add_connects(corners[j].connect)
            corners[j].connect = corners[i].connect
            return
        if self.is_connected(corners[i], corners[j], threshold):
            corners[i].add_connect(corners[j])
            if corners[i].connect is not corners[j].connect:
                corners[i].add_connects(corners[j].connect)
            # corners[j].add_connect(corners[i])
            # corners[i].add_connects(corners[j].connect)
            # corners[j].add_connects(corners[i].connect)
            corners[j].connect = corners[i].connect

    # def process_pair(self, i, j, threshold):
    #     corners = self.corner
    #     point1, point2 = corners[i], corners[j]
    #
    #     if point1.connect is point2.connect:
    #         return
    #
    #     if self.is_connected(point1, point2, threshold):
    #         point1.add_connects(point2.connect)
    #
    #         queue = deque([point2])
    #         visited = {point2}
    #
    #         while queue:
    #             current = queue.popleft()
    #             old_connect = current.connect
    #             current.connect = point1.connect
    #
    #             for connected in old_connect:
    #                 if connected not in visited:
    #                     visited.add(connected)
    #                     queue.append(connected)
    #                     point1.add_connects(connected.connect)
    #                     connected.connect = point1.connect
    #
    #         point2.connect = point1.connect

    @staticmethod
    def find_non_corner_connections(points, parallel_mode='none'):
        non_corner_points = [p for p in points if p.name != 'corner']
        connections = {p: set() for p in non_corner_points}
        connect_index = 0
        calculated_connections = {}

        def bidirectional_bfs(start, target):
            if start == target:
                return True

            forward_queue = deque([start])
            backward_queue = deque([target])
            forward_visited = {start}
            backward_visited = {target}

            while forward_queue and backward_queue:
                if len(forward_queue) <= len(backward_queue):
                    current = forward_queue.popleft()
                    for neighbor in current.connect:
                        if neighbor in backward_visited:
                            return True
                        if neighbor not in forward_visited and (neighbor.name == 'corner' or neighbor == target):
                            forward_visited.add(neighbor)
                            forward_queue.append(neighbor)
                else:
                    current = backward_queue.popleft()
                    for neighbor in current.connect:
                        if neighbor in forward_visited:
                            return True
                        if neighbor not in backward_visited and (neighbor.name == 'corner' or neighbor == start):
                            backward_visited.add(neighbor)
                            backward_queue.append(neighbor)

            return False

        def process_pair(point1, point2):
            nonlocal connect_index
            pair_key = frozenset([point1, point2])
            if pair_key not in calculated_connections:
                calculated_connections[pair_key] = point1.connect is point2.connect or bidirectional_bfs(point1, point2)
                # calculated_connections[pair_key] = point1.connect is point2.connect

            if calculated_connections[pair_key]:
                connections[point1].add(point2)
                connections[point2].add(point1)
                if point1.outputName:
                    point2.outputName = point1.outputName
                elif point2.outputName:
                    point1.outputName = point2.outputName
                else:
                    point1.outputName = point2.outputName = f'n{connect_index}'
                    connect_index += 1

        if parallel_mode == 'thread':
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = []
                for i, point1 in enumerate(non_corner_points):
                    for point2 in non_corner_points[i + 1:]:
                        if not (point1.outputName and point2.outputName):
                            futures.append(executor.submit(process_pair, point1, point2))
                for future in futures:
                    future.result()  # Wait for all threads to complete
        else:
            for i, point1 in enumerate(non_corner_points):
                for point2 in non_corner_points[i + 1:]:
                    if not (point1.outputName and point2.outputName):
                        process_pair(point1, point2)

        return connections


if __name__ == '__main__':
    from Core.Detection import Detection
    from Core.Corner import Corner
    from Core.ImageManager import Manager
    from Core.ocr import OCR
    import pickle
    import warnings

    warnings.filterwarnings("ignore")

    # image_manager = Manager('../public/case_17/image.png')
    image_manager = Manager(r'C:\Users\PC\Desktop\public\images\007.png')
    # image_manager = Manager(r'C:\Users\PC\Desktop\Circuit-Dataset\images\79.png')

    detector = Detection(model_path='../checkpoints/best-tuned.onnx')

    results = detector.predict(image=image_manager.image, conf=0.6, iou=0.1, imgsz=1024, draw=False)
    line_width = detector.find_line_widths(results, image_manager.binary_image)
    results = detector.fix(results, image_manager.binary_image, move2closest=True, find_pose=True, is_draw=False,
                           image=image_manager.image)

    results, pose, mode = detector.results2custom(results, image_manager.binary_image)
    print(results)
    print(f"{mode=}, {line_width=}")

    ocr_detector = OCR()
    ocr_result = ocr_detector.predict(image_manager, scale=0.8, conf=0.8, is_draw=False)

    corner_detector = Corner(image_manager=image_manager)
    alg = ['Harris', 'Shi-Tomasi', 'FAST', 'ORB', 'AGAST', ]
    # alg = ['Hough', 'Harris', 'Shi-Tomasi', 'FAST', 'ORB', 'AGAST', 'SIFT', 'BRISK', 'MSER']
    # alg = ['Hough']
    corner = corner_detector.predict(draw=False, threshold=100, results=results + ocr_result, line_width=line_width,
                                     algorithm=alg)

    corner += pose
    # corner = [i for i in corner if not (i.name == 'corner' and i.nCorner == 4)]
    corner_detector.plot(corner)
    # pickle.dump(corner, open('../Example/corner.pickle', 'wb'))
    graph_builder = Connect(corner=corner, binary_image=image_manager.binary_image, results=results, mode=mode)
    # print(graph_builder.overlap)
    # for node in graph_builder.overlap:
    #     print(node.xyxy)
    #
    # print(graph_builder.is_connected(corner[195], corner[251]))
    # print(graph_builder.is_connected(corner[85], corner[254]))
    print(graph_builder.is_connected(corner[86], corner[192]))
    print(graph_builder.is_connected(corner[85], corner[192]))
    # print(corner[210].x, corner[210].y)
    # print(corner[242].x, corner[242].y)
    # print(graph_builder.overlap)
    # for node in graph_builder.overlap:
    #     print(node.xyxy)
    # print(graph_builder.is_connected(corner[44], Point(x=150, y=126), threshold=line_width * 2))
    # print(graph_builder.is_connected(corner[671], corner[697]))
    # print(graph_builder.is_connected(corner[66], corner[85]))
    # print(graph_builder.is_connected(corner[35], corner[50]))
    # print(corner[218], corner[219], corner[220], corner[221])
    # print(corner[218].connect, corner[219].connect, corner[220].connect, corner[221].connect)
    # print(np.linalg.norm([corner[218].x - corner[219].x, corner[218].y - corner[219].y]))
    # print(np.linalg.norm([corner[220].x - corner[221].x, corner[220].y - corner[221].y]))

    # print(graph_builder.is_connected(corner[221], corner[218], threshold=line_width))
    # print(graph_builder.build_graph(threshold=line_width, parallel_mode='thread'))
