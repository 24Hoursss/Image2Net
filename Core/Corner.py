import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN
from Core.interface import Point
from numba import jit
from Core.RunTime import cal_time
from Core.ImageManager import Manager
from typing import Union, List


class Corner:
    def __init__(self, image_manager: Manager):
        self.image = image_manager.image
        self.binary_image = image_manager.binary_image
        self.image_cv = image_manager.image_cv
        self.gray = image_manager.gray_image
        self.gaus = image_manager.gaus
        self.results = []

    # @cal_time('s')
    def predict(self, algorithm: Union[str, List[str]] = 'Hough', draw=True, results=[], line_width=5, *args, **kwargs):
        self.results = results
        corners = []
        if isinstance(algorithm, str):
            algorithm = [algorithm]

        for alg_loop in algorithm:
            corners += self.run_algorithm(alg_loop, *args, **kwargs)

        corners = self.filter(corners, results)
        if not corners:
            return []
        # plt.figure()
        # plt.imshow(self.binary_image, cmap='gray')
        # for corner in corners:
        #     plt.plot(corner[0], corner[1], 'r.', markersize=1)
        # plt.show()

        labels = self.cluster(corners, line_width)
        corners = self.valid(corners, labels, line_width)

        # 删除过多的点（一般在文字处），不考虑时间的话可以不要这一步
        # corners = self.re_valid(corners, self.re_cluster(corners, line_width))

        corners = self.results2custom(corners)

        if draw:
            self.plot(corners)

        return corners

    def run_algorithm(self, algorithm, *args, **kwargs):
        if algorithm == 'Hough':
            return self.hough_detection(*args, **kwargs)
        elif algorithm == 'Harris':
            return self.harris_corner_detection()
        elif algorithm == 'Shi-Tomasi':
            return self.shi_tomasi_corner_detection()
        elif algorithm == 'FAST':
            return self.fast_corner_detection()
        elif algorithm == 'ORB':
            return self.orb_corner_detection()
        elif algorithm == 'AGAST':
            return self.agast_corner_detection()
        elif algorithm == 'SIFT':
            return self.sift_corner_detection()
        elif algorithm == 'SURF':
            return self.surf_corner_detection()
        elif algorithm == 'BRISK':
            return self.brisk_corner_detection()
        elif algorithm == 'MSER':
            return self.mser_corner_detection()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def hough_detection(self, threshold=100):
        gaus = self.gaus
        v = np.median(gaus)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(gaus, lower, upper, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        # plotHoughLines(self.image_cv, lines)

        height, width = self.image_cv.shape[:2]

        intersections = []

        if lines is not None:
            for i in range(len(lines)):
                rho1, theta1 = lines[i][0]
                for j in range(i + 1, len(lines)):
                    rho2, theta2 = lines[j][0]
                    inter = self.intersection((rho1, theta1), (rho2, theta2))
                    if inter:
                        x, y = inter
                        if 0 < x < width and 0 < y < height:
                            intersections.append(inter)
        return intersections

    @staticmethod
    def cluster(intersections, line_width):
        # eps 越小越精确，但时间耗时越大
        # height, width = self.binary_image.shape
        # eps = int(height * width / 300000)
        if line_width <= 3:
            eps = 1.5
        elif line_width <= 5:
            eps = 2
        else:
            # eps = 3
            eps = line_width / 2.5
        # eps = line_width
        # eps = 1
        # print(f"{eps=}")

        # Use DBSCAN to cluster points
        clustering = DBSCAN(eps=eps, min_samples=1).fit(intersections)

        return clustering.labels_

    # type 越大越精确，但时间耗时越大
    def valid(self, intersections, labels, line_width, type=-1):
        height, width = self.binary_image.shape
        valid_intersections = []
        added_points = set()
        black_threshold = line_width * 4

        # state = ['corner2', 'line', 'corner3', 'corner4', 'nan', 'corner1']
        # plt.figure()
        # plt.imshow(self.binary_image, cmap='gray')
        for label in set(labels) - {-1}:
            cluster_points = [intersections[i] for i in range(len(intersections)) if labels[i] == label]
            # density = len(cluster_points)

            # 计算质心
            x, y = map(int, np.mean(cluster_points, axis=0))
            # if self.binary_image[y][x] == 255:
            #     continue

            # plt.plot(x, y, 'r.', markersize=1)

            # 根据线宽选择搜索范围
            if line_width <= 5:
                dx_range = range(-2, 3)
            else:
                dx_range = range(-line_width // 2, line_width // 2 + 1)

            # 根据密度选择搜索范围
            # if density == 1:
            #     dx_range = [0]
            # else:
            #     dx_range = range(-2, 3)

            # dx_range = range(-2, 3)

            pool_points = [
                (x + dx, y + dy)
                for dx in [dx_range[0], 0, dx_range[-1]]
                for dy in [dx_range[0], 0, dx_range[-1]]
                if 0 <= x + dx < width and 0 <= y + dy < height
                   and self.binary_image[y + dy][x + dx] != 255 and not self.corner_in_box(x + dx, y + dy)
            ]
            flag = False
            for ex, ey in pool_points:
                if self.count_connect(ex, ey, black_threshold) >= 4:
                    flag = True
                    break
            if flag:
                continue

            pool_points = [
                (x + dx, y + dy)
                for dx in dx_range
                for dy in dx_range
                if 0 <= x + dx < width and 0 <= y + dy < height
                   and self.binary_image[y + dy][x + dx] != 255 and not self.corner_in_box(x + dx, y + dy)
            ]
            if not pool_points:
                continue

            # 加入全部可选点，并判断是否存在corner4，首末中三点
            if type == -1:
                temp = []
                for index, point in enumerate(pool_points):
                    pool_judge, _count = self.is_surrounded_by_black(point[0], point[1], black_threshold)
                    if pool_judge:
                        temp.append((point, _count))
                    if _count == 4:
                        temp = []
                        break

                length = len(temp)
                first_valid = None
                last_valid = None
                mid_valid = None
                if length == 1:
                    first_valid = temp[0]
                elif length == 2:
                    first_valid = temp[0]
                    last_valid = temp[-1]
                elif length >= 3:
                    first_valid = temp[0]
                    last_valid = temp[-1]
                    if line_width >= 5:
                        mid_valid = temp[length // 2]

                if first_valid and first_valid[0] not in added_points:
                    valid_intersections.append([first_valid[0][0], first_valid[0][1], first_valid[1]])
                    added_points.add(first_valid[0])

                if last_valid and last_valid[0] not in added_points:
                    valid_intersections.append([last_valid[0][0], last_valid[0][1], last_valid[1]])
                    added_points.add(last_valid[0])

                if mid_valid and mid_valid[0] not in added_points:
                    valid_intersections.append([mid_valid[0][0], mid_valid[0][1], mid_valid[1]])
                    added_points.add(mid_valid[0])

            # 加入全部可选点，并判断是否存在corner4，中间两点
            if type == -2:
                temp = []
                for index, point in enumerate(pool_points):
                    pool_judge, _count = self.is_surrounded_by_black(point[0], point[1], black_threshold)
                    if pool_judge:
                        temp.append((point, _count))
                    if _count == 4:
                        temp = []
                        break

                length = len(temp)
                first_valid = None
                last_valid = None
                if length == 1:
                    first_valid = temp[0]
                elif length >= 2:
                    first_valid = temp[length // 2 - 1]
                    last_valid = temp[length // 2]

                if first_valid and first_valid[0] not in added_points:
                    valid_intersections.append([first_valid[0][0], first_valid[0][1], first_valid[1]])
                    added_points.add(first_valid[0])

                if last_valid and last_valid[0] not in added_points:
                    valid_intersections.append([last_valid[0][0], last_valid[0][1], last_valid[1]])
                    added_points.add(last_valid[0])

            # 加入第一个可选点
            elif type == 0:
                for point in pool_points:
                    pool_judge, _count = self.is_surrounded_by_black(point[0], point[1], black_threshold)
                    if _count == 4:
                        break
                    if pool_judge:
                        if point not in added_points:
                            valid_intersections.append([point[0], point[1], _count])
                            added_points.add(point)
                        break

            # 加入首位和末位两个可选点
            elif type == 1:
                length = len(pool_points)
                left = 0
                right = length - 1
                first_valid = None
                last_valid = None
                _count_flag = False

                while left < length:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[left][0],
                                                                     pool_points[left][1],
                                                                     black_threshold)
                    if _count == 4:
                        _count_flag = True
                        break
                    if pool_judge and first_valid is None:
                        first_valid = (pool_points[left], _count)
                        break

                    left += 1

                while right > left and not _count_flag:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[right][0],
                                                                     pool_points[right][1],
                                                                     black_threshold)
                    if _count == 4:
                        first_valid = None
                        break
                    if pool_judge and last_valid is None:
                        last_valid = (pool_points[right], _count)
                        break

                    right -= 1

                if first_valid and first_valid[0] not in added_points:
                    valid_intersections.append([first_valid[0][0], first_valid[0][1], first_valid[1]])
                    added_points.add(first_valid[0])

                if last_valid and last_valid[0] not in added_points:
                    valid_intersections.append([last_valid[0][0], last_valid[0][1], last_valid[1]])
                    added_points.add(last_valid[0])

            # 加入首位、末位、中间三个可选点
            elif type == 2:
                length = len(pool_points)
                left = 0
                right = length - 1
                first_valid = None
                last_valid = None
                mid = length // 2
                mid_valid = None
                _count_flag = False

                while left < length:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[left][0],
                                                                     pool_points[left][1],
                                                                     black_threshold)
                    if _count == 4:
                        _count_flag = True
                        break
                    if pool_judge and first_valid is None:
                        first_valid = (pool_points[left], _count)
                        break

                    left += 1

                while right > left and not _count_flag:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[right][0],
                                                                     pool_points[right][1],
                                                                     black_threshold)
                    if _count == 4:
                        first_valid = None
                        _count_flag = True
                        break
                    if pool_judge and last_valid is None:
                        last_valid = (pool_points[right], _count)
                        break

                    right -= 1

                added = 1
                flag = 1
                while left < mid < right and not _count_flag:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[mid][0],
                                                                     pool_points[mid][1],
                                                                     black_threshold)
                    if _count == 4:
                        first_valid = None
                        last_valid = None
                        break
                    if pool_judge and mid_valid is None:
                        mid_valid = (pool_points[mid], _count)
                        break

                    mid += flag * added
                    added += 1
                    flag = -flag

                if first_valid and first_valid[0] not in added_points:
                    valid_intersections.append([first_valid[0][0], first_valid[0][1], first_valid[1]])
                    added_points.add(first_valid[0])

                if last_valid and last_valid[0] not in added_points:
                    valid_intersections.append([last_valid[0][0], last_valid[0][1], last_valid[1]])
                    added_points.add(last_valid[0])

                if mid_valid and mid_valid[0] not in added_points:
                    valid_intersections.append([mid_valid[0][0], mid_valid[0][1], mid_valid[1]])
                    added_points.add(mid_valid[0])

            # 加入全部可选点
            elif type == 3:
                temp = []
                for index, point in enumerate(pool_points):
                    pool_judge, _count = self.is_surrounded_by_black(point[0], point[1], black_threshold)
                    if _count == 4:
                        temp = []
                        break
                    if pool_judge and point not in added_points:
                        temp.append([point[0], point[1], _count])

                valid_intersections += temp
                for point in added_points:
                    added_points.add(point)

            # 加入中间开始首位和末位两个可选点
            elif type == 4:
                length = len(pool_points)
                mid = length // 2
                left = mid - 1
                right = mid
                first_valid = None
                last_valid = None
                _count_flag = False

                while 0 <= left:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[left][0],
                                                                     pool_points[left][1],
                                                                     black_threshold)
                    if _count == 4:
                        _count_flag = True
                        break
                    if pool_judge and first_valid is None:
                        first_valid = (pool_points[left], _count)
                        break

                    left -= 1

                while right < length and not _count_flag:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[right][0],
                                                                     pool_points[right][1],
                                                                     black_threshold)
                    if _count == 4:
                        first_valid = None
                        break
                    if pool_judge and last_valid is None:
                        last_valid = (pool_points[right], _count)
                        break

                    right += 1

                if first_valid and first_valid[0] not in added_points:
                    valid_intersections.append([first_valid[0][0], first_valid[0][1], first_valid[1]])
                    added_points.add(first_valid[0])

                if last_valid and last_valid[0] not in added_points:
                    valid_intersections.append([last_valid[0][0], last_valid[0][1], last_valid[1]])
                    added_points.add(last_valid[0])

            # 中间开始加入第一个可选点
            elif type == 5:
                length = len(pool_points)
                mid = length // 2
                added = 1
                flag = 1
                while 0 <= mid < length:
                    pool_judge, _count = self.is_surrounded_by_black(pool_points[mid][0], pool_points[mid][1],
                                                                     black_threshold)
                    if _count == 4:
                        break
                    if pool_judge:
                        if pool_points[mid] not in added_points:
                            valid_intersections.append([pool_points[mid][0], pool_points[mid][1], _count])
                            added_points.add(pool_points[mid])
                        break
                    mid += flag * added
                    added += 1
                    flag = -flag

        return valid_intersections

    @staticmethod
    def re_cluster(intersections, line_width):
        clustering = DBSCAN(eps=line_width * 2, min_samples=1).fit(intersections)

        return clustering.labels_

    @staticmethod
    def re_valid(intersections, labels):
        for label in set(labels):
            if label == -1:
                continue
            cluster_points = [intersections[i] for i in range(len(intersections)) if labels[i] == label]
            density = len(cluster_points)
            if density > 20:
                for point in cluster_points:
                    intersections.remove(point)
        return intersections

    @staticmethod
    def results2custom(results):
        corner = []
        for x, y, _count in results:
            point = Point(x=x, y=y, name='corner')
            point.nCorner = _count
            corner.append(point)
        return corner

    def plot(self, corners):
        image_cv_copy = self.image_cv.copy()

        for index, point in enumerate(corners):
            # if index in [241, 242, 243]:
            #     continue
            cv2.circle(image_cv_copy, (int(point.x), int(point.y)), 5, (255, 0, 0), -1)
            cv2.putText(image_cv_copy, str(index), (int(point.x) + 5, int(point.y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        image_rgb = cv2.cvtColor(image_cv_copy, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

        # def resize_image(image, scale_percent):
        #     width = int(image.shape[1] * scale_percent / 100)
        #     height = int(image.shape[0] * scale_percent / 100)
        #     dim = (width, height)
        #     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        #
        # # cv2.namedWindow('Result Image', cv2.WINDOW_NORMAL)
        # cv2.imshow('Result Image', image_cv_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def is_surrounded_by_black(self, x, y, margin):
        height, width = self.binary_image.shape
        bx, nx = max(x - margin, 0), min(x + margin, width - 1)
        by, ny = max(y - margin, 0), min(y + margin, height - 1)

        # 上下左右是否是完整的一整条线
        left = not self.binary_image[y, bx:x].any()
        right = not self.binary_image[y, x:nx].any()
        top = not self.binary_image[by:y, x].any()
        bottom = not self.binary_image[y:ny, x].any()
        _count = [left, right, top, bottom].count(True)
        if _count == 0:
            return False, 0
        if _count in [1, 3]:
            return True, _count
        if _count == 4:
            return False, 4
        horizon = left and right
        vertical = top and bottom
        if horizon and not top and not bottom or vertical and not left and not right:
            return False, 0
        return True, 2

    def count_connect(self, x, y, margin):
        height, width = self.binary_image.shape
        bx, nx = max(x - margin, 0), min(x + margin, width - 1)
        by, ny = max(y - margin, 0), min(y + margin, height - 1)
        total = 0

        for ex, ey in self.find_endpoints(bx, by, nx, ny, margin // 8):
            if self.is_diagonal_line_connected(x, y, ex, ey):
                total += 1

        return total

    def find_endpoints(self, x1, y1, x2, y2, threshold):
        cropped_binary_image = self.binary_image[y1:y2 + 1, x1:x2 + 1]
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

            black_segments = self.find_black_segments(line, threshold)

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

    def is_diagonal_line_connected(self, x1, y1, x2, y2):
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

        return True

    @staticmethod
    def filter(corners, results):
        corner_set = set(map(tuple, corners))

        for result in results:
            x1, y1, x2, y2 = result.xyxy
            corner_set -= {(x, y) for x in range(int(x1), int(x2) + 1)
                           for y in range(int(y1), int(y2) + 1)
                           if (x, y) in corner_set}

        return list(corner_set)

    def corner_in_box(self, x, y):
        for result in self.results:
            x1, y1, x2, y2 = result.xyxy
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    @staticmethod
    def filterSingle(point, results):
        x, y = point
        is_inside_any_box = False

        for result in results:
            x1, y1, x2, y2 = result.xyxy
            if x1 <= x <= x2 and y1 <= y <= y2:
                is_inside_any_box = True
                break
            if is_inside_any_box:
                break

        return is_inside_any_box

    @staticmethod
    @jit
    def intersection(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        A1, B1 = math.cos(theta1), math.sin(theta1)
        A2, B2 = math.cos(theta2), math.sin(theta2)
        C1, C2 = rho1, rho2

        determinant = A1 * B2 - A2 * B1

        if determinant == 0:
            return None  # Lines are parallel

        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return (int(x), int(y))

    def harris_corner_detection(self):
        dst = cv2.cornerHarris(self.gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        corners = np.argwhere(dst > 0.01 * dst.max())
        return [(int(pt[1]), int(pt[0])) for pt in corners]

    def shi_tomasi_corner_detection(self):
        corners = cv2.goodFeaturesToTrack(self.gray, 100, 0.01, 10)
        return [(int(pt[0][0]), int(pt[0][1])) for pt in corners]

    def fast_corner_detection(self):
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(self.gray, None)
        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def orb_corner_detection(self):
        orb = cv2.ORB_create()
        keypoints = orb.detect(self.gray, None)
        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def agast_corner_detection(self):
        agast = cv2.AgastFeatureDetector_create()
        keypoints = agast.detect(self.gray, None)
        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def sift_corner_detection(self):
        sift = cv2.SIFT_create()
        keypoints = sift.detect(self.gray, None)
        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def surf_corner_detection(self):
        surf = cv2.xfeatures2d.SURF_create()
        keypoints = surf.detect(self.gray, None)
        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def brisk_corner_detection(self):
        brisk = cv2.BRISK_create()
        keypoints = brisk.detect(self.gray, None)
        return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    def mser_corner_detection(self):
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(self.gray)
        corners = [pt for region in regions for pt in region]
        return list(set((int(pt[0]), int(pt[1])) for pt in corners))


def plotHoughLines(image_cv, lines):
    image_cv_copy = image_cv.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = np.int32((x0 - 1000 * b, y0 + 1000 * a))
            pt2 = np.int32((x0 + 1000 * b, y0 - 1000 * a))
            cv2.line(image_cv_copy, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Hough Image', image_cv_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    from Core.Detection import Detection
    from Core.ImageManager import Manager

    image_manager = Manager(r'C:\Users\PC\Desktop\public\images\058.png')
    # image_manager = Manager(r'C:\Users\PC\Desktop\Circuit-Dataset\images\58.png')

    detector = Detection(model_path='../checkpoints/best.pt')

    results = detector.predict(image=image_manager.image, conf=0.6, iou=0.1, draw=True)
    line_width = detector.find_line_widths(results, image_manager.binary_image)
    print(line_width)

    results = detector.fix(results, image_manager.binary_image, move2closest=True, find_pose=True, is_draw=True,
                           image=image_manager.image)
    results, pose, mode = detector.results2custom(results, image_manager.binary_image)

    # alg = ['Hough', 'Harris', 'Shi-Tomasi', 'FAST', 'ORB', 'AGAST', 'SIFT', 'SURF', 'BRISK', 'MSER']
    alg = ['Harris', 'Shi-Tomasi', 'ORB', 'SIFT', 'BRISK']

    corner_detector = Corner(image_manager=image_manager)
    corner = corner_detector.predict(draw=True, threshold=100, results=results, line_width=line_width, algorithm=alg)
    # corner_detector.plot(corner)
