from Core.ImageManager import Manager
import easyocr
import cv2
import matplotlib.pyplot as plt
from Core.interface import Node
import numpy as np


class OCR:
    def __init__(self):
        self.detector = easyocr.Reader(['en'], gpu=False)

    def predict(self, image_manager: Manager, conf=0.6, scale=0.8, is_draw=False):
        result = self.detector.readtext(np.array(image_manager.image), text_threshold=conf)
        if is_draw:
            self.plot(result, image_manager)
        result = self.result2custom(result, scale)
        if is_draw:
            self.plot_custom(result, image_manager)
        return result

    def plot(self, result, image_manager: Manager):
        image_cv_copy = image_manager.image_cv.copy()
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(image_cv_copy, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image_cv_copy, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        image_rgb = cv2.cvtColor(image_cv_copy, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    def result2custom(self, result, scale=0.8):
        nodes = []
        for (bbox, text, prob) in result:
            x = [point[0] for point in bbox]
            y = [point[1] for point in bbox]
            x1, x2 = min(x), max(x)
            y1, y2 = min(y), max(y)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            width = (x2 - x1) * scale
            height = (y2 - y1) * scale

            new_x1 = int(center_x - width / 2)
            new_x2 = int(center_x + width / 2)
            new_y1 = int(center_y - height / 2)
            new_y2 = int(center_y + height / 2)

            nodes.append(Node(xyxy=[new_x1, new_y1, new_x2, new_y2]))
        return nodes

    def plot_custom(self, nodes, image_manager: Manager):
        image_cv_copy = image_manager.image_cv.copy()
        for node in nodes:
            x1, y1, x2, y2 = node.xyxy
            cv2.rectangle(image_cv_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

        image_rgb = cv2.cvtColor(image_cv_copy, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    image_manager = Manager(r'C:\Users\PC\Desktop\public\images\023.png')
    ocr = OCR()
    ocr_box = ocr.predict(image_manager, is_draw=True)
    print(ocr_box)
