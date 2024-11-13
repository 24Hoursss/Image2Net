from PIL import Image
import numpy as np
import cv2
import io
from matplotlib import pyplot as plt


class Manager:
    def __init__(self, image, maxval=255, filter_gray=True, smooth=False):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image_stream = io.BytesIO(image)
            image = Image.open(image_stream)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError("Unsupported image type")

        image = self.scale_image(image, 1024)

        self.image = image.convert('RGB')
        # self.image = self.enhance_image(image.convert('RGB'))
        image_array = np.array(self.image)
        if filter_gray:
            image_array = self.filter_gray(image_array)
            # self.image = Image.fromarray(image_array)
        if self.is_low_quality(image_array):
            image_array = self.thicken_image(image_array, 3)
        else:
            image_array = self.thicken_image(image_array, 2)
        if smooth:
            image_array = self.smooth(image_array)

        self.gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        otsu_thresh, _ = cv2.threshold(self.gray_image, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, self.binary_image = cv2.threshold(self.gray_image, otsu_thresh + 25, maxval, cv2.THRESH_BINARY)

        self.image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        self.gaus = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

    @staticmethod
    def scale_image(image, max_size):
        width, height = image.size
        if max(width, height) > max_size:
            scaling_factor = max_size / float(max(width, height))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            return image.resize(new_size, Image.LANCZOS)
        return image

    @staticmethod
    def image2gaus(image_array):
        blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
        bilateral = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(bilateral, cv2.COLOR_RGB2GRAY)
        return gray

    @staticmethod
    def smooth(image_array):
        bilateral = cv2.bilateralFilter(image_array, d=9, sigmaColor=75, sigmaSpace=75)

        # cv2.imshow('smoothed', bilateral)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return bilateral

    @staticmethod
    def is_low_quality(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        # print(f'{laplacian_var=}')
        return laplacian_var < 2500

    @staticmethod
    def thicken_image(image, n=3):
        inverted_image = cv2.bitwise_not(np.array(image))

        kernel = np.ones((n, n), np.uint8)

        dilated_inverted_image = cv2.dilate(inverted_image, kernel, iterations=1)

        image = cv2.bitwise_not(dilated_inverted_image)

        # cv2.imshow('enhanced image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return image

    @staticmethod
    def enhance_image(image, is_draw=False):
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        laplacian = cv2.Laplacian(image_cv, cv2.CV_64F)
        sharp_image = cv2.convertScaleAbs(laplacian)
        enhanced_image = cv2.addWeighted(image_cv, 1.0, sharp_image, 0.3, 0)
        enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        enhanced_image_pil = Image.fromarray(enhanced_image_rgb)

        if is_draw:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(enhanced_image_pil)
            plt.axis('off')
            plt.show()

        return enhanced_image_pil

    @staticmethod
    def filter_gray(image_array):
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        max_gray_value = np.argmax(hist)
        lower_gray = np.array([0, 0, max_gray_value - 10])
        upper_gray = np.array([180, 50, max_gray_value + 10])

        mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
        white_image = np.full_like(image_array, 255)
        image_array[mask != 0] = white_image[mask != 0]

        return image_array


if __name__ == '__main__':
    # image_manager = Manager(r'C:\Users\PC\Desktop\Circuit-Dataset\images\79.png', thicken=False, filter_gray=True)
    image_manager = Manager(r'C:\Users\PC\Desktop\public\images\002.png')
    # image_manager = Manager(r'C:\Users\PC\Desktop\public\images\014.png')
    plt.imshow(image_manager.binary_image, cmap='gray')
    plt.show()
