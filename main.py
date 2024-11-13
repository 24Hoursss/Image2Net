from Core.Detection import Detection
from Core.Corner import Corner
from Core.interface import output, parseSingleInput
from Core.Connection import Connect
from Core.GCN import GCNClassifier
from Core.RunTime import cal_time
from Core.ImageManager import Manager
# from Core.ocr import OCR
import warnings
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

print('*' * 100)
print('Loading models')
detector = Detection(model_path='checkpoints/best-tuned.onnx', verbose=False)
detector.model.predict(Image.fromarray(np.ones_like((1, 3, 1024, 1024), dtype=np.uint8)), imgsz=1024, verbose=False)
classifier = GCNClassifier(model_path='checkpoints/GCN.pt')
# ocr_detector = OCR()
print('Done')
print('*' * 100 + '\n')


@cal_time('s')
def solution(image):
    print('*' * 100)
    print('Init')
    image_manager = Manager(image)
    print('Done')

    print('*' * 100)
    print('Yolo Detection')
    results = detector.predict(image=image_manager.image, conf=0.6, iou=0.5, imgsz=1024, draw=False, verbose=False)
    print('\tDone Detection')
    line_width = detector.find_line_widths(results, image_manager.binary_image)
    print(f'\tDone find_line_widths={line_width}')
    results = detector.fix(results, image_manager.binary_image, move2closest=True, find_pose=True, is_draw=False,
                           image=image_manager.image)
    print('\tDone fix position')
    results, pose, mode = detector.results2custom(results, image_manager.binary_image)
    print('\tDone result to custom')
    print('Done')

    print('*' * 100)

    print('OCR Detection')
    # ocr_result = ocr_detector.predict(image_manager, scale=0.8, conf=0.8, is_draw=False)
    ocr_result = []
    print('Done OCR detection')

    print('*' * 100)

    print('Corner Detection')
    corner_detector = Corner(image_manager=image_manager)
    alg = ['Harris', 'Shi-Tomasi', 'ORB', 'SIFT', 'BRISK']
    # alg = ['Hough']
    corner = corner_detector.predict(draw=False, threshold=100, results=results + ocr_result, line_width=line_width,
                                     algorithm=alg)
    print('Done')

    print('*' * 100)

    print(f'Build to Graph: Using {mode=}')
    corner += pose
    graph_builder = Connect(corner=corner, binary_image=image_manager.binary_image, results=results, mode=mode)
    corner = graph_builder.build_graph(threshold=line_width * 2, parallel_mode=None)
    print('Done')

    print('*' * 100)

    print('Output')
    net = output(corner)
    # print(net)
    print('Done')

    print('*' * 100)

    print(f'GCN Classification')
    graph = parseSingleInput(net)
    circuit_type = classifier.val(graph)
    print('Done')

    print('*' * 100)

    result = {
        "ckt_type": circuit_type,
        'ckt_netlist': net
    }
    print(result)
    return str(result)


if __name__ == '__main__':
    import os

    load_dir = r'C:\Users\PC\Desktop\public\images'
    save_dir = r'C:\Users\PC\Desktop\public\generate'

    # load_dir = r'/home/public/public/images'
    # save_dir = r'./generate'
    for file in os.listdir(load_dir):
        print('Begin File:', file)
        save_name = file.replace('.png', '.txt')
        result = solution(os.path.join(load_dir, file))
        with open(os.path.join(save_dir, save_name), 'w') as f:
            f.write(result)
