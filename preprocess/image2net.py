import os
import json
from multiprocessing import Pool, cpu_count
from Core.Detection import Detection
from Core.Corner import Corner
from Core.interface import output
from Core.Connection import Connect
from Core.ImageManager import Manager
from Core.Utils import plot_result

# from Core.ocr import OCR

load_dir = r'C:\Users\PC\Desktop\public\images'
# load_dir = r'C:\Users\PC\Desktop\Circuit-Dataset\images'
# load_dir = r'../cases/'
save_dir = r'../net'
detector = Detection(model_path='../checkpoints/best-tuned.onnx', verbose=False)


# ocr_detector = OCR()


def solution(image, type, is_draw=True, file_name=''):
    print('*' * 100)
    print('Init')
    image_manager = Manager(image)
    print('Done')

    print('*' * 100)
    print('Yolo Detection')
    results = detector.predict(image=image_manager.image, conf=0.6, iou=0.5, imgsz=1024, draw=False, verbose=False)
    print('\tDone Detection')
    line_width = detector.find_line_widths(results, image_manager.binary_image)
    print('\tDone find_line_widths')
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
    print('Done')

    result = {
        "ckt_type": type,
        'ckt_netlist': net
    }
    print(result)

    if is_draw:
        plot_result(corner, type, image_manager.image, f'../results/{file_name}')

    return result

# TYPE = 1: Visualization of all images
# TYPE != 1: Generate net from image for GCN training
def process_file(filename, type=1):
    if type == 1:
        if os.path.exists(f'../results/{filename}'):
            return
        print(f'Begin file {filename}')
        result = solution(os.path.join(load_dir, filename), filename, file_name=filename)
    else:
        case_index = os.path.splitext(filename)[0]
        if os.path.exists(os.path.join(save_dir, f'{case_index}_netlist_dict.txt')):
            return

        index_json_path = os.path.join(load_dir, f'{case_index}.json')
        with open(index_json_path, 'r') as file:
            index_data = json.load(file)

        flags = index_data['flags']
        true_keys = [key for key, value in flags.items() if value is True]
        if not true_keys or true_keys[0] == 'Else' or len(true_keys) > 1:
            return

        result = solution(os.path.join(load_dir, filename), true_keys[0], file_name=filename)

        with open(os.path.join(save_dir, f'{case_index}_netlist_dict.txt'), 'w') as file:
            json.dump(result, file)


if __name__ == '__main__':
    filenames = [f for f in os.listdir(load_dir) if f.endswith('.png')]

    # Create a pool of worker processes
    with Pool(processes=4) as pool:
        result = pool.map_async(process_file, filenames)
        result.get()

    # for f in filenames:
    #     process_file(f)
