import face_model
import argparse
import cv2
import sys
import numpy as np
import os

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='250,200', help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--rootPath', default='', type=str,
                    help='the rootPath of images to align')
parser.add_argument('--alignPath', default='', type=str,
                    help='the path to save the aligned face images')
args = parser.parse_args()

model = face_model.FaceModel(args)


def main():

    all_path_list = collect_path_list(args.rootPath)
    count = 0

    # face landmarks
    alignmarks = np.array([
        [72, 125],
        [128, 125],
        [100, 150],
        [80, 180],
        [120, 180]], dtype=np.float32)
    
    # crop size
    align_size = args.image_size

    for folder in all_path_list:
        save_path = os.path.join(args.alignPath, folder[len(args.rootPath)+1:])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_list = os.listdir(folder)
        for img_name in sorted(img_list):
            if img_name.endswith(('jpg', 'png', 'bmp')):
                img_path = os.path.join(args.rootPath, folder, img_name)
                img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_aligned = model.get_input_cb(
                    img_data, align_size, alignmarks)
                save_aligned_img(img_aligned, save_path, img_name)
                # face_aligen_img(img_path, save_path)
                count += 1
                print(count)

def save_aligned_img(img_aligned, save_path, save_name):
    img_aligned = np.transpose(img_aligned, (1, 2, 0))
    img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_path, save_name), img_aligned)

def gci(filepath, all_file_list):

    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            all_file_list.append(fi_d)
            gci(fi_d, all_file_list)
        else:
            pass

def collect_path_list(rootPath):
    all_file_list = []
    gci(rootPath, all_file_list)
    all_file_list.append(rootPath)

    return all_file_list


if __name__ == '__main__':
    main()

