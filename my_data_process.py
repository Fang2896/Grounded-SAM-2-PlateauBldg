import numpy as np
import os
import cv2
import json


def create_mask_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    return mask


def main():
    data_dir = './data/plateau_annotated'
    out_dir = './data/plateau_mask'

    for name in os.listdir(data_dir):
        if name.endswith('.json'):  # 确保处理的是 JSON 文件
            json_path = os.path.join(data_dir, name)
            
            mask = create_mask_from_json(json_path)
            
            mask_path = os.path.join(out_dir, os.path.splitext(name)[0] + '.png')
            cv2.imwrite(mask_path, mask)

            print(f'Mask saved to {mask_path}')


if __name__ == "__main__":
    main()

