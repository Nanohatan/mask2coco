import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

def draw_bboxes(img_path, txt_path):
    # 画像の読み込み
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # テキストファイルの読み込み
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # バウンディングボックスの描画
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        center_x = float(parts[1]) * width
        center_y = float(parts[2]) * height
        bbox_width = float(parts[3]) * width
        bbox_height = float(parts[4]) * height

        # 左上の座標
        top_left_x = int(center_x - bbox_width / 2)
        top_left_y = int(center_y - bbox_height / 2)
        # 右下の座標
        bottom_right_x = int(center_x + bbox_width / 2)
        bottom_right_y = int(center_y + bbox_height / 2)

        # バウンディングボックスを描画
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    # 画像の表示
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



# COCOアノテーションファイルのパス
ann_file = 'annotations.json'
image_dir = 'train'
output_dir = 'output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# COCOアノテーションを読み込む
coco = COCO(ann_file)
cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
class_map = {cat['id']: i for i, cat in enumerate(cats)}

# 画像リストを取得
img_ids = coco.getImgIds()
for img_id in tqdm(img_ids):
    img = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(ann_ids)

    file_name = img['file_name']
    txt_file = os.path.join(output_dir, file_name.replace('.jpeg', '.txt').replace('.png', '.txt'))
    
    with open(txt_file, 'a') as f:
        for ann in anns:
            bbox = ann['bbox']
            class_id = class_map[ann['category_id']]
            x_center = (bbox[0] + bbox[2] / 2) / img['width']
            y_center = (bbox[1] + bbox[3] / 2) / img['height']
            width = bbox[2] / img['width']
            height = bbox[3] / img['height']

            # YOLOフォーマット: <class_id> <x_center> <y_center> <width> <height>
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("Conversion completed!")
