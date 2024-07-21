import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

# JSONファイルと画像のパス
annotation_file = 'annotations.json'
image_file = "input/sample/origin_hari.jpeg"

# アノテーションJSONファイルを読み込む
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

# 画像を読み込む
image = cv2.imread(image_file)

# バウンディングボックスとセグメンテーションを可視化する関数
def visualize_annotations(image, annotations):
    for annotation in annotations:
        # バウンディングボックス
        bbox = annotation['bbox']
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 15)

        # セグメンテーション
        for segmentation in annotation['segmentation']:
            points = np.array(segmentation).reshape((-1, 2))
            cv2.polylines(image, [points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=15)

    return image

# アノテーションを取得
annotations = coco_data['annotations']

# 画像にアノテーションを可視化
visualized_image = visualize_annotations(image, annotations)

# 結果を表示
plt.imshow(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 画像を保存する場合
cv2.imwrite('visualized_image.png', visualized_image)
