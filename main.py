import cv2
import numpy as np
import json


def get_contours(fn,origin_fn,area_th,is_show=False):
    binary_image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    ret,binary_image = cv2.threshold(binary_image,0,255,cv2.THRESH_BINARY_INV)

    # 輪郭を検出する
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 閾値を超える輪郭のみをフィルタリング
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > area_th]

    if is_show:
        lw = 10
        
        # 検出された輪郭を表示する（オプション）
        origin_image = cv2.imread(origin_fn)
        cv2.drawContours(origin_image, filtered_contours, -1, (0, 255, 0), lw)
        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(origin_image, (x, y), (x + w, y + h), (255, 0, 0), lw)
        cv2.imshow('Filtered Contours', origin_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    annotations = contours_to_coco(filtered_contours)
    coco_format = {
    'images': [{
        'id': 1,
        'width': binary_image.shape[1],
        'height': binary_image.shape[0],
        'file_name': fn
    }],
    'annotations': annotations,
    'categories': [{
        'id': 1,
        'name': 'object'  # カテゴリ名を任意の値に設定
    }]
    }
    return filtered_contours, coco_format



def contours_to_coco(contours):
    annotations = []
    for i, contour in enumerate(contours):
        segmentation = contour.flatten().tolist()
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        annotation = {
            'segmentation': [segmentation],
            'area': area,
            'iscrowd': 0,
            'image_id': 1,  # 画像IDは任意の値に設定
            'bbox': [x, y, w, h],
            'category_id': 1,  # カテゴリIDも任意の値に設定
            'id': i + 1
        }
        annotations.append(annotation)
    return annotations

filtered_contours,coco_format = get_contours("input/sample/mask_hari.jpeg","input/sample/origin_hari.jpeg",1000,is_show=True)


with open('annotations.json', 'w') as f:
    json.dump(coco_format, f)

