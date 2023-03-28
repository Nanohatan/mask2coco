import collections as cl
import os
import cv2
def info():
    #後回し
    pass

def get_mask_info(fn):
    in_fn = os.path.join("input",fn)
    #まずサンプルのmask画像から輪郭とれるか試す
    gray = cv2.imread(in_fn,cv2.IMREAD_GRAYSCALE)
    bina = cv2.threshold(gray, 124, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    color = cv2.imread(os.path.join("input","sample_image.jpg"))
    thickness = 10
    cv2.drawContours(color, contours, -1, (0, 0, 255), thickness)
    out_fn = os.path.join("output",fn)
    cv2.imwrite(out_fn,color)

    

def main():
    fn = "mask.png"
    get_mask_info(fn)
    pass
if __name__=="__main__":
    main()