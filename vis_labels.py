import numpy as np
from skimage.color import label2rgb
import cv2
import sys

if __name__ == "__main__":
    image = sys.argv[1]
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    ids = np.unique(img)
    print("IDS in image: {}".format(ids))
    print("id at center: {}".format(img[119,159]))
    rgb = label2rgb(img)
    w = "Labels (float)"
    cv2.namedWindow(w)
    cv2.imshow(w, rgb)
    cv2.waitKey(-1)
