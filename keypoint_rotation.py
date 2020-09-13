import cv2
import argparse
import numpy as np

from tools.akaze_matcher import KAZEMatcher
from utils import show_image, crop_box

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template", required=True, help="Path to template files")
    ap.add_argument("-i", "--image", required=True, help="Path to image files")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    # convert to rgb
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    template = cv2.imread(args["template"])
    # create mask rcnn recognizer
    matcher = KAZEMatcher()

    # find template
    homo, match_num = matcher.match(image, template)

    if homo is not None:
        resutl = [] # contain result images
        template_corners = np.empty((4, 1, 2), dtype=np.float32)
        template_corners[0, 0, 0] = 0
        template_corners[0, 0, 1] = 0
        template_corners[1, 0, 0] = template.shape[1]
        template_corners[1, 0, 1] = 0
        template_corners[2, 0, 0] = template.shape[1]
        template_corners[2, 0, 1] = template.shape[0]
        template_corners[3, 0, 0] = 0
        template_corners[3, 0, 1] = template.shape[0]
        object_corners = cv2.perspectiveTransform(template_corners, homo)
        object_box = cv2.boundingRect(object_corners)
        object_box = [object_box[0], object_box[1], object_box[0] + object_box[2],
                      object_box[1] + object_box[3]]
        object_image = crop_box(image, object_box)
        show_image(object_image)
        resutl.append(object_image)

    # ********* all result is contained in "result" param" *********
