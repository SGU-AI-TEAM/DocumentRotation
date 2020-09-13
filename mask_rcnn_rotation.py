from copy import deepcopy

from tools.document_mask_rcnn import DocRecognizer
import cv2
import argparse
import numpy as np

from utils import draw_box, draw_mask_on_image_array, show_image, four_point_nearest_rectangle, \
    four_point_transform_no_order

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image files")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    # convert to rgb
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # create mask rcnn recognizer
    recognizer = DocRecognizer(model_path="inference/saved_model/")

    # inference
    predictions = recognizer.predict_one(data=image)

    # filter by confident threshold
    THRESHOLD = 0.7
    scores = predictions["detection_scores"][0]
    cls_ids = predictions["detection_classes"][0]
    boxes = predictions["detection_boxes"][0]
    maskes = predictions["detection_masks_reframed"][0]

    valid = np.where(scores >= THRESHOLD)[0]
    cls_ids = cls_ids[valid]
    boxes = boxes[valid]
    maskes = maskes[valid]
    scores = scores[valid]

    # find 4 points
    d_image = deepcopy(image)  # image to visualize
    results = []  # list containing rotated object images
    for cls_id, box, mask, score in zip(cls_ids, boxes, maskes, scores):
        # draw box
        d_image = draw_box(image=d_image, box=box, label="{}-{}".format(cls_id, score))
        # draw mask
        draw_mask_on_image_array(image=d_image, mask=mask)
        show_image(d_image)

        # find rotation rectangle from mask
        tmp_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(tmp_) == 3:
            contours, hierarchy = tmp_[1:]
        else:
            contours, hierarchy = tmp_
        max_contour = max(contours, key=lambda x: cv2.contourArea(x))
        rotrect = cv2.minAreaRect(max_contour)
        rotrect_box = cv2.boxPoints(rotrect)
        rotrect_box = np.int0(rotrect_box)

        # estimate 4 point of document by nearest rotated rectangle points
        doc_box = four_point_nearest_rectangle(rotrect_box, contours)
        doc_box = np.array(doc_box)
        obj_img = four_point_transform_no_order(image, doc_box)
        h, w = obj_img.shape[:2]

        #TODO: find the rotate angle later by detecting face or sign
        # just rotate 90 degree
        if h > w:
            obj_img = cv2.rotate(obj_img, cv2.ROTATE_90_CLOCKWISE)
        show_image(obj_img)
        results.append(obj_img)

    # ********* all result is contained in "result" param" *********
