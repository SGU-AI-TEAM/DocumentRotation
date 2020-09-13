from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageColor as ImageColor
from imutils.perspective import order_points
from scipy.spatial.distance import cdist
import numpy as np

def show_image(image):
    plt.imshow(image), plt.show()

def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*(mask > 0))).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))

def four_point_nearest_rectangle(four_point_rect, points):
    (tl, tr, br, bl) = order_points(four_point_rect)

    nearest_tl = min(np.squeeze(points), key=lambda x: cdist([x], [tl])).tolist()
    nearest_tr = min(np.squeeze(points), key=lambda x: cdist([x], [tr])).tolist()
    nearest_br = min(np.squeeze(points), key=lambda x: cdist([x], [br])).tolist()
    nearest_bl = min(np.squeeze(points), key=lambda x: cdist([x], [bl])).tolist()
    return [nearest_tl, nearest_tr, nearest_br, nearest_bl]


def four_point_transform_no_order(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    # rect1 = order_points(pts)
    rect = np.array(pts, dtype=np.float32)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def crop_box(image, box):
    """
    Args:
        image: the inputted image
        box: the gt_box to crop by
    Returns:
    """
    image_h, image_w, _ = image.shape
    xmin = max(0, int(box[0]))
    ymin = max(0, int(box[1]))
    xmax = min(image_w, int(box[2]))
    ymax = min(image_h, int(box[3]))
    cropped_image = image[ymin: ymax, xmin: xmax]
    # if IS_DEBUG:
    #     import matplotlib.pyplot as plt
    #     plt.imshow(cropped_image)
    #     plt.show()

    return cropped_image

def draw_box_4angle(image, box, color):
    image_h, image_w, _ = image.shape

    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    d_image = deepcopy(image)

    topleft = [xmin, ymin]
    botright = [xmax, ymax]
    topright = [xmax, ymin]
    botleft = [xmin, ymax]

    # 8 addition point
    n1_topleft = [xmin, int(ymin + (ymax - ymin) / 4)]
    n2_topleft = [int(xmin + (xmax - xmin) / 4), ymin]

    n1_topright = [int(xmin + (xmax - xmin) * 3 / 4), ymin]
    n2_topright = [xmax, int(ymin + (ymax - ymin) / 4)]

    n1_botright = [xmax, int(ymin + (ymax - ymin) * 3 / 4)]
    n2_botright = [int(xmin + (xmax - xmin) * 3 / 4), ymax]

    n1_botleft = [int(xmin + (xmax - xmin) / 4), ymax]
    n2_botleft = [xmin, int(ymin + (ymax - ymin) * 3 / 4)]

    # drawing
    topleft_points = np.array([n1_topleft, topleft, n2_topleft], np.int32)
    topleft_points = topleft_points.reshape((-1, 1, 2))

    topright_points = np.array([n1_topright, topright, n2_topright], np.int32)
    topright_points = topright_points.reshape((-1, 1, 2))

    botright_points = np.array([n1_botright, botright, n2_botright], np.int32)
    botright_points = botright_points.reshape((-1, 1, 2))

    botleft_points = np.array([n1_botleft, botleft, n2_botleft], np.int32)
    botleft_points = botleft_points.reshape((-1, 1, 2))
    cv2.polylines(d_image, [topleft_points, topright_points, botright_points, botleft_points], False, color,
                  thickness=3)

    return d_image

def draw_box(image, box, label=None, color=(255, 0, 0), label_color=(0, 255, 0), **kwargs):
    """
    Args:
        image:
        box:
        labels:
        color:
    Returns:
    """
    image_h, image_w, _ = image.shape

    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    d_image = deepcopy(image)

    # paste the replacement
    if kwargs is not None and "replace" in kwargs:
        replace_image = kwargs["replace"]
        h, w, _ = replace_image.shape
        r_ymin = max(0, ymin)
        r_ymax = r_ymin + h
        r_xmin = max(0, xmin)
        r_xmax = r_xmin + w
        if r_ymax > image_h:
            delta_y = r_ymax - image_h
            r_ymin -= delta_y
            r_ymin = image_h
        if r_xmax > image_w:
            delta_x = r_xmax - image_h
            r_xmin -= delta_x
            r_xmax = image_w
        d_image[r_ymin:r_ymax, r_xmin: r_xmax] = replace_image

    if kwargs.get("four_angles", False):
        d_image = draw_box_4angle(d_image, (xmin, ymin, xmax, ymax), color)
    else:
        cv2.rectangle(d_image, (xmin, ymin), (xmax, ymax), color, 2)
    if label:
        fontsize = 1e-3 * kwargs.get("fontsize", min(image_h, image_w))
        x_label = int(kwargs.get("xlabel", max(0, xmin - 100)))
        y_label = int(kwargs.get("ylabel", max(ymin - 13, 0)))

        for i, line in enumerate(label.split('\n')):
            y = int(y_label + i * (fontsize * 40 + 2))
            cv2.putText(d_image,
                        line,
                        (x_label, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontsize,
                        label_color, 2)
    return d_image