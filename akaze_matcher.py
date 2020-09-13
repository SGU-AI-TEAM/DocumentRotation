import cv2
import numpy as np

class KAZEMatcher():
    MIN_MATCH_COUNT = 10

    def match(self, input_img, template_img):
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        # img = cv2.Canny(img, 50, 200)
        # img = cv2.dilate(img, None, iterations=1)
        # img = cv2.erode(img, None, iterations=1)
        temp = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
        # temp = cv2.equalizeHist(temp)
        temp = cv2.GaussianBlur(temp, (5, 5), 0)
        # temp = cv2.Canny(temp, 50, 200)
        # temp = cv2.dilate(temp, None, iterations=1)
        # temp = cv2.erode(temp, None, iterations=1)
        # img = input_img
        # temp = template_img

        # Initiate STAR detector
        detector = cv2.AKAZE_create()
        # cv2.xfeatures2d_SIFT()

        # find the keypoints with SURF Detector
        img_kp, img_des = detector.detectAndCompute(img, None)
        tmp_kp, tmp_des = detector.detectAndCompute(temp, None)

        # verify if there is enough description
        if not (img_kp is not None and tmp_kp is not None and img_des is not None and tmp_des is not None and
                img_des.shape[0] >= self.MIN_MATCH_COUNT and tmp_des.shape[0] >= self.MIN_MATCH_COUNT):
            return None, None

        # -- Matching descriptor vectors
        FLANN_INDEX_LSH = 6
        # FLANN_INDEX_AUTOTUNED = 255
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=12,  # 12
                            key_size=20,  # 20
                            multi_probe_level=2)  # 2
        # AutotunedIndexParams = dict(
        #     algorithm=FLANN_INDEX_AUTOTUNED,
        #     target_precision=0.9,
        #     build_weight=0.01,
        #     memory_weight=0,
        #     sample_fraction=1)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = matcher.knnMatch(tmp_des, img_des, 2, )

        # -- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.75
        good_matches = []
        for match in knn_matches:
            if match and len(match) == 2:
                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= self.MIN_MATCH_COUNT:
            # -- Localize the object
            obj = np.empty((len(good_matches), 2), dtype=np.float32)
            scene = np.empty((len(good_matches), 2), dtype=np.float32)
            for i in range(len(good_matches)):
                # -- Get the keypoints from the good matches
                obj[i, 0] = tmp_kp[good_matches[i].queryIdx].pt[0]
                obj[i, 1] = tmp_kp[good_matches[i].queryIdx].pt[1]
                scene[i, 0] = img_kp[good_matches[i].trainIdx].pt[0]
                scene[i, 1] = img_kp[good_matches[i].trainIdx].pt[1]

            H, mask = cv2.findHomography(obj, scene, cv2.RANSAC)
            return H, len(good_matches)
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), self.MIN_MATCH_COUNT))
            return None, None