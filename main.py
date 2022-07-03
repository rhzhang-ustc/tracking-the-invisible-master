import cv2
import numpy as np
from numpy import matlib as npm
from collections import namedtuple
import math
import operator
from functools import reduce

# super params
theta_match = 0.95  # for dim=128 features
sigma_0 = 10
theta_move = 0
smoothing_alpha = 0.7
object_thresh = 500


def normalize_feat(feat):
    norm = np.linalg.norm(feat)
    feat = feat/(norm + 1e-4)
    return feat


def add_gaussion_map(img, pt, mean_r, mean_phi, var):
    # the bottleneck!

    h, w = img.shape
    center_x = mean_r*math.cos(mean_phi)+pt[0]
    center_y = mean_r*math.sin(mean_phi)+pt[1]

    distance_map = np.array([[[i - center_x, j-center_y] for j in range(w)] for i in range(h)])  # (480, 640, 2)
    distance_map = np.expand_dims(distance_map, axis=3)  # (480, 640, 2, 1)

    var_inverse = np.linalg.inv(var)
    det = np.linalg.det(var)

    exponent = np.array([[np.dot(np.dot(distance_map[i, j].T, var_inverse), distance_map[i, j]) for j in range(w)]
                         for i in range(h)])
    exponent = np.squeeze(exponent)

    img_addition = 1/np.sqrt(det) * np.exp(-0.5 * exponent)
    img = img + img_addition

    return np.max(img_addition)


def object_compare(obj1, obj2):
    #  return difference between two objects
    hist1 = cv2.calcHist([obj1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([obj2], [0], None, [256], [0, 256])
    differ = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a-b)**2, hist1, hist2)))/len(hist1))
    return differ


def to_polar(pt1, pt2):
    # return r & phi between two points
    arr = (pt2[0] - pt1[0], pt2[1] - pt1[1])
    distance = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    try:
        angle = math.atan(arr[1] / arr[0])
    except ZeroDivisionError:
        angle = np.pi/2
    return distance, angle


def obj_move(bbox, bbox_init):
    return np.sum(np.abs(bbox[i] - bbox_init[i]) for i in range(4)) > theta_move


def write_result(frame_lst, output_path, colored):

    frame_shape = frame_lst[0].shape
    video_size = (frame_shape[1], frame_shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, video_size, colored)

    for frame in frame_lst:
        video.write(frame)

    video.release()


# init detector & feature computer

feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=3.0,
                      blockSize=3, useHarrisDetector=True, k=0.04)  # param for corner detector
ptrGFTT = cv2.GFTTDetector_create(**feature_params)

sift = cv2.xfeatures2d.SIFT_create()

# read first frame
cap = cv2.VideoCapture("ethCup_input.WMV")
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

dst = cv2.cornerHarris(prev_gray, 2, 3, 0.02)
kp_mask = np.zeros_like(prev_gray)
kp_mask[dst > 0.001 * dst.max()] = 1
kp_mask = cv2.dilate(kp_mask, None)

kp1, des1 = sift.detectAndCompute(prev_gray, mask=kp_mask.astype(np.uint8))

# init object to track using RGB frames
tracker = cv2.TrackerKCF_create()
bbox = (150, 99, 78, 215)
bbox_prev = bbox
tracker.init(first_frame, bbox)
bbox_prev_center = (189, 206)
object_track = prev_gray[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]].copy()
result_lst = []

# init database
db_size = len(kp1)

des_db = [normalize_feat(des1[i]) for i in range(db_size)]
r_db = [to_polar(kp1[i].pt, bbox_prev_center)[0] for i in range(db_size)]
cov_db = [sigma_0 * np.eye(2) for i in range(db_size)]
phi_db = [to_polar(kp1[i].pt, bbox_prev_center)[1] for i in range(db_size)]
pt_db = [kp1[i].pt for i in range(db_size)]


# while run
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(frame_gray, 2, 3, 0.02)
    kp_mask = np.zeros_like(frame_gray)
    kp_mask[dst > 0.001 * dst.max()] = 1
    kp_mask = cv2.dilate(kp_mask, None)

    kp, des = sift.detectAndCompute(frame_gray, mask=kp_mask.astype(np.uint8))

    des = np.array([normalize_feat(feat) for feat in des])
    kp_num = len(kp)  # m

    corr = np.matmul(des_db, des.T)  # (198, m) m is the size of kps in next frame
    corr_reduce = corr.max(axis=0)  # (m ,0)

    idx_match = corr.argmax(axis=0)
    match_eval = corr_reduce > theta_match
    idx_match = [idx_match[i] if match_eval[i] else -1 for i in range(kp_num)]  # -1 no feature matched

    success, bbox = tracker.update(frame)
    object_detect = prev_gray[int(bbox[0]):int(bbox[0]+bbox[2]), int(bbox[1]):int(bbox[1]+bbox[3])].copy()

    diff = object_compare(object_track, object_detect)

    if diff < object_thresh:
        #  learn the model
        print("model learning")
        bbox_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

        if obj_move(bbox, bbox_prev):
            for i in range(kp_num):
                if idx_match[i] != -1:

                    origin_idx = idx_match[i]
                    pt = kp[i].pt
                    distance, phi = to_polar(pt, bbox_center)

                    r_db[origin_idx] = vote_r \
                        = smoothing_alpha * r_db[origin_idx] + (1-smoothing_alpha) * distance
                    phi_db[origin_idx] = vote_phi \
                        = smoothing_alpha * phi_db[origin_idx] + (1-smoothing_alpha) * phi

                    x_estimate = np.array([pt[0] + vote_r * math.cos(vote_phi), pt[1] + vote_r * math.sin(vote_phi)])
                    diff = np.expand_dims(x_estimate - np.array(bbox_center), axis=1)
                    sigma = np.dot(diff, diff.T)

                    cov_db[origin_idx] = smoothing_alpha * cov_db[origin_idx] + (1-smoothing_alpha) * sigma

                else:

                    pt = kp[i].pt
                    distance, phi = to_polar(pt, bbox_center)

                    des_db.append(des[i])
                    r_db.append(distance)
                    phi_db.append(phi)
                    cov_db.append(sigma_0 * np.eye(2))
                    pt_db.append(pt)

        bbox_prev = bbox
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        result_lst.append(frame)

    else:
        #  apply the model
        print("model applying")
        h, w, c = frame.shape
        gauss_map = np.zeros((h, w))   # (480, 640, 3)

        for i in range(kp_num):
            origin_idx = idx_match[i]
            if origin_idx != -1:
                # use matched feature to estimate objects

                distance = r_db[origin_idx]
                phi = phi_db[origin_idx]
                cov = cov_db[origin_idx]

                max_prob = add_gaussion_map(gauss_map, kp[i].pt, distance, phi, cov)

                center_x = int(distance * math.cos(phi) + kp[i].pt[0])
                center_y = int(distance * math.sin(phi) + kp[i].pt[1])

                start_point = (int(kp[i].pt[0]), int(kp[i].pt[1]))
                end_point = (center_x, center_y)

                cv2.line(frame, start_point, end_point, (0, 0, 255))
                cv2.rectangle(frame, start_point, (start_point[0] + 3, start_point[1] + 3), (0, 0, 255))
                cv2.rectangle(frame, end_point, (end_point[0] + 3, end_point[1] + 3), (255, 0, 0))

        maximum = np.unravel_index(np.argmax(gauss_map), gauss_map.shape)
        cv2.rectangle(frame, maximum, (maximum[0] + 3, maximum[1] + 3), (0, 255, 0), 2)
        result_lst.append(frame)

    write_result(result_lst, "test.mp4", True)  # indent not right
