import cv2
import numpy as np
from numpy import matlib as npm
from collections import namedtuple
import math
import operator
from functools import reduce

# super params
theta_match = 0.9  # for dim=128 features
target_match = 0.95
sigma_0 = 10
motion_thresh = 2
smoothing_alpha = 0.7
object_thresh = 500


def argmax2d(tensor):
    Y, X = list(tensor.shape)
    # flatten the Tensor along the height and width axes
    flat_tensor = tensor.reshape(-1)
    # argmax of the flat tensor
    argmax = np.argmax(flat_tensor)

    # convert the indices into 2d coordinates
    argmax_y = argmax // X  # row
    argmax_x = argmax % X  # col

    return argmax_y, argmax_x


def normalize(im):
    im = im - np.min(im)
    im = im / np.max(im)
    return im


def normalize_feat(feat):
    norm = np.linalg.norm(feat)
    feat = feat/(norm + 1e-4)
    return feat


def meshgrid2d(Y, X):
    grid_y = np.linspace(0.0, Y - 1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X - 1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    # outputs are Y x X
    return grid_y, grid_x


def to_polar(pt1, pt2):
    # return r & phi between two points
    arr = (pt2[0] - pt1[0], pt2[1] - pt1[1])
    distance = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    try:
        angle = math.atan(arr[1] / arr[0])
    except ZeroDivisionError:
        angle = np.pi/2
    return distance, angle


def write_result(frame_lst, output_path, colored):

    frame_shape = frame_lst[0].shape
    video_size = (frame_shape[1], frame_shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, video_size, colored)

    for frame in frame_lst:
        video.write(frame)

    video.release()


def generate_heatmap(frame, gauss_map):
    # generate vote space

    max_y, max_x = argmax2d(gauss_map)

    target_mask = np.zeros_like(frame)
    target_mask[int(max_y), int(max_x)] = 255
    target_mask = cv2.dilate(target_mask, None)

    H, W, C = np.array(frame).shape
    heatmap = (normalize(gauss_map) * 255).astype(np.uint8).reshape(H, W, 1)
    heatmap = np.repeat(heatmap, 3, 2)

    heat_vis = ((heatmap.astype(np.float32) + frame.astype(np.float32)) / 2.0).astype(np.uint8)

    heat_vis[target_mask > 0] = 255

    return heat_vis


# init detector & feature computer

feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=3.0,
                      blockSize=3, useHarrisDetector=True, k=0.04)  # param for corner detector
ptrGFTT = cv2.GFTTDetector_create(**feature_params)

sift = cv2.xfeatures2d.SIFT_create()

# read first frame
cap = cv2.VideoCapture("data/visuo_test.mp4")
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

H, W, C = first_frame.shape
grid_y, grid_x = meshgrid2d(H, W)
grid_xy = np.stack([grid_x, grid_y], axis=2)  # H x W x 2

dst = cv2.cornerHarris(prev_gray, 2, 3, 0.02)
kp_mask = np.zeros_like(prev_gray)
kp_mask[dst > 0.001 * dst.max()] = 1
kp_mask = cv2.dilate(kp_mask, None)

kp, des = sift.detectAndCompute(prev_gray, mask=kp_mask.astype(np.uint8))

# init object to track using RGB frames
obj_center = (288, 208)
result_lst = []

# init database
db_size = len(kp)

des_db = [normalize_feat(des[i]) for i in range(db_size)]
r_db = [to_polar(kp[i].pt, obj_center)[0] for i in range(db_size)]
cov_db = [sigma_0 * np.eye(2) for i in range(db_size)]
phi_db = [to_polar(kp[i].pt, obj_center)[1] for i in range(db_size)]
pt_db = [kp[i].pt for i in range(db_size)]


# select the target
target = np.array(obj_center).reshape(1, 2)  # [[189, 206]]
all_pt = np.stack([np.array(pt) for pt in pt_db])  # (186, 2)

dists = np.linalg.norm(all_pt - target, axis=1)
target_ind = np.argmin(dists)
target_pt = pt_db[target_ind]
target_feat = des_db[target_ind]

print("target_index", target_ind)


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

    des = np.array([normalize_feat(feat) for feat in des])  # （175，128）
    kp_num = len(kp)  # m

    # find target point first
    corr_target = np.matmul(des, target_feat.T).reshape(-1, 1)  # (kp_num, 1)
    max_corr = np.max(corr_target)

    if max_corr > target_match:
        # find the target
        target_ind = np.argmax(corr_target)
        target_prev_pt = target_pt  # for motion evaluation
        target_pt = kp[target_ind].pt
        target_feat = des[target_ind]

    else:
        target_ind = -1

    # find each key point's match in database
    corr = np.matmul(des_db, des.T)  # (198, m) m is the size of kps in next frame
    corr_reduce = corr.max(axis=0)  # (m ,0)

    idx_match = corr.argmax(axis=0)
    match_eval = corr_reduce > theta_match
    idx_match = [idx_match[i] if match_eval[i] else -1 for i in range(kp_num)]  # -1 no feature matched

    if target_ind != -1:
        #  learn the model
        print("model learning")

        motion = np.linalg.norm(np.array(target_prev_pt) - np.array(target_pt))

        if motion > motion_thresh:
            for i in range(kp_num):

                if idx_match[i] != -1:
                    origin_idx = idx_match[i]
                    pt = kp[i].pt

                    motion_kp = np.linalg.norm(np.array(pt_db[origin_idx]) - np.array(pt))

                    distance, phi = to_polar(pt, target_pt)

                    r_db[origin_idx] = vote_r \
                        = smoothing_alpha * r_db[origin_idx] + (1-smoothing_alpha) * distance
                    phi_db[origin_idx] = vote_phi \
                        = smoothing_alpha * phi_db[origin_idx] + (1-smoothing_alpha) * phi

                    x_estimate = np.array([pt[0] + vote_r * math.cos(vote_phi), pt[1] + vote_r * math.sin(vote_phi)])
                    diff = np.expand_dims(x_estimate - np.array(target_pt), axis=1)
                    sigma = np.dot(diff, diff.T)

                    cov_db[origin_idx] = smoothing_alpha * cov_db[origin_idx] + (1-smoothing_alpha) * sigma
                    pt_db[origin_idx] = pt

                else:

                    pt = kp[i].pt
                    distance, phi = to_polar(pt, target_pt)

                    des_db.append(des[i])
                    r_db.append(distance)
                    phi_db.append(phi)
                    cov_db.append(sigma_0 * np.eye(2))
                    pt_db.append(pt)

        x, y = target_pt
        cv2.rectangle(frame, (int(x-5), int(y-5)), (int(x+5), int(y+5)), (0, 255, 255), 2)
        result_lst.append(frame)

    else:
        #  apply the model
        print("model applying")
        gauss_map = np.zeros((H, W))   # (360, 640)

        for i in range(kp_num):
            origin_idx = idx_match[i]
            if origin_idx != -1:
                # use matched feature to estimate objects

                pt = kp[i].pt
                distance = r_db[origin_idx]
                phi = phi_db[origin_idx]
                cov = cov_db[origin_idx]

                mu_vote = np.array([pt[0] + distance * np.cos(phi),
                                    pt[1] + distance * np.sin(phi)])

                diff = grid_xy.reshape(-1, 1, 2) - mu_vote.reshape(1, 1, 2)  # H*W x 1 x 2
                diff_cov = np.matmul(diff, np.linalg.inv(cov).reshape(1, 2, 2))  # H*W x 1 x 2
                data_term = np.matmul(diff_cov, diff.reshape(H * W, 2, 1))
                data_term = data_term.reshape(-1)

                prob = 1 / np.sqrt(2 * np.pi * np.sum(np.abs(cov))) * np.exp(-0.5 * data_term)

                max_prob = np.max(prob)

                if max_prob > 0.09:
                    # draw instruct line
                    gauss_map += prob.reshape(H, W)

                    center_x = int(distance * math.cos(phi) + kp[i].pt[0])
                    center_y = int(distance * math.sin(phi) + kp[i].pt[1])

                    start_point = (int(kp[i].pt[0]), int(kp[i].pt[1]))
                    end_point = (center_x, center_y)

                    cv2.line(frame, start_point, end_point, (0, 0, 255))
                    cv2.rectangle(frame, start_point, (start_point[0] + 3, start_point[1] + 3), (0, 0, 255))
                    cv2.rectangle(frame, end_point, (end_point[0] + 3, end_point[1] + 3), (255, 0, 0))

        frame_heatmap = generate_heatmap(frame, gauss_map)  # frame is already plotted with lines
        result_lst.append(frame_heatmap)

    write_result(result_lst, "test.mp4", True)  # wrong indent for test