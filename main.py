
from utils import *
import argparse

# super params
theta_match = 0.9  # for dim=128 features
target_match = 0.95
sigma_0 = 10
motion_thresh = 2
smoothing_alpha = 0.7


# init argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, help='The video need to be tracked',
                    default="data/visuo_test.mp4")
parser.add_argument('--object_location', type=tuple, help='object location in (x, y) form',
                    default=(288, 208))
parser.add_argument('--output_path', type=str, help='result video output path',
                    default='test.mp4')
args = parser.parse_args()


# init detector & feature computer
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=3.0,
                      blockSize=3, useHarrisDetector=True, k=0.04)  # param for corner detector
ptrGFTT = cv2.GFTTDetector_create(**feature_params)
sift = cv2.xfeatures2d.SIFT_create()


# read first frame
cap = cv2.VideoCapture(args.video_path)
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
obj_center = args.object_location
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
target_prev_pt = target_pt
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

                if max_prob > 0.05:
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

    write_result(result_lst, args.output_path, True)  # wrong indent for test
