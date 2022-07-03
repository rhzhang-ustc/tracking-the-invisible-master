import cv2
import numpy as np
import math


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

