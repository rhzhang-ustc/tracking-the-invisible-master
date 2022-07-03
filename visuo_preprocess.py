import cv2
import numpy as np


def write_result(frame_lst, output_path, colored):

    frame_shape = frame_lst[0].shape
    video_size = (frame_shape[1], frame_shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, video_size, colored)

    for frame in frame_lst:
        video.write(frame)

    video.release()


def split_frames(video_path, output_path, start, end, step):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_lst = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = np.array(frame).shape
        frame = frame[:600]
        frame = cv2.resize(frame, dsize=(int(w/2), int(h/2)))

        if start <= count < end and (count-start) % step == 0:
            frame_lst.append(frame)
        count = count+1

    write_result(frame_lst, output_path, True)


split_frames("data/visuo.mp4", "data/visuo_test.mp4", 4800, 4940, 2)
