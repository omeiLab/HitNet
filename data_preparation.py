# The functions and classes imported from utils.py are adapted from ai_badminton (Adobe Research)
# Original source: https://github.com/jhwang7628/monotrack/blob/main/modified-tracknet/train-hitnet.ipynb
# Licensed under the Adobe Research License - for non-commercial research use only.
import os
import pandas as pd
from utils import Trajectory, read_player_poses, resample

window_size = 7
half = window_size // 2
speed = 1.0

def read_court(court_path):
    # Open the file and read the first four lines
    with open(court_path, "r") as file:
        lines = file.readlines()
        # Extract the first four lines (court corner coordinates)
    corners = []
    for i in range(4):  # Read only the first four lines
        x, y = map(float, lines[i].strip().split(";"))  # Convert to float
        corners.append((x, y))  # Store as (x, y) tuple
    return corners

def fetch_data(basedir: str, rally: str):
    '''
    Fetch the corners, trajectory, hits, poses from a rally

    Argument:
        basedir: the directiory path of match
        rally: the name of rally, it's something like 'clip_1'
    
    Return:
        corners: the coordinate of 4 court corners
        trajectory: the processed trajectory
        hit: the label (0 = no hit, 1 = hit)
        bottom_player, top_player: poses data
    '''
    court_path = f'{basedir}/court/{rally}.txt'
    traj_path = f'{basedir}/TrackNet/{rally}.csv'
    hit_path = f'{basedir}/hits/{rally}_hit.csv'
    pose_path = f'{basedir}/poses/{rally}'


    # Note: Consider using the same court for all rallies
    if not os.path.exists(court_path):
    #     print(f'{court_path} not found')
        return None
    # court_pts = read_court(court_path)
    # orners = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

    trajectory = Trajectory(traj_path, interp = False)
    hit = pd.read_csv(hit_path)
    hit = hit.values[:, 1]
    poses = read_player_poses(pose_path)
    bottom_player, top_player = poses[0], poses[1]

    return {'trajectory': trajectory, 'hit': hit, 'bottom_player': bottom_player, 'top_player': top_player}

def process(basedir: str, rally: str):
    '''
    Prepare frame-level input data with fixed-size sliding window (7 frames) and edge padding.

    Args:
        basedir (str): Path to match directory
        rally (str): Rally name like 'clip_1'

    Returns:
        x_t (ndarray): Feature vectors of shape (num_frames, 7 * feature_dim)
        y_t (ndarray): Binary labels per frame (num_frames,)
    '''
    import numpy as np
    import cv2

    video_path = f'{basedir}/rally_video/{rally}.mp4'
    x_list, y_list = [], []

    # fetch rally data
    data_dict = fetch_data(basedir, rally)
    if data_dict is None:
        return None

    trajectory = data_dict['trajectory']
    hit = data_dict['hit']
    bottom_player = data_dict['bottom_player']
    top_player = data_dict['top_player']

    # get frame shape (optional)
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    height, width = frame.shape[:2]

    # resample all inputs
    trajectory.X = resample(np.array(trajectory.X), speed)
    trajectory.Y = resample(np.array(trajectory.Y), speed)
    hit = resample(hit, speed).round().astype(int)
    bottom_player = resample(bottom_player.values, speed)
    top_player = resample(top_player.values, speed)

    num_frames = len(hit)

    # label smoothing for temporal context
    smooth_hit = hit.copy()
    for t in range(num_frames):
        if hit[t] == 1:
            if t > 0:
                smooth_hit[t-1] = 1
            if t < num_frames - 1:
                smooth_hit[t+1] = 1

    def get_padded_window(arr, t):
        start = max(0, t - half)
        end = min(num_frames, t + half + 1)
        window = arr[start:end]

        # Padding with edge values
        if len(window) < window_size:
            if start == 0:
                pad = [arr[0]] * (window_size - len(window))
                window = pad + list(window)
            else:
                pad = [arr[-1]] * (window_size - len(window))
                window = list(window) + pad

        return np.array(window)

    for t in range(num_frames):
        bird_xy = np.stack([
            get_padded_window(trajectory.X, t),
            get_padded_window(trajectory.Y, t)
        ], axis=1)  # (7, 2)

        bottom = get_padded_window(bottom_player, t)  # (7, pose_dim)
        top = get_padded_window(top_player, t)        # (7, pose_dim)

        x = np.hstack([bird_xy, bottom, top])         # (7, feature_dim)
        x_list.append(x)                              # flatten to 1D
        y_list.append(smooth_hit[t])                  # binary label

    x_t = np.array(x_list)
    y_t = np.array(y_list)
    return x_t, y_t



