# Some functions in this file are adapted from ai_badminton (Adobe Research)
# Original source: https://github.com/jhwang7628/monotrack/blob/main/modified-tracknet/train-hitnet.ipynb
# Licensed under the Adobe Research License - for non-commercial research use only.
import pandas as pd
from skimage.transform import resize

def resample(series, s):
    flatten = False
    if len(series.shape) == 1:
        series.resize((series.shape[0], 1))
        series = series.astype('float64')
        flatten = True
    series = resize(
        series, (int(s * series.shape[0]), series.shape[1]),
    )
    if flatten:
        series = series.flatten()
    return series   

'''
Read the player poses. poses[0] is the bottom player, poses[1] is the top.
'''
col_names = ['frame']
def read_player_poses(input_prefix):
    
    if len(col_names) == 1:
        for i in range(34):
            col_names.append(f'x{i}')

    # print(col_names)

    bottom_player = pd.read_csv(input_prefix + '_bottom.csv', names=col_names, header=0, skip_blank_lines=False)
    top_player = pd.read_csv(input_prefix + '_top.csv', names=col_names, header=0, skip_blank_lines=False)

    bottom_player.drop('frame', axis=1, inplace=True)
    top_player.drop('frame', axis=1, inplace=True)
    bottom_player.fillna(method='bfill', inplace=True)
    bottom_player.fillna(method='ffill', inplace=True)
    top_player.fillna(method='bfill', inplace=True)
    top_player.fillna(method='ffill', inplace=True)
    bottom_player.fillna(0, inplace=True)
    top_player.fillna(0, inplace=True)

    poses = [bottom_player, top_player]
    return poses

'''
The trajectory
'''
class Trajectory(object):
    def __init__(self, filename, interp=True):
        # Get poses and trajectories
        trajectory = pd.read_csv(filename)

        if interp:
            trajectory[trajectory.X == 0] = float('nan')
            trajectory[trajectory.Y == 0] = float('nan')
            trajectory = trajectory.assign(X_pred=trajectory.X.interpolate(method='slinear'))
            trajectory = trajectory.assign(Y_pred=trajectory.Y.interpolate(method='slinear'))

            trajectory.fillna(method='bfill', inplace=True)
            trajectory.fillna(method='ffill', inplace=True)

            Xb, Yb = trajectory.X_pred.tolist(), trajectory.Y_pred.tolist()
        else:
            Xb, Yb = trajectory.X.tolist(), trajectory.Y.tolist()

        self.X = Xb
        self.Y = Yb

def read_trajectory_3d(file_path):
    return pd.read_csv(str(file_path))