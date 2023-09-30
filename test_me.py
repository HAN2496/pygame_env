import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

CARX = 3
CARY = 0
CARYAW = 0
def make_trajectory(action):
    def rotate(x1, y1, angle):
        x1 -= CARX
        y1 -= CARY
        x_new = x1 * math.cos(angle) - y1 * math.sin(angle)
        y_new = x1 * math.sin(angle) + y1 * math.cos(angle)
        x_new += CARX
        y_new += CARY
        return (x_new, y_new)

    traj = []
    for idx, angle in enumerate(action):
        traj_before_rotate = np.array([CARX + idx * 6, CARY])
        traj.append(rotate(traj_before_rotate[0], traj_before_rotate[1], angle))

    traj_interpolated1 = np.array([(traj[0][0] + traj[1][0]) / 2, (traj[0][1] + traj[1][1]) / 2])
    traj_interpolated2 = np.array([((traj[1][0] + traj[2][0]) / 2, (traj[1][1] + traj[2][1]) / 2)])
    traj = np.insert(traj, 1, traj_interpolated1, axis=0)
    traj = np.insert(traj, 3, traj_interpolated2, axis=0)
    return traj

def interpolate_arr(arr, interval=0.01):
    interpolated_arr = []
    for idx, (x, y) in enumerate(arr):
        x1 = arr[idx+1][0]
        x_new = np.arange(x, x1, interval)
        y_new = np.interp(x_new, [x, x1])
    pass


def calculate_dev(traj):
    f = interp1d(traj[:, 0], traj[:, 1])
    xnew = np.arange(traj[0][0], traj[-1][0], 0.01)
    ynew = f(xnew)
    plt.scatter(xnew, ynew)
    plt.show()
    arr = np.array(list(zip(xnew, ynew)))
    distances = np.sqrt(np.sum((arr - [CARX, CARY]) ** 2, axis=1))
    dist_index = np.argmin(distances)
    devDist = distances[dist_index]
    devAng1 = (arr[dist_index + 1][1] - arr[dist_index][1]) / (arr[dist_index + 1][0] - arr[dist_index][0])
    devAng2 = (arr[dist_index][1] - arr[dist_index - 1][1]) / (arr[dist_index][0] - arr[dist_index - 1][0])
    devAng = - np.arctan((devAng1 + devAng2) / 2) - CARYAW
    return devDist, devAng

action = [0, 0, -0.2]
traj_abs = make_trajectory(action)
print(traj_abs)
print(np.sqrt((traj_abs[-1][0] - 3) ** 2 + traj_abs[-1][1] ** 2))
dev = calculate_dev(traj_abs)