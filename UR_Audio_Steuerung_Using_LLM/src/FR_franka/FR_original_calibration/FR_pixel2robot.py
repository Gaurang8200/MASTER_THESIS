# MO_Changes
# Standard Bibliotheken
import json

# Externe Bibliotheken
# from unittest import result
import numpy as np

# Eigene Module
from . import FR_function_pool as fp


def pixel2robot(cordx: float, cordy: float, nr: int) -> tuple[float, float]:
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    input_name = "output_wp2camera.json"
    input_name2 = "output_c2f.json"
    input_name3 = "robot_poses.json"
    x = cordx
    y = cordy
    nr = nr
        
    pixel_coords = np.array([[x], [y], [1]])
    tvec, rvec, camera_matrix, dist = fp.read_wp2c(input_name)
    fTc = fp.read_c2f(input_name2)
    bTf_i, _ = fp.read_bTf(input_name3)
    #print("fTc: \n", fTc)
    #print("bTf_i: \n", bTf_i[nr])

    result, bTc, rot_c2p, trans_c2p, bTp, Spitze_mat = fp.calc_pixel2robot(tvec, rvec, camera_matrix, bTf_i, fTc, pixel_coords, nr, False)#printout results False
    #print(result.shape)
    x_robot = float(result[0, 0])
    y_robot = float(result[1, 0])
    print(f"pixel2robot result: x={x_robot:.2f} mm, y={y_robot:.2f} mm")

    return x_robot, y_robot

