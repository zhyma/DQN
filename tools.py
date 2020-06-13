import cv2
import numpy as np

def pre_process(observation):
    """Process (210, 160, 3) picture into (1, 84, 84)"""
    x_t = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 10, 255, cv2.THRESH_BINARY)
    return x_t


def init_state(processed_obs):
    return np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=0)