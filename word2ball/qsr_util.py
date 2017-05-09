
import numpy as np


def dis_between_ball_centers(ball1, ball2):
    return np.sqrt(np.dot(ball1[:-1] - ball2[:-1], ball1[:-1] - ball2[:-1]))

def get_qsr(ball1, ball2):
    """
    compute the relation between ball1 and ball2
    :param ball1:
    :param ball2:
    :return: ['part_of']
    """
    r1 = ball1[-1]
    r2 = ball2[-1]
    d = dis_between_ball_centers(ball1, ball2)
    if r1 + d <= r2:
        return 'part_of'
    if r1 + r2 < d:
        return 'disconnect'
    return 'unknown'