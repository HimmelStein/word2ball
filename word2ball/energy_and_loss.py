
import numpy as np
from pylab import *
#from .qsr_util import dis_between, vec_length, circles, DynamicUpdate
from .qsr_util import dis_between, vec_length, circles, DynamicUpdate


def do_func(funcName="part_of"):
    fdic = {
            "part_of": qsr_part_of_characteristic_function
            }
    if funcName in fdic.keys():
        return fdic[funcName]
    else:
        print("unknown qsr reltion:", funcName)
        return -1


def qsr_part_of_characteristic_function(ball1, ball2):
    """
    ball1, ball2 are vectors in the form of  alpha1, l1, r1 = ball1[:-2], ball1[-2], ball1[-1]
    alpha1 is a unit point on the unit ball!
    distance between the center points of ball1 and ball2 + radius of ball1 - radius of ball2
    return <=0 ball1 part of ball2
           > 0 ball1 not part of ball2

    :param ball1:
    :param ball2:
    :return: R
    """
    alpha1, l1, r1 = ball1[:-2], ball1[-2], ball1[-1]
    alpha2, l2, r2 = ball2[:-2], ball2[-2], ball2[-1]
    return dis_between(np.multiply(l1, alpha1), np.multiply(l2, alpha2)) + r1 - r2


def energy(ball1, ball2, func="part_of"):
    """
    compute the energy of ball1 being part of ball2

    ball1 \part_of ball2 = distance

    :param ball1:
    :param ball2:
    :return:
    """
    qsr_func = do_func(funcName=func)
    assert qsr_func != -1
    qsrIndicator = qsr_func(ball1, ball2)
    return 1/(1 + np.exp(-qsrIndicator))


def loss(ball_w, ball_u, negBalls=[], func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls: a list of balls as negative sample
    :return:
    """
    Lw = np.log(energy(ball_u, ball_w, func=func))
    for ball_i in negBalls:
        Lw += np.log(1 - energy(ball_i, ball_w, func=func))
    return Lw

def borderline_loss(ball_w, ball_u, negBalls=[], func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    Lw = np.log(0.5)
    for ball_i in negBalls:
        Lw += np.log(0.5)
    return Lw


def partial_derative_lw(ball_w, ball_u, negBalls=[], func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_u, lu, ru = ball_u[:-2], ball_u[-2], ball_u[-1]
    hight = np.dot(alpha_w, alpha_u)
    result = (1 - energy(ball_u, ball_w, func=func)) * (lw - np.multiply(lu, hight))\
             / np.sqrt(lw * lw + lu * lu - np.multiply(2*lu*lw, hight))
    for ball_i in negBalls:
        alpha_i, li, ri = ball_i[:-2], ball_i[-2], ball_i[-1]
        hight = np.dot(alpha_i, alpha_w)
        result -= energy(ball_i, ball_w, func=func) * (lw - np.multiply(li, hight))\
             / np.sqrt(lw * lw + li * li - np.multiply(2*li*lw, hight))
    return result


def partial_derative_rw(ball_w, ball_u, negBalls=[], func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    result = energy(ball_u, ball_w, func=func) - 1
    for ball_i in negBalls:
        result += energy(ball_i, ball_w, func=func)
    return result


def partial_derative_lu(ball_w, ball_u, negBalls=[], func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_u, lu, ru = ball_u[:-2], ball_u[-2], ball_u[-1]
    hight = np.dot(alpha_w, alpha_u)
    result = (1 - energy(ball_u, ball_w, func=func)) * (lu - np.multiply(lw, hight)) \
             / np.sqrt(lw * lw + lu * lu - np.multiply(2 * lu * lw, hight))
    return result


def partial_derative_ru(ball_w, ball_u, negBalls=[], func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    return 1 - energy(ball_u, ball_w, func=func)


def partial_derative_li(ball_w, ball_i, func="part_of"):
    """

    :param ball_w:
    :param ball_i:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_i, li, ri = ball_i[:-2], ball_i[-2], ball_i[-1]
    hight = np.dot(alpha_i, alpha_w)
    result = energy(ball_i, ball_w, func=func) * (np.multiply(lw, hight) - li)\
             / np.sqrt(lw * lw + li * li - np.multiply(2*li*lw, hight))
    return result


def partial_derative_ri(ball_w, ball_i, func="part_of"):
    """

    :param ball_w:
    :param ball_i:
    :param func:
    :return:
    """
    return - energy(ball_i, ball_w, func=func)


def update_balls(ball_w, ball_u, negBalls=[], func="part_of", rate=0.1):
    """
    ball_w shall contain ball_u, and disconnects from balls in negBalls
    that is, ball_u is 'part of' ball_w
    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return: new ball_w, ball_u, negBalls=[]
    """
    dL_dlw = partial_derative_lw(ball_w, ball_u, negBalls=[], func=func)
    ball_w[-2] -= dL_dlw * rate

    dL_drw = partial_derative_rw(ball_w, ball_u, negBalls=[], func=func)
    ball_w[-1] -= dL_drw * rate

    dL_dlu = partial_derative_lu(ball_w, ball_u, negBalls=[], func=func)
    ball_u[-2] -= dL_dlu * rate

    dl_dru = partial_derative_ru(ball_w, ball_u, negBalls=[], func=func)
    ball_u[-1] -= dl_dru * rate

    for ball_i in negBalls:
        dL_dli = partial_derative_li(ball_w, ball_i, func=func)
        ball_i[-2] -= dL_dli * rate

        dl_dri = partial_derative_ri(ball_w, ball_i, func=func)
        ball_i[-1] -= dl_dri * rate


def train_2D(ball_w, ball_u, negBalls=[], func="part_of", rate=0.1):
    """
    balls are two dimensional!
    training balls: ball_u shall be part of ball_w, balls in negBalls shall
    disconnect from ball_w
    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :param rate:
    :return:
    """
    clos = loss(ball_w, ball_u, negBalls=negBalls, func=func)
    minlos = borderline_loss(ball_w, ball_u, negBalls=negBalls, func=func)
    count, delta = 0, clos - minlos

    import matplotlib.pyplot as plt
    plt.ion()
    d = DynamicUpdate()

    fig = figure(figsize=(-10, 100))
    ax = subplot(aspect='equal')

    cball_w = circles(ball_w[0] * ball_w[-2], ball_w[1]*ball_w[-2], ball_w[-1],
                      c='r')

    cball_u = circles(ball_u[0] * ball_u[-2], ball_u[1] * ball_u[-2], ball_u[-1],
                      c='b')
    fig.show()

    while delta > 0:
        update_balls(ball_w, ball_u, negBalls=[], func="part_of", rate=0.1)



ball_n2 = [np.cos(np.pi/4), np.sin(np.pi/4), 4, 3]
ball_n1 = [np.cos(np.pi/6), np.sin(np.pi/6), 5, 3]
ball_w = [np.cos(np.pi/2), np.sin(np.pi/2), 10, 1]
ball_u = [np.cos(np.pi/2), np.sin(np.pi/2), 1, 1]
negBall = [ball_n1, ball_n2]


