
import numpy as np
from pylab import *
#from .qsr_util import dis_between, vec_length, circles, DynamicUpdate
from qsr_util import qsr_part_of_characteristic_function, qsr_disconnect_characteristic_function, dis_between, vec_length


def do_func(funcName="part_of"):
    fdic = {
            "part_of": qsr_part_of_characteristic_function,
            "disconnect": qsr_disconnect_characteristic_function
            }
    if funcName in fdic.keys():
        return fdic[funcName]
    else:
        print("unknown qsr reltion:", funcName)
        return -1


def energy_tanh(ball1, ball2, func="part_of"):
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
    if qsrIndicator <= 0:
        return 0
    else:
        return np.tanh(qsrIndicator)


def loss_tanh(ball_w, ball_u, negBalls=[], func="part_of", negFunc="disconnect", rate=0.1):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls: a list of balls as negative sample
    :return:
    """
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_w, ball_u)
    if qsrIndicator <= 0:
        Lw = 0
    else:
        Lw = energy_tanh(ball_u, ball_w, func=func)
    for ball_i in negBalls:
        qsr_func = do_func(funcName="disconnect")
        qsrIndicator = qsr_func(ball_w, ball_u)
        if qsrIndicator >0:
            Lw += energy_tanh(ball_i, ball_w, func="disconnect")
    return Lw

def partial_tanh_derative_lw(ball_w, ball_u, negBalls=[], func="part_of", negFunc="disconnect"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_u, lu, ru = ball_u[:-2], ball_u[-2], ball_u[-1]
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_w, ball_u)

    if qsrIndicator <= 0:
        return 0
    else:
        e_x = np.exp(qsrIndicator)
        e_x_1 = np.exp(-qsrIndicator)
        rlt = 4/(e_x + e_x_1)/(e_x + e_x_1)
        hight = np.dot(alpha_w, alpha_u)
        rlt *= (lw - lu* np.dot(alpha_w, alpha_u))/np.sqrt(lw * lw + lu * lu - np.multiply(2*lu*lw, hight))
        for ball_i in negBalls:
            alpha_i, li, ri = ball_i[:-2], ball_i[-2], ball_i[-1]
            qsr_func = do_func(funcName=negFunc)
            qsrIndicator = qsr_func(ball_i, ball_w)
            if qsrIndicator > 0:
                e_x = np.exp(qsrIndicator)
                e_x_1 = np.exp(-qsrIndicator)
                rlt0 = 4/(e_x + e_x_1)/(e_x + e_x_1)
                hight = np.dot(alpha_w, alpha_i)
                rlt0 *= (lw - li * np.dot(alpha_w, alpha_i)) / np.sqrt(lw * lw + li * li - np.multiply(2*li*lw, hight))
                rlt -= rlt0
    return rlt


def partial_tanh_derative_rw(ball_w, ball_u, negBalls=[], func="part_of", negFunc="disconnect"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_u, lu, ru = ball_u[:-2], ball_u[-2], ball_u[-1]
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_w, ball_u)

    if qsrIndicator <= 0:
        return 0
    else:
        e_x = np.exp(qsrIndicator)
        e_x_1 = np.exp(-qsrIndicator)
        rlt = - 4 / (e_x_1 + e_x) / (e_x + e_x_1)
        for ball_i in negBalls:
            alpha_i, li, ri = ball_i[:-2], ball_i[-2], ball_i[-1]
            qsr_func = do_func(funcName=negFunc)
            qsrIndicator = qsr_func(ball_i, ball_w)
            if qsrIndicator > 0:
                e_x = np.exp(qsrIndicator)
                e_x_1 = np.exp(-qsrIndicator)
                rlt0 = 4 / (e_x + e_x_1) / (e_x + e_x_1)
                rlt += rlt0
    return rlt


def partial_tanh_derative_lu(ball_w, ball_u, func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_u, lu, ru = ball_u[:-2], ball_u[-2], ball_u[-1]
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_w, ball_u)

    if qsrIndicator <= 0:
        return 0
    else:
        e_x = np.exp(qsrIndicator)
        e_x_1 = np.exp(-qsrIndicator)
        rlt = 4 / (e_x + e_x_1) / (e_x + e_x_1)
        hight = np.dot(alpha_w, alpha_u)
        rlt *= (lu - lw * np.dot(alpha_w, alpha_u)) / np.sqrt(lw * lw + lu * lu - np.multiply(2 * lu * lw, hight))
    return rlt


def partial_tanh_derative_ru(ball_w, ball_u, func="part_of"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_w, ball_u)

    if qsrIndicator <= 0:
        return 0
    else:
        e_x = np.exp(qsrIndicator)
        e_x_1 = np.exp(-qsrIndicator)
        rlt = 4 / (e_x + e_x_1) / (e_x + e_x_1)
    return rlt


def partial_tanh_derative_li(ball_w, ball_i, func="part_of"):
    """

    :param ball_w:
    :param ball_i:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_i, li, ri = ball_i[:-2], ball_u[-2], ball_u[-1]
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_w, ball_i)

    if qsrIndicator <= 0:
        return 0
    else:
        e_x = np.exp(qsrIndicator)
        e_x_1 = np.exp(-qsrIndicator)
        rlt = 4 / (e_x + e_x_1) / (e_x + e_x_1)
        hight = np.dot(alpha_w, alpha_i)
        rlt *= (lw * np.dot(alpha_w, alpha_i) - li) / np.sqrt(lw * lw + li * li - np.multiply(2 * li * lw, hight))
    return rlt


def partial_tanh_derative_ri(ball_w, ball_i, func="part_of"):
    """

    :param ball_w:
    :param ball_i:
    :param func:
    :return:
    """
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_w, ball_i)

    if qsrIndicator <= 0:
        return 0
    else:
        e_x = np.exp(qsrIndicator)
        e_x_1 = np.exp(-qsrIndicator)
        rlt = 4 / (e_x + e_x_1) / (e_x + e_x_1)
    return rlt


def update_balls_tanh(ball_w, ball_u, negBalls=[], func="part_of", negFunc="disconnect", rate=0.1, alphaL=5, alphaR=5):
    """
    ball_w shall contain ball_u, and disconnects from balls in negBalls
    that is, ball_u is 'part of' ball_w
    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return: new ball_w, ball_u, negBalls=[]
    """
    dL_dlw = partial_tanh_derative_lw(ball_w, ball_u, negBalls=negBalls, func=func, negFunc=negFunc)
    ball_w[-2] -= dL_dlw * rate
    if ball_w[-2] < 0:
        ball_w[-2] += dL_dlw * rate
        ball_u[-2] += dL_dlw * rate


    dL_drw = partial_tanh_derative_rw(ball_w, ball_u, negBalls=negBalls, func=func, negFunc=negFunc)
    ball_w[-1] -= dL_drw * rate
    if ball_w[-1] < 0:
        ball_w[-1] += dL_drw * rate
        ball_u[-1] += dL_drw * rate

    dL_dlu = partial_tanh_derative_lu(ball_w, ball_u, func=func)
    ball_u[-2] -= dL_dlu * rate
    if ball_u[-2] < 0:
        ball_w[-2] += dL_dlu * rate
        ball_u[-2] += dL_dlu * rate

    dl_dru = partial_tanh_derative_ru(ball_w, ball_u, func=func)
    ball_u[-1] -= dl_dru * rate
    if ball_u[-1] < 0:
        ball_w[-1] += dl_dru * rate
        ball_u[-1] += dl_dru * rate

    for ball_i in negBalls:
        dL_dli = partial_tanh_derative_li(ball_w, ball_i, func=func)
        ball_i[-2] -= dL_dli * rate
        if ball_i[-2] < 0:
            ball_i[-2] += dL_dli * rate
            ball_w[-2] += dL_dli * rate

        dl_dri = partial_tanh_derative_ri(ball_w, ball_i, func=func)
        ball_i[-1] -= dl_dri * rate
        if ball_i[-1] < 0:
            ball_i[-1] += dl_dri * rate
            ball_w[-1] -= dl_dri * rate


def train_tanh_2D(ball_w, ball_u, negBalls=[], func="part_of", negFunc="disconnect",rate=0.1):
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
    clos = loss_tanh(ball_w, ball_u, negBalls=negBalls, func=func, negFunc=negFunc,rate=rate)
    while clos > 0:
        update_balls_tanh(ball_w, ball_u, negBalls=negBalls, func=func, negFunc=negFunc, rate=rate)


ball_n2 = [np.cos(np.pi/4), np.sin(np.pi/4), 4, 3]
ball_n1 = [np.cos(np.pi/6), np.sin(np.pi/6), 5, 3]
ball_w = [np.cos(np.pi/2), np.sin(np.pi/2), 10, 1]
ball_u = [np.cos(np.pi/2), np.sin(np.pi/2), 1, 1]
negBall = [ball_n1, ball_n2]
