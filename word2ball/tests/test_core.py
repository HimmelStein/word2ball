# -*- coding: utf-8 -*-

from .context import word2ball
import numpy as np
import unittest


class TestCore(unittest.TestCase):
    """Basic test cases."""

    def test_check_qsr(self):
        ball1 = np.array([0,0,0,0, 0.3])
        ball2 = np.array([2,2,2,2, 0.1])
        qsr = word2ball.get_qsr(ball1, ball2)
        print(qsr)
        assert qsr == 'disconnect'

    def test_distance(self):
        ball1 = np.array([0, 0, 0, 0, 0.3])
        ball2 = np.array([2, 2, 2, 2, 0.1])
        dis = word2ball.dis_between_ball_centers(ball1, ball2)
        print('dis', dis)
        assert dis == 4.0

    def test_make_part_of(self):
        ball1 = np.array([0, 0, 0.3])
        ball2 = np.array([2, 2, 0.1])
        ball1, ball2, result = word2ball.update_to_part_of(ball1, ball2)
        print("ball1", ball1)
        print("ball2", ball2)
        assert result

    def test_make_part_of_dev(self):
        ball1 = np.array([0, 0, 0.3])
        ball2 = np.array([2, 2, 0.1])
        ball1, ball2, result = word2ball.update_to_part_of_by_dev(ball1, ball2)
        print("ball1", ball1)
        print("ball2", ball2)
        print(result)
        assert result

    def test_make_disconnect(self):
        ball1 = np.array([0, 0, 3.0])
        ball2 = np.array([2, 2, 10.0])
        ball1, ball2, result = word2ball.update_to_disconnected(ball1, ball2)
        print("ball1", ball1)
        print("ball2", ball2)
        assert result

if __name__ == '__main__':
    unittest.main()