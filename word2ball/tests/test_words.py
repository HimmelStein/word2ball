# -*- coding: utf-8 -*-

from .context import word2ball
import numpy as np
import unittest


GLOVE300D = '/Users/tdong/data/glove/glove.6B/glove.6B.300d.txt'
VOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.voc.txt'
HYPER_VOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.hyper.txt'


class TestWords(unittest.TestCase):

    def test_create_voc(self):
        vsize = word2ball.create_vocabulary(GLOVE300D, VOC_GLOVE6B)
        assert vsize == 400000

    def test_create_hyper_pair(self):
        hsize = word2ball.create_hypernym_table(VOC_GLOVE6B, HYPER_VOC_GLOVE6B)
        assert hsize > 0

if __name__ == '__main__':
    unittest.main()