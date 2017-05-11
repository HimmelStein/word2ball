# -*- coding: utf-8 -*-

from .context import word2ball
import numpy as np
import unittest

GLOVE50D = '/Users/tdong/data/glove/glove.6B/glove.6B.50d.txt'
GLOVE100D = '/Users/tdong/data/glove/glove.6B/glove.6B.100d.txt'
GLOVE200D = '/Users/tdong/data/glove/glove.6B/glove.6B.200d.txt'
GLOVE300D = '/Users/tdong/data/glove/glove.6B/glove.6B.300d.txt'
VOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.voc.txt'
CVOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.cntvoc.txt'
HYPER_VOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.hyper.txt'
LHYPER_VOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.lev_hyper.txt'
PICKLE_SYNSET = '/Users/tdong/data/glove/glove.6B/glove.6B.synset.pickle'

class TestWords(unittest.TestCase):

    @unittest.skip('')
    def test_create_voc(self):
        vsize = word2ball.create_vocabulary(GLOVE300D, VOC_GLOVE6B)
        assert vsize == 400000

    @unittest.skip('')
    def test_create_content_voc(self):
        vsize = word2ball.get_content_voc(VOC_GLOVE6B, CVOC_GLOVE6B)
        assert vsize > 0

    @unittest.skip('')
    def test_make_synset_pickle(self):
        print("testing make_synset_pickle...\n")
        psize = word2ball.make_synsets_pickle(CVOC_GLOVE6B, PICKLE_SYNSET)
        assert psize > 0

    @unittest.skip('')
    def test_create_hyper_pair(self):
        assert word2ball.create_hypernym_table(CVOC_GLOVE6B, PICKLE_SYNSET, HYPER_VOC_GLOVE6B)

    def test_create_hyper_pair(self):
        hsize = word2ball.create_hypernym_tree(CVOC_GLOVE6B, PICKLE_SYNSET, LHYPER_VOC_GLOVE6B)
        assert hsize > 0

if __name__ == '__main__':
    unittest.main()