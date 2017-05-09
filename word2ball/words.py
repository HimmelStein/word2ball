
import os
from nltk.corpus import wordnet as wn

GLOVE50D = '/Users/tdong/data/glove/glove.6B/glove.6B.50d.txt'
GLOVE100D = '/Users/tdong/data/glove/glove.6B/glove.6B.100d.txt'
GLOVE200D = '/Users/tdong/data/glove/glove.6B/glove.6B.200d.txt'
GLOVE300D = '/Users/tdong/data/glove/glove.6B/glove.6B.300d.txt'

VOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.voc.txt'
HYPER_VOC_GLOVE6B = '/Users/tdong/data/glove/glove.6B/glove.6B.hyper.txt'


def create_vocabulary(w2vFile, vocFile):
    """
    :param w2vFile: input pre-trained word2vec file
    :param vocFile: output vocabulary file, one word a line
    :return: vocabulary size
    """
    if not os.path.isfile(w2vFile):
        print('file does not exist:', w2vFile)
        return
    with open(w2vFile, 'r') as w2v:
        wlst = []
        for line in w2v.readlines():
            wlst.append(line.strip().split()[0])
    with open(vocFile, 'w+') as fh:
        fh.write("\n".join(wlst))
    return len(wlst)


def get_voc_list(vocFile):
    """
    :param vocFile: output vocabulary file, one word a line
    :return: voc list
    """
    if not os.path.isfile(vocFile):
        print('file does not exist:', vocFile)
        return []
    with open(vocFile, 'r') as w2v:
        wlst = []
        for line in w2v.readlines():
            wlst.append(line.strip().split()[0])
    return wlst


def create_hypernym_table(vocFile, hyperFile):
    """
    :param vocFile:
    :param hyperFile:
    :return:
    """
    wlst = get_voc_list(vocFile)
    sz = len(wlst)
    hyperLst = []
    for i in range(sz):
        for j in range(i+1, sz):
            for w1 in wn.synsets(wlst[i]):
                for w2 in wn.synsets(wlst[j]):
                    if w2 in w1.lowest_common_hypernyms(w2):
                        print(w1.name().split('.')[0], w2.name().split('.')[0])
                        hyperLst.append(w1.name().split('.')[0]+"       "+w2.name().split('.')[0])
    with open(hyperFile, 'w+') as fh:
        fh.write("\n".join(hyperLst))
    return len(hyperLst)





