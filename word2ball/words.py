
import os
import nltk
import pickle
from nltk.corpus import wordnet as wn
from collections import defaultdict

JJTAGS = ['JJ', 'JJR', 'JJS']
NNTAGS = ['NN', 'NNP', 'NNPS', 'NNS']
VBTAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBD', 'VBZ']


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


def get_content_voc(vocFile, cntVoc):
    """

    :param vocFile:
    :param cntVoc:
    :return:
    """
    cwlst = []
    CTAGS = JJTAGS + NNTAGS + VBTAGS
    with open(cntVoc, 'a') as fh:
        for wd in get_voc_list(vocFile):
            if wd in cwlst:
                continue
            for pos in map(lambda e:e[1], nltk.pos_tag([wd])):
                if pos in CTAGS:
                    cwlst.append(wd)
                    fh.write(wd+"\n")
    return len(cwlst)


def make_synsets_pickle(vocFile, vocPickle):
    pdic = defaultdict(list)
    for wd in get_voc_list(vocFile):
        lst = []
        for w1 in wn.synsets(wd):
            if w1.name().startswith(wd):
                lst.append(w1.name())
        if lst:
            pdic[wd] = lst
    with open(vocPickle, 'wb') as hd:
        pickle.dump(pdic, hd)
    return len(pdic)


def create_hypernym_table(vocFile, vocPickle, hyperFile):
    """
    vlst X vlst search!
    :param vocFile:
    :param hyperFile:
    :return:
    """
    wlst = get_voc_list(vocFile)
    sz = len(wlst)
    with open(vocPickle, 'rb') as hd:
        pdic = pickle.load(hd)
    with open(hyperFile, 'a') as fh:
        for i in range(sz):
            for j in range(i+1, sz):
                if wlst[i] != wlst[j]:
                    for w1 in pdic[wlst[i]]:
                        w1s = wn.synset(w1)
                        for w2 in pdic[wlst[j]]:
                            w2s = wn.synset(w2)
                            if w2s in w1s.lowest_common_hypernyms(w2s):
                                line = w1+"       "+w2+ "\n"
                                fh.write(line)
    return True


def create_hypernym_tree(vocFile, vocPickle, hyperFile):
    """
    :param vocFile:
    :param vocPickle:
    :param hyperFile:
    :return:
    """
    wlst = get_voc_list(vocFile)
    with open(vocPickle, 'rb') as hd:
        pdic = pickle.load(hd)
    wlst1 = []
    hplst = []
    level = 0
    hyperHandle = open(hyperFile, 'a')
    while len(wlst) > 1:
        print("level",level, "lens:", len(wlst), len(wlst1))
        for w in wlst:
            for wsyn in pdic[w]:
                w1 = wn.synset(wsyn)
                for uw in map(lambda e1:e1[0], filter(lambda e:e[1] == 1, w1.hypernym_distances())):
                    if w == w1.name().split('.')[0] and w != uw.name().split('.')[0] \
                            and uw.name().split('.')[0] in wlst \
                            and [wsyn, uw] not in hplst:
                        hplst.append([wsyn, uw])
                        hyperHandle.write("  ".join([wsyn, uw.name(), str(level)]) + "\n")
                        wlst1.append(uw.name().split('.')[0])
        wlst = wlst1.copy()
        wlst1 = []
        level += 1
    hyperHandle.close()
    return len(hplst)








