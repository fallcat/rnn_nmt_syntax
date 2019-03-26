import torch
from data.text import TextDataset
import torch.utils.data as Data
# from model import EOS_token, DEVICE, UNK_token
from model import EOS_token, DEVICE, UNK_token



class IWSLTDataset(TextDataset):
    """ CLass that encapsulates IWSLT Dataset"""
    DIR_PATH = "/mnt/nfs/work1/miyyer/wyou/iwslt16_en_de/"
    VOCAB_FILE = 'vocab.bpe.37000'
    SPLITS = {
        'valid': 'dev.tok.bpe.37000',
        'train': 'train.tok.bpe.37000'
    }

    """
    Prepare data from WMTDataset
    """
    def __init__(self, max_length, span_size, split="train", reverse=False):
        super(IWSLTDataset, self).__init__(max_length, span_size, split, reverse)

    def read_vocab(self):
        vocab = open(IWSLTDataset.DIR_PATH + IWSLTDataset.VOCAB_FILE, 'r').read().strip().split('\n')
        for v in vocab:
            self.add_word(v.split()[0])

    def read_langs(self):
        print("Reading lines...")

        en_lines = open(IWSLTDataset.DIR_PATH + '%s.en' % (IWSLTDataset.SPLITS[self.split])).read().strip().split('\n')
        de_lines = open(IWSLTDataset.DIR_PATH + '%s.de' % (IWSLTDataset.SPLITS[self.split])).read().strip().split('\n')

        # Split every line into pairs
        pairs = [[s1, s2] for s1, s2 in zip(de_lines, en_lines)]

        # Reverse pairs, make Lang instances
        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]

        print("Read %s sentence pairs in %s" % (len(pairs), self.split))

        pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))

        self.pairs = pairs
