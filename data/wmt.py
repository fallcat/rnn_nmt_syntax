import tarfile
from data.text import TextDataset


class WMTDataset(TextDataset):
    """ CLass that encapsulates WMT Dataset"""
    TAR_PATH = "/mnt/nfs/work1/miyyer/datasets/wmt/wmt_en_de.tar.gz"
    VOCAB_FILE = 'vocab.bpe.32000'
    SPLITS = {
        'valid': 'newstest2013.tok',
        'test': 'newstest2014.tok',
        'train': 'train.tok.clean'
    }

    """
    Prepare data from WMTDataset
    """
    def __init__(self, max_length, span_size, sort, filter, split="train", reverse=False):
        super(WMTDataset, self).__init__(max_length, span_size, sort, filter, split, reverse)

    def read_vocab(self):
        t = tarfile.open(WMTDataset.TAR_PATH, "r")
        vocab = str(t.extractfile(WMTDataset.VOCAB_FILE).read(), 'utf-8').strip().split('\n')
        for v in vocab:
            self.add_word(v)

    def read_langs(self):
        print("Reading lines...")

        t = tarfile.open(self.TAR_PATH, "r")

        en_lines = str(t.extractfile('%s.bpe.32000.en' % (WMTDataset.SPLITS[self.split])).read(), 'utf-8').strip().split('\n')
        de_lines = str(t.extractfile('%s.bpe.32000.de' % (WMTDataset.SPLITS[self.split])).read(), 'utf-8').strip().split('\n')

        # Split every line into pairs
        pairs = [[s1, s2] for s1, s2 in zip(de_lines, en_lines)]

        # Reverse pairs, make Lang instances
        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]

        print("Read %s sentence pairs in %s" % (len(pairs), self.split))
        if self.filter:
            pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        if self.sort:
            pairs = sorted(pairs, key=lambda x: len(x[1]))

        self.pairs = pairs