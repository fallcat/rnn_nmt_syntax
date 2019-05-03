from data.text import TextDataset, SOS


class IWSLTDataset(TextDataset):
    """ CLass that encapsulates IWSLT Dataset"""
    DIR_PATH = "/mnt/nfs/work1/miyyer/wyou/iwslt16_en_de/"
    VOCAB_FILE = 'vocab.bpe.37000'
    SPLITS = {
        'valid': 'dev.tok.bpe.37000',
        'train': 'train.tok.bpe.37000'
    }

    """
    Prepare data from IWSLTDataset
    """
    def __init__(self, max_length, span_size, filter, split="train", reverse=False, trim=False):
        super(IWSLTDataset, self).__init__(max_length, span_size, filter, split, reverse, trim)

    def read_vocab(self):
        vocab = open(IWSLTDataset.DIR_PATH + IWSLTDataset.VOCAB_FILE, 'r').read().strip().split('\n')
        for v in vocab:
            self.add_word(v.split()[0])

    def read_langs(self):
        print("Reading lines...")

        en_lines = open(IWSLTDataset.DIR_PATH + '%s.en' % (IWSLTDataset.SPLITS[self.split])).read().strip().split('\n')
        de_lines = open(IWSLTDataset.DIR_PATH + '%s.de' % (IWSLTDataset.SPLITS[self.split])).read().strip().split('\n')

        # Split every line into pairs
        if self.reverse:
            pairs = [[s2, (SOS + ' ') * self.span_size + s1] for s1, s2 in zip(de_lines, en_lines)]
        else:
            pairs = [[s1, (SOS + ' ') * self.span_size + s2] for s1, s2 in zip(de_lines, en_lines)]

        # print("pairs[0]", pairs[0])

        print("Read %s sentence pairs in %s" % (len(pairs), self.split))

        if self.filter:
            pairs = self.filter_pairs(pairs)

        if self.trim:
            print("Trimmed to max_length")
            pairs = self.trim_pairs(pairs)

        print("Trimmed to %s sentence pairs" % len(pairs))
        # print(len(sorted(pairs, key=lambda x: len(x[1]))[-1][0].split(" ")))

        self.pairs = pairs
