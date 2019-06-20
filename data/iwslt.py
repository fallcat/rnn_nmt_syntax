from data.text import SOS
from data.annotated import AnnotatedTextDataset


class IWSLTDataset(AnnotatedTextDataset):
    """ CLass that encapsulates IWSLT Dataset"""
    DIR_PATH = "/mnt/nfs/work1/miyyer/wyou/iwslt/"
    LANGUAGE_PAIR = ('de', 'en')

    VOCAB_FILE = 'vocab.bpe.32000'
    SPLITS = {
        'train': 'train.tok.bpe.32000',
        'valid': 'dev.tok.bpe.32000',
        'test': 'test.tok.bpe.32000'
    }

    """
    Prepare data from IWSLTDataset
    """
    def __init__(self, config, max_length, span_size, filter, split="train", reverse=False, trim=False):
        super(IWSLTDataset, self).__init__(config, max_length, span_size, filter, split, reverse, trim)
