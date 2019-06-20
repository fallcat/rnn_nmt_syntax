import re
import os
from data.text import TextDataset


class TextAnnotation(enum.Enum):
    ''' An enumeration of text annotation types '''
    NONE = ('', 'bpe.32000.bin', 'bpe.32000')
    CONSTITUENCY_PARSE = ('parsed', '{lang}.parse', 'parse.fully.upto.span{span}')
    PARSE_SPANS = ('spans', '{lang}.parse', 'bpe.32000')

    def __init__(self, identifier, ext, vocab_ext):
        ''' Initialize the text annotation '''
        self.ext = ext
        self.vocab_ext = vocab_ext
        self.identifier = identifier

    def data_path(self, split, directory, **kwargs):
        ''' Return the data path '''
        data_ext = self.ext.format(**kwargs)
        return os.path.join(directory, f'{split}.{data_ext}')

    def vocab_path(self, directory, **kwargs):
        ''' Return the vocab path '''
        vocab_ext = self.vocab_ext.format(**kwargs)
        return os.path.join(directory, f'vocab.{vocab_ext}')


class AnnotatedTextDataset(TextDataset):
    ''' Class that encapsulates an annotated text dataset '''
    NAME = ''
    LANGUAGE_PAIR = ('en', 'en')

    URLS = []
    RAW_SPLITS = {}
    SPLITS = {
        'train': 'train.tok',
        'valid': 'valid.tok',
        'dev': 'valid.tok',
        'test': 'test.tok'
    }

    IGNORE_REGEX_LIST = []
    SEGMENT_REGEX = re.compile(r'<\s*seg\s+id\s*=\s*"\d+"\s*>\s*(.+)\s*<\s*/\s*seg\s*>')

    def __init__(self, max_length, span_size, filter, split="train", reverse=False, trim=False, annotation=TextAnnotation.NONE):
        super(AnnotatedTextDataset, self).__init__(max_length, span_size, filter, split, reverse, trim)

        self.segmenters = []
        self.annotation = annotation

    @classmethod
    def name(cls, reverse=False, annotation=TextAnnotation.NONE):
        ''' Return a name for the dataset given the passed in configuration '''
        config = [cls.NAME] + list(reversed(cls.LANGUAGE_PAIR) if reverse else cls.LANGUAGE_PAIR)
        if annotation.identifier:
            config += [annotation.identifier]

        return '_'.join(config)