import re
import os
import enum
from data.text import TextDataset, SOS
from data import preprocess


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
    LANGUAGE_PAIR = ('de', 'en')

    DIR_PATH = "/mnt/nfs/work1/miyyer/wyou/iwslt/"

    VOCAB_FILE = 'vocab.bpe.32000'

    URLS = []
    RAW_SPLITS = {}
    SPLITS = {
        'train': 'train.tok.bpe.32000',
        'valid': 'dev.tok.bpe.32000',
        'test': 'test.tok.bpe.32000'
    }

    IGNORE_REGEX_LIST = []
    SEGMENT_REGEX = re.compile(r'<\s*seg\s+id\s*=\s*"\d+"\s*>\s*(.+)\s*<\s*/\s*seg\s*>')

    def __init__(self, config, max_length, span_size, filter, split="train", reverse=False, trim=False, annotation=TextAnnotation.NONE):
        super(AnnotatedTextDataset, self).__init__(max_length, span_size, filter, split, reverse, trim)

        self.segmenters = []
        self.annotation = annotation
        self.preprocess_directory = config['preprocess_directory']
        self.config = config

    @classmethod
    def name(cls, reverse=False, annotation=TextAnnotation.NONE):
        ''' Return a name for the dataset given the passed in configuration '''
        config = [cls.NAME] + list(reversed(cls.LANGUAGE_PAIR) if reverse else cls.LANGUAGE_PAIR)
        if annotation.identifier:
            config += [annotation.identifier]

        return '_'.join(config)

    @property
    def source_language(self):
        ''' Return the source language '''
        return type(self).LANGUAGE_PAIR[1 if self.reverse else 0]

    @property
    def target_language(self):
        ''' Return the target language '''
        return type(self).LANGUAGE_PAIR[0 if self.reverse else 1]

    @property
    def base_data_path(self):
        ''' Get the path of the processed data file '''
        return TextAnnotation.NONE.data_path(
            type(self).SPLITS[self.split],
            self.preprocess_directory
        )

    @property
    def source_annotation_data_path(self):
        ''' Get the path of the processed data file '''
        return self.annotation.data_path(
            type(self).SPLITS[self.split],
            self.preprocess_directory,
            lang=self.source_language
        )

    @property
    def target_annotation_data_path(self):
        ''' Get the path of the processed data file '''
        return self.annotation.data_path(
            type(self).SPLITS[self.split],
            self.preprocess_directory,
            lang=self.target_language
        )

    @property
    def data_paths(self):
        ''' Get the list of data files '''
        return {
            self.base_data_path,
            self.source_annotation_data_path,
            self.target_annotation_data_path
        }

    @property
    def base_vocab_path(self):
        ''' Get the path of the vocab file '''
        return TextAnnotation.NONE.vocab_path(
            self.preprocess_directory,
            span=self.config['span_size']
        )

    @property
    def annotation_vocab_path(self):
        ''' Get the path of the annotation specific vocab file '''
        return self.annotation.vocab_path(
            self.preprocess_directory,
            span=self.config['span_size']
        )

    @property
    def constituent_vocab_path(self):
        ''' Get the path of the constituent vocab file '''
        return TextAnnotation.CONSTITUENCY_PARSE.vocab_path(
            self.preprocess_directory,
            span=self.config['span_size']
        )

    @property
    def vocab_paths(self):
        ''' Get the list of vocab files '''
        return {self.base_vocab_path, self.annotation_vocab_path}

    def preprocess(self):
        ''' Do any data preprocessing if needed '''
        if (
                all(os.path.exists(p) for p in self.data_paths) and
                all(os.path.exists(p) for p in self.vocab_paths)
        ):
            return

        if not os.path.exists(self.preprocess_directory):
            os.makedirs(self.preprocess_directory)

        # self.download_and_extract()
        # self.preprocess_raw()
        #
        # # Make sure we have loaded the vocab
        # self.load_vocab(preprocessing=True)
        #
        # split_filename = type(self).SPLITS[self.split]
        # self.preprocess_bpe(split_filename)

        if self.annotation in (
                TextAnnotation.PARSE_SPANS,
                TextAnnotation.CONSTITUENCY_PARSE
        ):
            base_annotation_id = len(self.index2word)
            for filename in type(self).SPLITS.values():
                self.preprocess_parse(filename)

            if not os.path.exists(self.constituent_vocab_path):
                with open(self.constituent_vocab_path, 'wt') as file:
                    file.write('\n'.join([
                        self.index2word[annotation_id]
                        for annotation_id in range(base_annotation_id, len(self.index2word))
                    ]))

    def preprocess_parse(self, filename):
        ''' Preprocess the parse data '''
        base_path = os.path.join(self.preprocess_directory, f'{filename}')
        tokenized_bpe_path = f'{base_path}.bpe.32000'

        source_path = f'{base_path}.{self.source_language}.parse'
        if not os.path.exists(source_path):
            preprocess.parse(
                f'{tokenized_bpe_path}.{self.source_language}',
                source_path,
                self.config['preprocess_buffer_size']
            )

        target_path = f'{base_path}.{self.target_language}.parse'
        if not os.path.exists(target_path):
            preprocess.parse(
                f'{tokenized_bpe_path}.{self.target_language}',
                target_path,
                self.config['preprocess_buffer_size']
            )

        if os.path.exists(self.constituent_vocab_path):
            return

        bpe_path = os.path.join(self.preprocess_directory, 'bpe.32000')
        self.segmenters = [
            preprocess.ParseSegmenter(
                bpe_path, span, self.config.max_span, self.config.randomize_chunks
            )
            for span in range(1, self.config.span + 1)
        ]

        vocab = preprocess.get_parse_vocab(
            f'{base_path}.{self.source_language}.parse',
            self.segmenters, self.config['preprocess_buffer_size']
        )
        vocab.update(preprocess.get_parse_vocab(
            f'{base_path}.{self.target_language}.parse',
            self.segmenters, self.config['preprocess_buffer_size']
        ))

        for token in vocab:
            if token not in self.word2index:
                self.word2index[token] = len(self.index2word)
                self.index2word.append(token)

    def load(self, preprocess=True):
        if preprocess:
            self.preprocess()
        return self

    def read_vocab(self):
        vocab = open(self.DIR_PATH + self.VOCAB_FILE, 'r').read().strip().split('\n')
        for v in vocab:
            self.add_word(v.split()[0])

    def read_langs(self):
        print("Reading lines...")
        if self.split != "test":
            l2_lines = open(self.DIR_PATH + '%s.' + self.LANGUAGE_PAIR[1] % (self.SPLITS[self.split])).read().strip().split('\n')
        l1_lines = open(self.DIR_PATH + '%s.' + self.LANGUAGE_PAIR[0] % (self.SPLITS[self.split])).read().strip().split('\n')

        # Split every line into pairs
        if self.split != "test":
            if self.reverse:
                pairs = [[s2, (SOS + ' ') * self.span_size + s1] for s1, s2 in zip(l1_lines, l2_lines)]
            else:
                pairs = [[s1, (SOS + ' ') * self.span_size + s2] for s1, s2 in zip(l1_lines, l2_lines)]
        else:
            pairs = [[s, ""]for s in l1_lines]

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
