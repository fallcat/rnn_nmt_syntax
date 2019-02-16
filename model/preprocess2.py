from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch
import tarfile
from model import SOS_token, EOS_token, MAX_LENGTH, SPAN_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    t = tarfile.open("/mnt/nfs/work1/miyyer/datasets/wmt/wmt_en_de.tar.gz", "r")

    lang1_lines = str(t.extractfile('train.tok.clean.bpe.32000.%s' % (lang1)).\
        read(), 'utf-8').strip().split('\n')
    lang2_lines = str(t.extractfile('train.tok.clean.bpe.32000.%s' % (lang2)).\
        read(), 'utf-8').strip().split('\n')

    # Read the file and split into lines
    # lang1_lines = open('/mnt/nfs/work1/miyyer/datasets/wmt/train.tok.clean.bpe.32000.%s' % (lang1), encoding='utf-8').\
    #     read().strip().split('\n')
    # lang2_lines = open('/mnt/nfs/work1/miyyer/datasets/wmt/train.tok.clean.bpe.32000.%s' % (lang2), encoding='utf-8'). \
    #     read().strip().split('\n')
    # lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
    #     read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s1, s2] for s1, s2 in zip(lang1_lines, lang2_lines)]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def get_vocab():
    t = tarfile.open("/mnt/nfs/work1/miyyer/datasets/wmt/wmt_en_de.tar.gz", "r")
    vocab = str(t.extractfile('vocab.bpe.32000').read(), 'utf-8').strip().split('\n')
    # vocab = open('/mnt/nfs/work1/miyyer/datasets/wmt/vocab.bpe.32000', encoding='utf-8').read().strip().split('\n')
    return vocab

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    vocab = get_vocab()
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for v in vocab:
        input_lang.add_word(v)
        output_lang.add_word(v)
    # for pair in pairs:
    #     print(pair)
    #     input_lang.add_sentence(pair[0])
    #     output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, vocab


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor
